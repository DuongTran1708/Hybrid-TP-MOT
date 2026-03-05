from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from loguru import logger
from thermal_pedestrian.core.factory.builder import IDENTIFICATIONS
from thermal_pedestrian.trackers.hybridtpmot.utils import matching
from thermal_pedestrian.trackers.hybridtpmot.utils.gmc import GMC
from thermal_pedestrian.trackers.hybridtpmot.utils.kalman_filter import KalmanFilterXYWH, KalmanFilterXYAH

from ultralytics.trackers.basetrack import TrackState, BaseTrack
# from ultralytics.trackers.bot_sort import ReID as YOLO_ReID
from ultralytics.utils.ops import xywh2ltwh

class HTPMTrack(BaseTrack):
	"""An extended version of the STrack class for YOLO, adding object tracking features.
	Single object tracking representation that uses Kalman filtering for state estimation

	This class extends the STrack class to include additional functionalities for object tracking, such as feature
	smoothing, Kalman filter prediction, and reactivation of tracks.

	Attributes:
		shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of HTPMTrack.
		smooth_feat (np.ndarray): Smoothed feature vector.
		curr_feat (np.ndarray): Current feature vector.
		features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
		alpha (float): Smoothing factor for the exponential moving average of features.
		mean (np.ndarray): The mean state of the Kalman filter.
		covariance (np.ndarray): The covariance matrix of the Kalman filter.

	Methods:
		update_features: Update features vector and smooth it using exponential moving average.
		predict: Predict the mean and covariance using Kalman filter.
		re_activate: Reactivate a track with updated features and optionally new ID.
		update: Update the track with new detection and frame ID.
		tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
		multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
		convert_coords: Convert tlwh bounding box coordinates to xywh format.
		tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

	Examples:
		Create a HTPMTrack instance and update its features
		>>> bo_track = HTPMTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
		>>> bo_track.predict()
		>>> new_track = HTPMTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
		>>> bo_track.update(new_track, frame_id=2)
	"""

	shared_kalman = KalmanFilterXYWH()

	def __init__(
			self,
			xywh        : np.ndarray,
			score       : float,
			cls         : int,
			feat        : np.ndarray | None = None,
			feat_history: int = 50,
			alpha       : float = 0.9,
			delta_t     : int = 3,
	):
		"""Initialize a HTPMTrack object with temporal parameters, such as feature history, alpha, and current features.

		Args:
			xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
				the center, (w, h) are width and height, and `idx` is the detection index.
			score (float): Confidence score of the detection.
			cls (int): Class ID of the detected object.
			feat (np.ndarray, optional): Feature vector associated with the detection.
			feat_history (int): Maximum length of the feature history deque.
			alpha (float): Momentum of embedding update.
			delta_t (int): Time step difference to estimate direction.
		"""
		super().__init__()
		# xywh+idx or xywha+idx
		assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
		self._tlwh         = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
		self.kalman_filter = None
		self.mean, self.covariance = None, None
		self.is_activated  = False

		self.score        = score
		self.tracklet_len = 0
		self.cls          = cls
		self.idx          = xywh[-1]
		self.angle        = xywh[4] if len(xywh) == 6 else None

		self.smooth_feat = None
		self.curr_feat   = None
		if feat is not None:
			self.update_features(feat)
		self.features = deque([], maxlen=feat_history)
		self.alpha    = alpha

		# HybridSORT parameters
		"""
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
		self.last_observation      = np.array([-1, -1, -1, -1, -1])  # placeholder
		self.last_observation_save = np.array([-1, -1, -1, -1, -1])
		self.observations          = dict()
		self.history_observations  = []
		self.velocity_lt           = None
		self.velocity_rt           = None
		self.velocity_lb           = None
		self.velocity_rb           = None
		self.delta_t               = delta_t
		self.confidence_pre        = None
		self.confidence            = score

	def update_features(self, feat: np.ndarray) -> None:
		"""Update the feature vector and apply exponential moving average smoothing."""
		feat /= np.linalg.norm(feat)
		self.curr_feat = feat
		if self.smooth_feat is None:
			self.smooth_feat = feat
		else:
			self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
		self.features.append(feat)
		self.smooth_feat /= np.linalg.norm(self.smooth_feat) #

	def predict(self) -> None:
		"""Predict the object's future state using the Kalman filter to update its mean and covariance."""
		mean_state = self.mean.copy()
		if self.state != TrackState.Tracked:
			mean_state[6] = 0
			mean_state[7] = 0

		self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

	def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
		"""Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
		self.kalman_filter = kalman_filter
		self.track_id = self.next_id()
		self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

		self.tracklet_len = 0
		self.state = TrackState.Tracked
		if frame_id == 1:
			self.is_activated = True
		self.frame_id    = frame_id
		self.start_frame = frame_id

	def re_activate(self, new_track: HTPMTrack, frame_id: int, new_id: bool = False) -> None:
		"""Reactivate a track with updated features and optionally assign a new ID."""
		if new_track.curr_feat is not None:
			self.update_features(new_track.curr_feat)

		self.mean, self.covariance = self.kalman_filter.update(
			self.mean, self.covariance, self.convert_coords(new_track.tlwh)
		)
		self.tracklet_len = 0
		self.state        = TrackState.Tracked
		self.is_activated = True
		self.frame_id     = frame_id
		if new_id:
			self.track_id = self.next_id()
		self.score = new_track.score
		self.cls   = new_track.cls
		self.angle = new_track.angle
		self.idx   = new_track.idx

	def update(self, new_track: HTPMTrack, frame_id: int) -> None:
		"""Update the track with new detection information and the current frame ID.
		Args:
			new_track (HTPMTrack): The new track containing updated information.
			frame_id (int): The ID of the current frame.

		Examples:
			Update the state of a track with new detection information
			>>> track = HTPMTrack([100, 200, 50, 80, 0.9, 1])
			>>> new_track = HTPMTrack([105, 205, 55, 85, 0.95, 1])
			>>> track.update(new_track, 2)
		"""
		if new_track.curr_feat is not None:
			self.update_features(new_track.curr_feat)

		self.frame_id      = frame_id
		self.tracklet_len += 1

		new_tlwh = new_track.tlwh

		self.mean, self.covariance = self.kalman_filter.update(
			self.mean, self.covariance, self.convert_coords(new_tlwh)
		)
		self.state        = TrackState.Tracked
		self.is_activated = True

		self.score = new_track.score
		self.cls   = new_track.cls
		self.angle = new_track.angle
		self.idx   = new_track.idx

	@property
	def tlwh(self) -> np.ndarray:
		"""Return the current bounding box position in `(top left x, top left y, width, height)` format."""
		if self.mean is None:
			return self._tlwh.copy()
		ret      = self.mean[:4].copy()
		ret[:2] -= ret[2:] / 2
		return ret

	@staticmethod
	def multi_predict(stracks: list[HTPMTrack]) -> None:
		"""Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
		if len(stracks) <= 0:
			return
		multi_mean       = np.asarray([st.mean.copy() for st in stracks])
		multi_covariance = np.asarray([st.covariance for st in stracks])
		for i, st in enumerate(stracks):
			if st.state != TrackState.Tracked:
				multi_mean[i][6] = 0
				multi_mean[i][7] = 0
		multi_mean, multi_covariance = HTPMTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
		for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
			stracks[i].mean       = mean
			stracks[i].covariance = cov

	@staticmethod
	def multi_gmc(stracks: list[HTPMTrack], H: np.ndarray = np.eye(2, 3)):
		"""Update state tracks positions and covariances using a homography matrix for multiple tracks."""
		if stracks:
			multi_mean       = np.asarray([st.mean.copy() for st in stracks])
			multi_covariance = np.asarray([st.covariance for st in stracks])

			R    = H[:2, :2]
			R8x8 = np.kron(np.eye(4, dtype=float), R)
			t    = H[:2, 2]

			for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
				mean = R8x8.dot(mean)
				mean[:2] += t
				cov  = R8x8.dot(cov).dot(R8x8.transpose())

				stracks[i].mean       = mean
				stracks[i].covariance = cov

	def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
		"""Convert tlwh bounding box coordinates to xywh format."""
		return self.tlwh_to_xywh(tlwh)

	@staticmethod
	def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
		"""Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
		ret      = np.asarray(tlwh).copy()
		ret[:2] += ret[2:] / 2
		return ret

	@property
	def xyxy(self) -> np.ndarray:
		"""Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
		ret      = self.tlwh.copy()
		ret[2:] += ret[:2]
		return ret

	@staticmethod
	def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
		"""Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
		ret      = np.asarray(tlwh).copy()
		ret[:2] += ret[2:] / 2
		ret[2]  /= ret[3]
		return ret

	@property
	def xywh(self) -> np.ndarray:
		"""Get the current position of the bounding box in (center x, center y, width, height) format."""
		ret      = np.asarray(self.tlwh).copy()
		ret[:2] += ret[2:] / 2
		return ret

	@property
	def xywha(self) -> np.ndarray:
		"""Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
		if self.angle is None:
			logger.warning("`angle` attr not found, returning `xywh` instead.")
			return self.xywh
		return np.concatenate([self.xywh, self.angle[None]])

	@property
	def result(self) -> list[float]:
		"""Get the current tracking results in the appropriate bounding box format."""
		coords = self.xyxy if self.angle is None else self.xywha
		return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

	def __repr__(self) -> str:
		"""Return a string representation of the STrack object including start frame, end frame, and track ID."""
		return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class HybridTPMOT:
	"""An Thermal Object Tracking.

	Attributes:
		tracked_stracks (list[HTPMTrack]): List of successfully activated tracks.
		lost_stracks (list[HTPMTrack]): List of lost tracks.
		removed_stracks (list[HTPMTrack]): List of removed tracks.
		frame_id (int): The current frame ID.
		args (Namespace): Command-line arguments.
		max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
		kalman_filter (KalmanFilterXYAH): Kalman Filter object.
		proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
		appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
		encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
		gmc (GMC): An instance of the GMC algorithm for data association.
		args (Any): Parsed command-line arguments containing tracking parameters.

	Methods:
		update(results, img=None, feats=None) -> np.ndarray
			Update the tracker with new detections and optional features and return currently tracked objects.
		get_kalmanfilter() -> KalmanFilterXYWH
			Create and return a KalmanFilterXYWH instance.
		init_track(results, img=None) -> list[HTPMTrack]
			Initialize HTPMTrack objects from detection results and optional ReID features.
		get_dists(tracks, detections) -> np.ndarray
			Compute a distance matrix between tracks and detections using IoU and optional ReID embedding distances.
		multi_predict(tracks) -> None
			Predict the mean and covariance for multiple tracks using a shared Kalman filter.
		reset() -> None
			Reset the tracker to its initial state and clear internal parameters.
		reset_id() -> None
			Reset the global ID counter used by HTPMTrack instances.
		joint_stracks(tlista, tlistb) -> list[HTPMTrack]
			Return the union of two HTPMTrack lists without duplicate track IDs.
		sub_stracks(tlista, tlistb) -> list[HTPMTrack]
			Return items from `tlista` whose IDs are not present in `tlistb`.
		remove_duplicate_stracks(stracksa, stracksb) -> tuple[list[HTPMTrack], list[HTPMTrack]]
			Remove duplicate tracks between two lists using IoU and prefer the longer-lived track.

	Examples:
		Initialize HybridTPMOT and process detections
		>>> hybrid_tp_mot = HybridTPMOT(args, frame_rate=30)
		>>> hybrid_tp_mot.init_track(dets, feats, img)
		>>> hybrid_tp_mot.multi_predict(tracks)

	Notes:
		- Designed to work with a YOLO detector.
		- ReID is used only when `args.with_reid` is True and an encoder is provided.
	"""

	# create default association function
	asso_func = "iou"

	def __init__(self, args: Any, frame_rate: int = 30):
		"""Initialize HybridTPMOT object with ReID module and GMC algorithm.

		Args:
			args (Any): Parsed command-line arguments containing tracking parameters.
			frame_rate (int): Frame rate of the video being processed.
		"""
		self.tracked_stracks = []  # type: list[HTPMTrack]
		self.lost_stracks    = []  # type: list[HTPMTrack]
		self.removed_stracks = []  # type: list[HTPMTrack]

		self.frame_id      = 0
		self.args          = args
		# self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
		self.max_time_lost = args.track_buffer  # buffer to calculate the time when to remove tracks
		self.kalman_filter = self.get_kalmanfilter()
		self.reset_id()

		self.gmc = GMC(method=args.gmc_method)

		# ReID module
		self.proximity_thresh  = args.proximity_thresh
		self.appearance_thresh = args.appearance_thresh
		HybridTPMOT.asso_func  = args.asso_func
		self.encoder = None
		if args.with_reid:
			# self.encoder = YOLO_ReID(args.reid_model)
			self.encoder = []
			for reid_model in args["reid_model"]:
				if isinstance(reid_model, dict):
					self.encoder.append(IDENTIFICATIONS.build(**reid_model['model']))
				else:
					raise ValueError(f"Cannot initialize identification with {reid_model['name']}.")

	def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
		"""Update the tracker with new detections and return the current list of tracked objects."""
		self.frame_id    += 1
		activated_stracks = []
		refind_stracks    = []
		lost_stracks      = []
		removed_stracks   = []

		scores      = results.conf
		remain_inds = scores >= self.args.track_high_thresh
		inds_low    = scores > self.args.track_low_thresh
		inds_high   = scores < self.args.track_high_thresh

		inds_second    = inds_low & inds_high
		results_second = results[inds_second]
		results        = results[remain_inds]
		feats_keep     = feats_second = img
		if feats is not None and len(feats):
			feats_keep   = feats[remain_inds]
			feats_second = feats[inds_second]

		# Step 1: Get detections from results and feats (keep high score and low score respectively)
		if feats is not None and len(feats):  # Do we have external ReID features?
			detections = self.init_track(results, feats_keep, img)
		else:
			detections = self.init_track(results, None, img)

		# Add newly detected tracklets to tracked_stracks
		unconfirmed     = []
		tracked_stracks = []  # type: list[HTPMTrack]
		for track in self.tracked_stracks:
			if not track.is_activated:
				unconfirmed.append(track)
			else:
				tracked_stracks.append(track)

		# Step 2: First association, with high score detection boxes
		strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
		# Predict the current location with KF
		self.multi_predict(strack_pool)
		if hasattr(self, "gmc") and img is not None:
			# use try-except here to bypass errors from gmc module
			try:
				warp = self.gmc.apply(img, results.xyxy)
			except Exception:
				warp = np.eye(2, 3)
			HTPMTrack.multi_gmc(strack_pool, warp)
			HTPMTrack.multi_gmc(unconfirmed, warp)

		dists = self.get_dists(strack_pool, detections)
		matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh_stage_1)

		for itracked, idet in matches:
			track = strack_pool[itracked]
			det   = detections[idet]
			if track.state == TrackState.Tracked:
				track.update(det, self.frame_id)
				activated_stracks.append(track)
			else:
				track.re_activate(det, self.frame_id, new_id=False)
				refind_stracks.append(track)

		# Step 3: Second association, with low score detection boxes association the untrack to the low score detections
		if feats is not None and len(feats): # Do we have external ReID features?
			detections_second = self.init_track(results_second, feats_second, img)
		else:
			detections_second = self.init_track(results_second, None, img)

		r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
		# TODO: consider fusing scores or appearance features for second association.
		# dists = matching.iou_distance(r_tracked_stracks, detections_second, HybridTPMOT.asso_func)
		# SUGAR:
		dists = self.get_dists(r_tracked_stracks, detections_second)
		matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=self.args.match_thresh_stage_2)
		for itracked, idet in matches:
			track = r_tracked_stracks[itracked]
			det = detections_second[idet]
			if track.state == TrackState.Tracked:
				track.update(det, self.frame_id)
				activated_stracks.append(track)
			else:
				track.re_activate(det, self.frame_id, new_id=False)
				refind_stracks.append(track)

		for it in u_track:
			track = r_tracked_stracks[it]
			if track.state != TrackState.Lost:
				track.mark_lost()
				lost_stracks.append(track)

		# Deal with unconfirmed tracks, usually tracks with only one beginning frame
		detections = [detections[i] for i in u_detection]
		dists = self.get_dists(unconfirmed, detections)
		matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh_stage_3)
		for itracked, idet in matches:
			unconfirmed[itracked].update(detections[idet], self.frame_id)
			activated_stracks.append(unconfirmed[itracked])
		for it in u_unconfirmed:
			track = unconfirmed[it]
			track.mark_removed()
			removed_stracks.append(track)

		# Step 4: Init new stracks
		for inew in u_detection:
			track = detections[inew]
			if track.score < self.args.new_track_thresh:
				continue
			track.activate(self.kalman_filter, self.frame_id)
			activated_stracks.append(track)

		# Step 5: Update state
		for track in self.lost_stracks:
			if self.frame_id - track.end_frame > self.max_time_lost:
				track.mark_removed()
				removed_stracks.append(track)

		self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
		self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
		self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
		self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
		self.lost_stracks.extend(lost_stracks)
		self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.removed_stracks)
		self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
		self.removed_stracks.extend(removed_stracks)
		if len(self.removed_stracks) > 1000:
			self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

		return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

	def get_kalmanfilter(self) -> KalmanFilterXYWH:
		"""Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
		return KalmanFilterXYWH()

	def init_track(self, results, feats = None, img: np.ndarray | None = None) -> list[HTPMTrack]:
		"""Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
		if len(results) == 0:
			return []

		bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
		bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

		if self.args.with_reid and feats is not None and feats[0] is not None:  # if use external ReID features
			return [HTPMTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, feats)]
		elif self.args.with_reid and self.encoder is not None and (feats is None or feats[0] is None): # extract ReID features
			if isinstance(self.encoder, list):
				features_list = []
				for encoder in self.encoder:
					features = encoder.extract_feature(img, bboxes)
					features_list.append(features)
				features_keep = np.concatenate(features_list, axis=0)
			else:
				features_keep = self.encoder.extract_feature(img, bboxes)
			return [HTPMTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
		else:
			return [HTPMTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

	def get_dists(self, tracks: list[HTPMTrack], detections: list[HTPMTrack]) -> np.ndarray:
		"""Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
		dists      = matching.iou_distance(tracks, detections, HybridTPMOT.asso_func)

		# Compute multiple IoU-based distances for potential fusion
		dists_ciou = matching.iou_distance(tracks, detections, "ciou")
		dists_giou = matching.iou_distance(tracks, detections, "giou")
		dists_diou = matching.iou_distance(tracks, detections, "diou")
		# dists_mask = dists > (1 - self.proximity_thresh)

		if self.args.fuse_score:
			dists = matching.fuse_score(dists, detections)

		if self.args.with_reid and self.encoder is not None:
			# emb_dists = matching.embedding_distance(tracks, detections) / 2.0
			# emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
			# emb_dists[dists_mask] = 1.0
			# dists = np.minimum(dists, emb_dists)

			# SUGAR:
			emb_dists = matching.embedding_distance(tracks, detections)
			# dists = dists + emb_dists * 20  # TINY ORI
			# dists = dists + emb_dists * 4  # TINY NEW
			# dists = dists + emb_dists * 7.7   # BASE NEW

			dists = (dists_ciou * 0.5 + dists_giou * 0.25 + dists_diou * 0.25) + emb_dists * 4  # TINY NEW
		return dists

	def multi_predict(self, tracks: list[HTPMTrack]) -> None:
		"""Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
		HTPMTrack.multi_predict(tracks)

	def reset(self) -> None:
		"""Reset the HybridTPMOT tracker to its initial state, clearing all tracked objects and internal states."""
		self.tracked_stracks = []  # type: list[STrack]
		self.lost_stracks    = []  # type: list[STrack]
		self.removed_stracks = []  # type: list[STrack]
		self.frame_id        = 0
		self.kalman_filter   = self.get_kalmanfilter()
		self.reset_id()
		self.gmc.reset_params()

	@staticmethod
	def reset_id():
		"""Reset the ID counter for HTPMTrack instances to ensure unique track IDs across tracking sessions."""
		HTPMTrack.reset_id()

	@staticmethod
	def joint_stracks(tlista: list[HTPMTrack], tlistb: list[HTPMTrack]) -> list[HTPMTrack]:
		"""Combine two lists of HTPMTrack objects into a single list, ensuring no duplicates based on track IDs."""
		exists = {}
		res = []
		for t in tlista:
			exists[t.track_id] = 1
			res.append(t)
		for t in tlistb:
			tid = t.track_id
			if not exists.get(tid, 0):
				exists[tid] = 1
				res.append(t)
		return res

	@staticmethod
	def sub_stracks(tlista: list[HTPMTrack], tlistb: list[HTPMTrack]) -> list[HTPMTrack]:
		"""Filter out the stracks present in the second list from the first list."""
		track_ids_b = {t.track_id for t in tlistb}
		return [t for t in tlista if t.track_id not in track_ids_b]

	@staticmethod
	def remove_duplicate_stracks(stracksa: list[HTPMTrack], stracksb: list[HTPMTrack]) -> tuple[list[HTPMTrack], list[HTPMTrack]]:
		"""Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
		pdist = matching.iou_distance(stracksa, stracksb, HybridTPMOT.asso_func)
		pairs = np.where(pdist < 0.15)
		dupa, dupb = [], []
		for p, q in zip(*pairs):
			timep = stracksa[p].frame_id - stracksa[p].start_frame
			timeq = stracksb[q].frame_id - stracksb[q].start_frame
			if timep > timeq:
				dupb.append(q)
			else:
				dupa.append(p)
		resa = [t for i, t in enumerate(stracksa) if i not in dupa]
		resb = [t for i, t in enumerate(stracksb) if i not in dupb]
		return resa, resb

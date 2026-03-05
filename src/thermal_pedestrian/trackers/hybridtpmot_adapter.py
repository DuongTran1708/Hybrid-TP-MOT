from __future__ import annotations

from typing import (Any, Union, List)
import sys
import warnings
from typing import Optional

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform

from munch import Munch
from torch import Tensor
import torch

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.image import to_channel_first
from thermal_pedestrian.trackers import BaseTracker_Adapter
from thermal_pedestrian.trackers.hybridtpmot.hybrid_tp_mot import HybridTPMOT

from ultralytics.engine.results import Boxes

__all__ = [
	"HybridTPMOT_Adapter"
]


# MARK: - HybridTPMOT_Adapter

@TRACKERS.register(name="hybridtpmot")
class HybridTPMOT_Adapter(BaseTracker_Adapter):
	"""HybridTPMOT

	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions

	def __init__(self, hybridtpmot_config: dict, **kwargs):
		super().__init__(**kwargs)
		self.hybridtpmot_config = hybridtpmot_config
		self.init_model()

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		self.model = HybridTPMOT(self.hybridtpmot_config)

	# MARK: Update

	def update(self, detections: List[Instance], image: Any, *args, **kwargs):
		"""Update ``self.tracks`` with new detections.

		Args:
			detections (list):
				The list of newly ``Instance`` objects.
			image (any):
				The current frame image.
			features (any):
				The current bbox external features.

		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:

		"""
		boxes_data = []
		features   = None
		for det in detections:
			# boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
			#                 (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
			box_data = [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence, det.class_label['id']]
			boxes_data.append(box_data)
			if self.hybridtpmot_config["with_reid"]:
				if features is None:
					features = []
				features.append(det.feature)

		if len(boxes_data) == 0:
			self.tracks = []
			return

		boxes_data = np.array(boxes_data)
		boxes      = Boxes(boxes_data, image.shape[:2])
		features   = np.array(features) if features is not None else None

		# DEBUG:
		# print("*" * 50)
		# print(boxes_data[0])
		# print(features[0][:5])
		# print("*" * 50)
		# import sys
		# sys.exit()

		# coords.tolist() + [self.track_id, self.score, self.cls, self.idx]
		results   = self.model.update(results = boxes, img = image, feats = features)

		if len(results) == 0:
			self.tracks = []
			return

		# bboxes       = results[:,: 4].astype(float)
		track_ids    = results[:, 4].astype(int)
		# confs        = results[:, 5].astype(int)
		# clsss        = results[:, 6].astype(int)
		indexes_bbox = results[:, 7].astype(int)
		self.tracks = []
		for track_id, idx in zip(track_ids, indexes_bbox):
			gmo  = General_Moving_Object.gmo_from_detection(detections[idx])
			gmo.id = int(track_id)
			self.tracks.append(gmo)


	def update_matched_tracks(
			self,
			matched   : Union[List, np.ndarray],
			detections: List[Instance]
	):
		"""Update the track that has been matched with new detection

		Args:
			matched (list or np.ndarray):
				Matching between self.tracks index and detection index.
			detections (any):
				The newly detections.
		"""
		pass

	def create_new_tracks(
			self,
			unmatched_dets: Union[List, np.ndarray],
			detections    : List[Instance]
	):
		"""Create new tracks.

		Args:
			unmatched_dets (list or np.ndarray):
				Index of the newly detection in ``detections`` that has not matched with any tracks.
			detections (any):
				The newly detections.
		"""
		pass

	def delete_dead_tracks(
			self
	):
		"""Delete dead tracks.
		"""
		pass

	def associate_detections_to_tracks(
			self,
			dets: np.ndarray,
			trks: np.ndarray,
			**kwargs
	):
		"""Assigns detections to ``self.tracks``

		Args:
			dets (np.ndarray):
				The list of newly ``Instance`` objects.
			trks (np.ndarray):

		Returns:
			3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		pass

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()

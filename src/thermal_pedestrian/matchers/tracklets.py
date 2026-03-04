from scipy.spatial.distance import cdist
import numpy as np

def is_matching_tracklets(tracklet_a, tracklet_b, heuristic_cfg):
	"""
	Determine if two tracklets match based on their frame, feats.

	tracklet = {
				"label_id"   : label_id,
				"frames"     : [],
				"bboxes"     : [],
				"scores"     : [],
				"feats"      : [],
				"frame_start": frame_index,  # PBVS rule, frame start from 1
				"frame_end"  : frame_index,  # PBVS rule, frame start from 1
			}

	Args:
		tracklet_a (dict):

		tracklet_b (dict):
	Returns:
		bool: True if the tracklets match, False otherwise.
	"""
	return False

	reid_model_config = heuristic_cfg["reid_model"]

	# check frame overlap
	# for frame in tracklet_a["frames"]:
	# 	if frame in tracklet_b["frames"]:
	# 		return False
	if (tracklet_a["frame_start"] - tracklet_b["frame_start"]) * (tracklet_a["frame_end"] - tracklet_b["frame_end"]) <= 0:
		return False

	# check frame gap
	if heuristic_cfg["min_frame_gap"] < abs(tracklet_b["frame_start"] - tracklet_a["frame_end"]) < heuristic_cfg["max_frame_gap"] or \
			heuristic_cfg["min_frame_gap"] < abs(tracklet_a["frame_start"] - tracklet_b["frame_end"]) < heuristic_cfg["max_frame_gap"]:
		return False

	# DEBUG:
	# print("Computing ReID cost between tracklets {}-{} and {}-{}".format(
	# 	tracklet_a["frame_start"], tracklet_a["frame_end"],
	# 	tracklet_b["frame_start"], tracklet_b["frame_end"],
	# ))

	# get variant variables and features
	reid_feats_cost_matrix = []
	arr_var    = []
	for index_model, reid_model in enumerate(reid_model_config):
		arr_var.append(float(reid_model['model']['ratio']))

		# get all features for the current reid model of both tracklets
		reid_feat_a = []
		for bbox, score, feat in zip(tracklet_a["bboxes"], tracklet_a["scores"], tracklet_a["feats"]):
			if bbox[3] / bbox[2] > heuristic_cfg["ratio_height_width"]: # height / width ratio filter
				continue
			if score < heuristic_cfg["min_confidence_det"]:
				continue
			feat_value_temp = np.array(feat["reid"][index_model], dtype=np.float32).reshape(-1)
			reid_feat_a.append(feat_value_temp)

		reid_feat_b = []
		for bbox, score, feat in zip(tracklet_b["bboxes"], tracklet_b["scores"], tracklet_b["feats"]):
			if bbox[3] / bbox[2] > heuristic_cfg["ratio_height_width"]: # height / width ratio filter
				continue
			if score < heuristic_cfg["min_confidence_det"]:
				continue
			feat_value_temp = np.array(feat["reid"][index_model], dtype=np.float32).reshape(-1)
			reid_feat_b.append(feat_value_temp)


		metric      = "euclidean"
		reid_feat_a = np.array(reid_feat_a)
		reid_feat_b = np.array(reid_feat_b)

		# DEBUG:
		# print(reid_feat_a.shape)
		# print(reid_feat_b.shape)
		# import sys
		# sys.exit()

		# DEBUG:
		# a = a / np.linalg.norm(a)
		# b = b / np.linalg.norm(b)
		# return 1 - np.dot(a, b.T)


		# DEBUG:
		# print(np.mean(cost_matrix))
		# print(1 - np.mean(cost_matrix))
		# import sys
		# sys.exit()

		cost_matrix = np.maximum(0.0, cdist(reid_feat_a, reid_feat_b, metric))  # Normalized features
		cost_matrix = cost_matrix.reshape(-1)

		# a = reid_feat_a / np.linalg.norm(reid_feat_a)
		# b = reid_feat_b / np.linalg.norm(reid_feat_b)
		# if len(a) == 0 or len(b) == 0:
		# 	continue
		# cost_matrix =  1 - np.dot(a, b.T)

		reid_feats_cost_matrix.append(cost_matrix)

	mean_cost = 0
	for index_model, cost_matrix in enumerate(reid_feats_cost_matrix):
		weighted_cost_matrix = np.min(cost_matrix) * arr_var[index_model]
		mean_cost += weighted_cost_matrix

	# DEBUG:
	# print(mean_cost)
	# import sys
	# sys.exit()

	# print(mean_cost)
	if mean_cost < heuristic_cfg["max_reid_distance"]:
		return False

	return True
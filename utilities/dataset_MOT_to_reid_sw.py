# ==================================================================== #
# Copyright (C) 2026 - Automation Lab - Sungkyunkwan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# ==================================================================== #
"""Person re-identification dataset generation from MOT datasets.

This program converts thermal multi-object tracking (MOT) datasets (TMOT,
VT-MOT) into person re-identification (Re-ID) datasets in the Market-1501
layout:

	bounding_box_train/  Training set: images the model learns from.
	bounding_box_test/   Gallery: images searched during evaluation.
	query/               Probe: images matched against the gallery.

In every sequence the pedestrian identities are randomly split into
disjoint training and testing identities. For each testing identity with
enough crops, a small share of its gallery crops is moved to the query set.
For VT-MOT, a crop is kept only when a YOLO pose model detects all 17 COCO
keypoints on the paired RGB frame, i.e. the crop shows a full-body person.
The thermal crop is the image that is saved.

File names follow the Market-1501 convention
``{person_id:08d}_c{camera_id}_{frame_id:08d}_00.jpg``. Each sequence is
treated as a separate camera. Person IDs and camera IDs are offset so that
they remain globally unique across all sequences and datasets.

Usage:
	python utilities/dataset_MOT_to_reid_sw.py
"""

from __future__ import annotations

import glob
import math
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# region Configuration

POSE_MODEL_PATH = "yolo11x-pose.pt"

DATASETS = (
	{
		"name"       : "VTMOT",
		"splits"     : ("train", "test"),
		"gt_root"    : "<path_to_dataset_folder>/VTMOT/images/{split}",
		"image_root" : "<path_to_dataset_folder>/VTMOT/images/{split}",
		"gt_file"    : "gt/gt.txt",
		"thermal_dir": "infrared",
		"rgb_dir"    : "visible",   # RGB frames used by the full-body pose filter.
		"image_ext"  : "jpg",
		"output_root": "<path_to_dataset_folder>/VTMOT/re_identification/train/vtmot/",
	},
)

MARKET1501_SUBDIRS = ("bounding_box_train", "bounding_box_test", "query")

MOT_PEDESTRIAN_CLASS = 1     # Pedestrian class ID in MOT annotations.
MIN_CROP_SIZE        = 5     # Crops with width or height <= 5 pixels are skipped.
TRAIN_RATIO          = 0.5   # Fraction of the identities of a sequence used for training.
QUERY_PERCENT        = 5     # Percentage of a testing identity's crops moved to the query set.
MIN_QUERY_CROPS      = 3     # Testing identities with fewer crops get no query images.
QUERY_CAMERA_BASE    = 9999  # Query camera ID is QUERY_CAMERA_BASE - camera_id.
POSE_CONF_THRESHOLD  = 0.25  # Detection confidence for the pose model.
NUM_KEYPOINTS        = 17    # Number of COCO body keypoints.
KEYPOINT_CONF        = 0.5   # Minimum confidence for a keypoint to be visible.

# endregion


# region Utilities

def create_output_dirs(output_root: str):
	"""Create the Market-1501 sub-directories of the Re-ID dataset.

	Args:
		output_root: Root directory of the output dataset.
	"""
	for subdir in MARKET1501_SUBDIRS:
		os.makedirs(os.path.join(output_root, subdir), exist_ok=True)


def clamp_bbox(x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> tuple[int, int, int, int]:
	"""Clamp a bounding box so that it does not exceed the image boundaries.

	Args:
		x, y: Top-left corner of the box.
		w, h: Width and height of the box.
		img_width, img_height: Size of the image.

	Returns:
		The clamped box ``(x, y, w, h)`` in integer pixels. Width or height
		may be non-positive when the box lies outside the image.
	"""
	x = max(0, int(x))
	y = max(0, int(y))
	w = min(int(w), img_width - x)
	h = min(int(h), img_height - y)
	return x, y, w, h


def load_mot_ground_truth(gt_path: str) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
	"""Load pedestrian annotations from a MOT-format ground-truth file.

	Each line has the format ``frame, id, left, top, width, height[, conf,
	class, visibility]``. Lines that are empty, contain ``nan``, or have
	fewer than six columns are ignored. Only pedestrians are kept; when the
	class column is missing the object is assumed to be a pedestrian.

	Args:
		gt_path: Path to the ground-truth file.

	Returns:
		A mapping ``{frame_id: [(person_id, (x, y, w, h)), ...]}``.
	"""
	frame_map = defaultdict(list)
	with open(gt_path, "r") as f_read:
		for line in f_read:
			line = line.strip()
			if not line or "nan" in line:
				continue

			parts = line.split(",")
			if len(parts) < 6:
				continue

			class_id = int(parts[7]) if len(parts) >= 8 else MOT_PEDESTRIAN_CLASS
			if class_id != MOT_PEDESTRIAN_CLASS:
				continue

			frame_id  = int(parts[0])
			person_id = int(float(parts[1]))
			bbox      = tuple(float(v) for v in parts[2:6])
			frame_map[frame_id].append((person_id, bbox))
	return frame_map


def is_full_body_person(model_pose: YOLO, crop: np.ndarray) -> bool:
	"""Check whether a crop contains a full-body person.

	The crop is full-body when, in any pose result, the first detected
	person has all ``NUM_KEYPOINTS`` keypoints visible, i.e. with a
	confidence above ``KEYPOINT_CONF`` and positive coordinates.

	Args:
		model_pose: YOLO pose-estimation model.
		crop: Person crop of shape ``[H, W, 3]``.

	Returns:
		``True`` if the crop contains a full-body person.
	"""
	try:
		results = model_pose(crop, conf=POSE_CONF_THRESHOLD, verbose=False)
	except Exception:  # noqa: BLE001 - a failed inference skips the crop instead of aborting the run
		return False

	for result in results:
		if result.keypoints is None or len(result.keypoints.data) == 0:
			continue

		person_kpts = result.keypoints.data[0]  # First person: [17, 3] as (x, y, conf)
		x, y, conf  = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
		if int(((conf > KEYPOINT_CONF) & (x > 0) & (y > 0)).sum()) == NUM_KEYPOINTS:
			return True
	return False

# endregion


# region Conversion

def split_query_from_gallery(output_root: str, test_ids: set[int], id_offset: int, camera_id: int):
	"""Move a share of each testing identity's gallery crops to the query set.

	Moved crops get the camera ID ``QUERY_CAMERA_BASE - camera_id``, because
	the standard Re-ID evaluation ignores gallery images that share both the
	identity and the camera of the query.

	Args:
		output_root: Root directory of the output dataset.
		test_ids: Local person IDs of the testing identities.
		id_offset: Value added to every person ID to make it globally unique.
		camera_id: Camera ID of the sequence.
	"""
	gallery_dir = os.path.join(output_root, "bounding_box_test")
	query_dir   = os.path.join(output_root, "query")
	for person_id in test_ids:
		gallery_paths = glob.glob(os.path.join(gallery_dir, f"{person_id + id_offset:08d}_c{camera_id}_*.jpg"))
		if len(gallery_paths) < MIN_QUERY_CROPS:
			continue

		num_query = math.ceil(len(gallery_paths) * QUERY_PERCENT / 100)
		for gallery_path in random.sample(gallery_paths, num_query):
			filename   = os.path.basename(gallery_path).replace(f"_c{camera_id}_", f"_c{QUERY_CAMERA_BASE - camera_id}_")
			query_path = os.path.join(query_dir, filename)
			if not os.path.exists(query_path):
				shutil.move(gallery_path, query_path)


def convert_sequence(
		dataset   : dict,
		split     : str,
		seq_name  : str,
		camera_id : int,
		id_offset : int,
		model_pose: YOLO,
) -> int:
	"""Convert one MOT sequence into Re-ID training, gallery and query images.

	Args:
		dataset: Dataset configuration (an entry of ``DATASETS``).
		split: Dataset split, e.g. ``"train"``.
		seq_name: Name of the sequence directory.
		camera_id: Camera ID written into the output file names.
		id_offset: Value added to every person ID to make it globally unique.
		model_pose: YOLO pose-estimation model, used only when the dataset
			has an ``rgb_dir``.

	Returns:
		The largest global person ID written, or ``id_offset`` if none.
	"""
	gt_root     = dataset["gt_root"].format(split=split)
	image_root  = dataset["image_root"].format(split=split)
	gt_path     = os.path.join(gt_root, seq_name, dataset["gt_file"].format(seq=seq_name))
	thermal_dir = os.path.join(image_root, seq_name, dataset["thermal_dir"])
	rgb_dir     = os.path.join(image_root, seq_name, dataset["rgb_dir"]) if dataset["rgb_dir"] else None
	output_root = dataset["output_root"]

	if not os.path.exists(gt_path):
		print(f"Error: ground truth not found at {gt_path}")
		return id_offset

	frame_map = load_mot_ground_truth(gt_path)

	# Split the identities into disjoint training and testing sets.
	person_ids = list({person_id for detections in frame_map.values() for person_id, _ in detections})
	random.shuffle(person_ids)
	num_train = int(len(person_ids) * TRAIN_RATIO)
	train_ids = set(person_ids[:num_train])
	test_ids  = set(person_ids[num_train:])
	print(f"\nSequence: {seq_name} | Total IDs: {len(person_ids)} | Train: {len(train_ids)} | Test + Query: {len(test_ids)}")

	thermal_paths  = sorted(glob.glob(os.path.join(thermal_dir, f"*.{dataset['image_ext']}")))
	max_global_pid = id_offset
	# shortcut: pairs the i-th annotated frame with the i-th image, so it assumes every frame has annotations; match by frame number if that ever fails.
	frame_pairs = zip(sorted(frame_map), thermal_paths)
	for frame_id, thermal_path in tqdm(frame_pairs, total=len(thermal_paths), desc=f"Processing {seq_name}"):
		img_thermal = cv2.imread(thermal_path)
		if img_thermal is None:
			continue

		img_rgb = None
		if rgb_dir is not None:
			rgb_path = os.path.join(rgb_dir, os.path.basename(thermal_path))
			img_rgb  = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None
			if img_rgb is None:
				continue

		img_h, img_w = img_thermal.shape[:2]
		for person_id, bbox in frame_map[frame_id]:
			x, y, w, h = clamp_bbox(*bbox, img_w, img_h)
			if w <= MIN_CROP_SIZE or h <= MIN_CROP_SIZE:
				continue

			if img_rgb is not None and not is_full_body_person(model_pose, img_rgb[y:y + h, x:x + w]):
				continue

			global_pid     = person_id + id_offset
			max_global_pid = max(max_global_pid, global_pid)
			subdir         = "bounding_box_train" if person_id in train_ids else "bounding_box_test"
			filename       = f"{global_pid:08d}_c{camera_id}_{frame_id:08d}_00.jpg"
			cv2.imwrite(os.path.join(output_root, subdir, filename), img_thermal[y:y + h, x:x + w])

	split_query_from_gallery(output_root, test_ids, id_offset, camera_id)
	return max_global_pid


def convert_dataset(
		dataset   : dict,
		model_pose: YOLO,
		id_offset : int = -1,
		camera_id : int = -1,
) -> tuple[int, int]:
	"""Convert every sequence of a MOT dataset into a Re-ID dataset.

	Each sequence is treated as a separate camera. Person IDs continue from
	``id_offset`` and camera IDs continue from ``camera_id`` so that several
	datasets can be merged into one Re-ID dataset.

	Args:
		dataset: Dataset configuration (an entry of ``DATASETS``).
		model_pose: YOLO pose-estimation model.
		id_offset: Last global person ID used so far.
		camera_id: Last camera ID used so far.

	Returns:
		The updated ``(id_offset, camera_id)``.
	"""
	create_output_dirs(dataset["output_root"])
	for split in dataset["splits"]:
		sequences = sorted(os.listdir(dataset["gt_root"].format(split=split)))
		for seq_name in tqdm(sequences, desc=f"{dataset['name']} {split}"):
			camera_id += 1
			id_offset  = convert_sequence(dataset, split, seq_name, camera_id, id_offset + 1, model_pose)
	return id_offset, camera_id

# endregion


# region Main

def main():
	"""Convert all configured MOT datasets into Re-ID datasets."""
	model_pose = YOLO(POSE_MODEL_PATH)

	id_offset, camera_id = -1, -1
	for dataset in DATASETS:
		id_offset, camera_id = convert_dataset(dataset, model_pose, id_offset, camera_id)
		print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")


if __name__ == "__main__":
	main()

# endregion

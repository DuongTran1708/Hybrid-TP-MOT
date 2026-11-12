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
"""Multi-modal pedestrian instance extraction from MOT datasets.

This program converts thermal/RGB multi-object tracking (MOT) datasets
(TMOT, VT-MOT) into a person re-identification (Re-ID) dataset. Each
annotated pedestrian is cropped from the paired thermal and RGB frames and
kept only when a YOLO pose model detects all 17 COCO keypoints, i.e. the
crop contains a full-body person. For every kept instance the following
images are written:

	bbox_flir/          Thermal crop.
	bbox_rgb/           RGB crop.
	bbox_rgb_keypoint/  RGB crop with the pose skeleton drawn.
	bbox_rgb_polygon/   RGB crop with the segmentation contour drawn.
	instance_mask/      Binary person mask (255 = person).

File names follow the Market-1501 convention
``{person_id:08d}_c{camera_id}_{frame_id:08d}_00.jpg``. Person IDs and
camera IDs are offset so that they remain globally unique across all
sequences and datasets.

Usage:
	python utilities/dataset_MOT_to_pose_bbox_segmen_reid_sw.py
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# region Configuration

POSE_MODEL_PATH = "yolo11x-pose.pt"
SEG_MODEL_PATH  = "yolo11x-seg.pt"

DATASETS = (
	{
		"name"       : "VTMOT",
		"splits"     : ("train", "test"),
		"gt_root"    : "<path_to_dataset_folder>/VTMOT/images/{split}",
		"image_root" : "<path_to_dataset_folder>/VTMOT/images/{split}",
		"gt_file"    : "gt/gt.txt",
		"thermal_dir": "infrared",
		"rgb_dir"    : "visible",
		"image_ext"  : "jpg",
		"min_size"   : (100, 150),  # Crops with width <= 100 or height <= 150 are skipped.
		"output_root": "<path_to_dataset_folder>/VTMOT/instance_extraction/vtmot/",
	},
)

OUTPUT_SUBDIRS = ("bbox_flir", "bbox_rgb", "bbox_rgb_polygon", "bbox_rgb_keypoint", "instance_mask")

MOT_PEDESTRIAN_CLASS  = 1     # Pedestrian class ID in MOT annotations.
COCO_PERSON_CLASS     = 0     # Person class ID in COCO (segmentation model).
MODEL_CONF_THRESHOLD  = 0.1   # Detection confidence for the pose/segmentation models.
NUM_KEYPOINTS         = 17    # Number of COCO body keypoints.
KEYPOINT_CONF         = 0.15  # Minimum confidence for a keypoint to be visible.
MIN_VISIBLE_KEYPOINTS = 5     # Keypoints are returned only if more than this many are visible.

# COCO skeleton links (0-indexed keypoint pairs).
COCO_SKELETON = (
	(0, 1), (0, 2), (1, 3), (2, 4),             # Head
	(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # Upper body
	(5, 11), (6, 12), (11, 12),                 # Torso
	(11, 13), (13, 15), (12, 14), (14, 16),     # Lower body
)

# One BGR color per keypoint.
KEYPOINT_COLORS = (
	(0, 0, 255),   (0, 255, 0),   (255, 0, 0),   (0, 255, 255), (255, 255, 0),
	(255, 0, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (0, 0, 128),
	(0, 128, 0),   (128, 0, 0),   (255, 165, 0), (0, 165, 255), (165, 0, 255),
	(255, 255, 255), (128, 128, 128),
)

# endregion


# region Utilities

def create_output_dirs(output_root: str):
	"""Create the output sub-directories of the Re-ID dataset.

	Args:
		output_root: Root directory of the output dataset.
	"""
	for subdir in OUTPUT_SUBDIRS:
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

# endregion


# region Person Analysis

def check_full_body_person(
		model_pose: YOLO,
		model_seg : YOLO,
		crop_rgb  : np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
	"""Check whether a crop contains a full-body person and segment it.

	The pose model must detect all ``NUM_KEYPOINTS`` keypoints of the first
	person. If so, the segmentation model extracts the contour of the first
	detected person.

	Args:
		model_pose: YOLO pose-estimation model.
		model_seg: YOLO instance-segmentation model.
		crop_rgb: RGB person crop of shape ``[H, W, 3]``.

	Returns:
		A tuple ``(is_full_body, polygon, keypoints)`` where
			- ``is_full_body`` is ``True`` if all keypoints are visible.
			- ``polygon`` is an ``int32`` array ``[N, 2]`` of the person
			  contour, or ``None`` if segmentation failed.
			- ``keypoints`` is a ``[17, 3]`` array ``(x, y, conf)``, or
			  ``None`` if too few keypoints are visible.
	"""
	# Pose estimation
	try:
		pose_results = model_pose(crop_rgb, conf=MODEL_CONF_THRESHOLD, verbose=False)
	except Exception:  # noqa: BLE001 - a failed inference skips the crop instead of aborting the run
		return False, None, None

	is_full_body = False
	keypoints    = None
	for result in pose_results:
		if result.keypoints is None or len(result.keypoints.data) == 0:
			continue

		person_kpts = result.keypoints.data[0]  # First person: [17, 3]
		x, y, conf  = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
		num_visible = int(((conf > KEYPOINT_CONF) & (x > 0) & (y > 0)).sum())

		if num_visible > MIN_VISIBLE_KEYPOINTS:
			keypoints = person_kpts.cpu().numpy()
		if num_visible == NUM_KEYPOINTS:
			is_full_body = True
			break

	if not is_full_body:
		return False, None, keypoints

	# Instance segmentation
	try:
		seg_results = model_seg(crop_rgb, conf=MODEL_CONF_THRESHOLD, verbose=False)
	except Exception:  # noqa: BLE001 - keep the instance without a mask
		return True, None, keypoints

	for result in seg_results:
		if result.masks is None or result.boxes is None:
			continue

		classes        = result.boxes.cls.cpu().numpy().astype(int)
		person_indices = np.flatnonzero(classes == COCO_PERSON_CLASS)
		if len(person_indices) == 0:
			continue

		polygon = result.masks.xy[person_indices[0]]  # [N, 2] float pixel coordinates
		if len(polygon) > 0:
			return True, polygon.astype(np.int32), keypoints

	return True, None, keypoints


def draw_pose_on_image(img: np.ndarray, keypoints: np.ndarray, conf_threshold: float = KEYPOINT_CONF):
	"""Draw COCO keypoints and skeleton on an image in place.

	Args:
		img: BGR image to draw on.
		keypoints: Keypoints of shape ``[17, 3]`` as ``(x, y, conf)``.
		conf_threshold: Minimum confidence for a keypoint to be drawn.
	"""
	for p1, p2 in COCO_SKELETON:
		if p1 < len(keypoints) and p2 < len(keypoints):
			x1, y1, c1 = keypoints[p1]
			x2, y2, c2 = keypoints[p2]
			if c1 > conf_threshold and c2 > conf_threshold:
				color = KEYPOINT_COLORS[p2 % len(KEYPOINT_COLORS)]
				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

	for i, (x, y, conf) in enumerate(keypoints):
		if conf > conf_threshold:
			color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
			cv2.circle(img, (int(x), int(y)), 2, color, -1)


def save_instance(
		output_root : str,
		filename    : str,
		crop_thermal: np.ndarray,
		crop_rgb    : np.ndarray,
		polygon     : np.ndarray | None,
		keypoints   : np.ndarray | None,
):
	"""Write all images of one person instance to the output dataset.

	Args:
		output_root: Root directory of the output dataset.
		filename: Market-1501 style file name of the instance.
		crop_thermal: Thermal crop.
		crop_rgb: RGB crop.
		polygon: Person contour ``[N, 2]``, or ``None`` to skip the polygon
			and mask images.
		keypoints: Pose keypoints ``[17, 3]``, or ``None`` to skip the
			keypoint image.
	"""
	cv2.imwrite(os.path.join(output_root, "bbox_flir", filename), crop_thermal)
	cv2.imwrite(os.path.join(output_root, "bbox_rgb",  filename), crop_rgb)

	if polygon is not None:
		crop_polygon = crop_rgb.copy()
		cv2.polylines(crop_polygon, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)
		cv2.imwrite(os.path.join(output_root, "bbox_rgb_polygon", filename), crop_polygon)

		mask = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
		cv2.fillPoly(mask, [polygon], 255)
		cv2.imwrite(os.path.join(output_root, "instance_mask", filename), mask)

	if keypoints is not None:
		crop_pose = crop_rgb.copy()
		draw_pose_on_image(crop_pose, keypoints)
		cv2.imwrite(os.path.join(output_root, "bbox_rgb_keypoint", filename), crop_pose)

# endregion


# region Conversion

def convert_sequence(
		dataset   : dict,
		split     : str,
		seq_name  : str,
		camera_id : int,
		id_offset : int,
		model_pose: YOLO,
		model_seg : YOLO,
) -> int:
	"""Extract full-body person instances from one MOT sequence.

	Args:
		dataset: Dataset configuration (an entry of ``DATASETS``).
		split: Dataset split, e.g. ``"train"``.
		seq_name: Name of the sequence directory.
		camera_id: Camera ID written into the output file names.
		id_offset: Value added to every person ID to make it globally unique.
		model_pose: YOLO pose-estimation model.
		model_seg: YOLO instance-segmentation model.

	Returns:
		The largest global person ID written, or ``id_offset`` if none.
	"""
	gt_root     = dataset["gt_root"].format(split=split)
	image_root  = dataset["image_root"].format(split=split)
	gt_path     = os.path.join(gt_root, seq_name, dataset["gt_file"].format(seq=seq_name))
	thermal_dir = os.path.join(image_root, seq_name, dataset["thermal_dir"])
	rgb_dir     = os.path.join(image_root, seq_name, dataset["rgb_dir"])
	min_w, min_h = dataset["min_size"]

	if not os.path.exists(gt_path):
		print(f"Error: ground truth not found at {gt_path}")
		return id_offset

	frame_map  = load_mot_ground_truth(gt_path)
	num_people = len({person_id for detections in frame_map.values() for person_id, _ in detections})
	print(f"\nSequence: {seq_name} | Total IDs: {num_people}")

	thermal_paths  = sorted(glob.glob(os.path.join(thermal_dir, f"*.{dataset['image_ext']}")))
	max_global_pid = id_offset
	# shortcut: pairs the i-th annotated frame with the i-th image, so it assumes every frame has annotations; match by frame number if that ever fails.
	frame_pairs = zip(sorted(frame_map), thermal_paths)
	for frame_id, thermal_path in tqdm(frame_pairs, total=len(thermal_paths), desc=f"Processing {seq_name}"):
		rgb_path = os.path.join(rgb_dir, os.path.basename(thermal_path))
		if not os.path.exists(rgb_path):
			continue

		img_thermal = cv2.imread(thermal_path)
		img_rgb     = cv2.imread(rgb_path)
		if img_thermal is None or img_rgb is None:
			continue
		img_h, img_w = img_thermal.shape[:2]

		for person_id, bbox in frame_map[frame_id]:
			x, y, w, h = clamp_bbox(*bbox, img_w, img_h)
			if w <= min_w or h <= min_h:
				continue

			crop_thermal = img_thermal[y:y + h, x:x + w]
			crop_rgb     = img_rgb[y:y + h, x:x + w]
			if crop_thermal.size == 0 or crop_rgb.size == 0:
				continue

			is_full_body, polygon, keypoints = check_full_body_person(model_pose, model_seg, crop_rgb)
			if not is_full_body:
				continue

			global_pid     = person_id + id_offset
			max_global_pid = max(max_global_pid, global_pid)
			filename       = f"{global_pid:08d}_c{camera_id}_{frame_id:08d}_00.jpg"
			save_instance(dataset["output_root"], filename, crop_thermal, crop_rgb, polygon, keypoints)

	return max_global_pid


def convert_dataset(
		dataset   : dict,
		model_pose: YOLO,
		model_seg : YOLO,
		id_offset : int = -1,
		camera_id : int = -1,
) -> tuple[int, int]:
	"""Extract full-body person instances from every sequence of a dataset.

	Each sequence is treated as a separate camera. Person IDs continue from
	``id_offset`` and camera IDs continue from ``camera_id`` so that several
	datasets can be merged into one Re-ID dataset.

	Args:
		dataset: Dataset configuration (an entry of ``DATASETS``).
		model_pose: YOLO pose-estimation model.
		model_seg: YOLO instance-segmentation model.
		id_offset: Last global person ID used so far.
		camera_id: Last camera ID used so far.

	Returns:
		The updated ``(id_offset, camera_id)``.
	"""
	create_output_dirs(dataset["output_root"])
	for split in dataset["splits"]:
		gt_root   = dataset["gt_root"].format(split=split)
		sequences = sorted(os.listdir(gt_root))
		for seq_name in tqdm(sequences, desc=f"{dataset['name']} {split}"):
			camera_id += 1
			id_offset  = convert_sequence(dataset, split, seq_name, camera_id, id_offset + 1, model_pose, model_seg)
	return id_offset, camera_id

# endregion


# region Main

def main():
	"""Convert all configured datasets into one Re-ID instance dataset."""
	model_pose = YOLO(POSE_MODEL_PATH)
	model_seg  = YOLO(SEG_MODEL_PATH)

	id_offset, camera_id = -1, -1
	for dataset in DATASETS:
		id_offset, camera_id = convert_dataset(dataset, model_pose, model_seg, id_offset, camera_id)
		print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")


if __name__ == "__main__":
	main()

# endregion

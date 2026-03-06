import glob
import os
import shutil

import cv2
import random
import math

import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO

def create_dir_structure(root_path):
	"""Creates the necessary Market-1501 directory structure."""
	dirs = ['bbox_flir', 'bbox_rgb', 'bbox_rgb_polygon', 'bbox_rgb_keypoint', 'instance_mask']
	for d in dirs:
		os.makedirs(os.path.join(root_path, d), exist_ok=True)

def clamp_bbox(x, y, w, h, img_width, img_height):
	"""Ensures bbox coordinates do not go outside image boundaries."""
	x = max(0, int(x))
	y = max(0, int(y))
	w = min(int(w), img_width - x)
	h = min(int(h), img_height - y)
	return x, y, w, h

def check_crop_have_full_person_keypoints(model_pose, model_seg, crop_img_rgb):
	"""
	Check if crop has full person keypoints and return instance segmentation polygon.
	Returns: (bool, polygon_points or None, keypoints or None)
		- (True, polygon, keypoints)  => full keypoints found + segmentation mask extracted
		- (False, None, None)    => keypoints incomplete or no detection
	"""
	REQUIRED_KEYPOINTS   = 17
	VISIBILITY_THRESHOLD = 0.15

	# ── 1. Pose check ────────────────────────────────────────────────────────
	try:
		pose_results = model_pose(crop_img_rgb, conf=0.1, verbose=False)
	except Exception:
		return False, None, None

	has_full_keypoints = False
	found_keypoints    = None
	for result in pose_results:
		if result.keypoints is None or len(result.keypoints) == 0:
			continue

		kpts = result.keypoints.data  # [num_persons, 17, 3]
		if len(kpts) == 0:
			continue

		keypoints     = kpts[0]       # first person, shape [17, 3]
		visible_count = 0
		for i in range(REQUIRED_KEYPOINTS):
			x, y, conf = keypoints[i]
			if conf > VISIBILITY_THRESHOLD and x > 0 and y > 0:
				visible_count += 1

		if visible_count > 5:
			found_keypoints    = keypoints.cpu().numpy()

		if visible_count == REQUIRED_KEYPOINTS:
			has_full_keypoints = True
			break

	if not has_full_keypoints:
		return False, None, found_keypoints

	# ── 2. Segmentation polygon ───────────────────────────────────────────────
	try:
		seg_results = model_seg(crop_img_rgb, conf=0.1, verbose=False)
	except Exception:
		return True, None, found_keypoints  # keypoints OK but seg failed

	for result in seg_results:
		if result.masks is None or result.boxes is None:
			continue

		# Find first 'person' detection (class 0 in COCO)
		classes = result.boxes.cls.cpu().numpy().astype(int)
		person_indices = np.where(classes == 0)[0]
		if len(person_indices) == 0:
			continue

		idx = person_indices[0]  # take the first person

		# masks.xy gives a list of (N,2) polygon arrays, one per detection
		polygon = result.masks.xy[idx]  # shape: [N, 2], float32 pixel coords

		if polygon is not None and len(polygon) > 0:
			polygon = polygon.astype(np.int32)
			return True, polygon, found_keypoints

	# Keypoints passed but no seg polygon found
	return True, None, found_keypoints


def draw_pose_on_image(img, keypoints, conf_threshold=0.15):
	"""
	Draws COCO keypoints and skeleton on the image.
	keypoints: shape (17, 3) -> [x, y, conf]
	"""
	# COCO skeleton connections (0-indexed)
	skeleton = [
		(0, 1), (0, 2), (1, 3), (2, 4),  # Head
		(5, 6), (5, 7), (7, 9),          # Upper body
		(6, 8), (8, 10),
		(5, 11), (6, 12),                # Body
		(11, 12),
		(11, 13), (13, 15),              # Lower body
		(12, 14), (14, 16)
	]
	
	# Color palette (BGR)
	colors = [
		(0, 0, 255),   (0, 255, 0),   (255, 0, 0),   (0, 255, 255), (255, 255, 0),
		(255, 0, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (0, 0, 128),
		(0, 128, 0),   (128, 0, 0),   (255, 165, 0), (0, 165, 255), (165, 0, 255),
		(255, 255, 255), (128, 128, 128)
	]

	# Draw connections (limbs)
	for p1, p2 in skeleton:
		if p1 < len(keypoints) and p2 < len(keypoints):
			x1, y1, c1 = keypoints[p1]
			x2, y2, c2 = keypoints[p2]
			if c1 > conf_threshold and c2 > conf_threshold:
				# Use the color of the second point for the link
				color = colors[p2 % len(colors)]
				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

	# Draw keypoints
	for i, (x, y, conf) in enumerate(keypoints):
		if conf > conf_threshold:
			color = colors[i % len(colors)]
			cv2.circle(img, (int(x), int(y)), 2, color, -1)


def convert_mot_to_reid_tmot_dataset(
		annotation_root,
		image_flir_root,
		image_rgb_root,
		output_root,
		seq_name,
		camera_id   = 1,
		id_offset   = 0,
		model_pose  = None,
		model_seg   = None
):
	# 1. Setup Paths
	# data/tmot_dataset/annotations/train/seq1/thermal/seq1_thermal.txt
	gt_path = os.path.join(annotation_root, seq_name, f'thermal/{seq_name}_thermal.txt')
	# data/tmot_dataset/images/train/seq1/thermal
	img_flir_dir = os.path.join(image_flir_root, seq_name, 'thermal')
	img_rgb_dir  = os.path.join(image_rgb_root, seq_name, 'rgb')

	if not os.path.exists(gt_path):
		print(f"Error: {seq_name}_thermal.txt not found at {gt_path}")
		return

	# Create output directories
	create_dir_structure(output_root)

	# 2. Parse Ground Truth Manually
	# Data Structure: { frame_number: [ (pid, x, y, w, h), ... ] }
	frame_map  = defaultdict(list)
	unique_ids = set()

	with open(gt_path, 'r') as f_read:
		for line in f_read:
			line = line.strip()

			if not line or "nan" in line:  # Skip empty or invalid lines
				continue

			parts = line.split(',')
			# MOT Format: frame, id, left, top, w, h, conf, class, vis
			# 1,0,541.0,206.6,22.4,69.9,1.0,1,-1,-1
			# We need at least 6 columns
			if len(parts) < 6:
				continue

			frame_idx  = int(parts[0])
			pid        = int(float(parts[1]))
			x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

			# Handling optional columns (conf, class, vis)
			# Defaults: class=1 (pedestrian), vis=1.0 (visible)
			class_id = int(parts[7]) if len(parts) >= 8 else 1
			vis      = float(parts[8]) if len(parts) >= 9 else 1.0

			# Filter: Only keep pedestrians (Class 1) and visible objects
			if class_id == 1:
				frame_map[frame_idx].append({
					'pid': pid,
					'bbox': (x, y, w, h)
				})
				unique_ids.add(pid)

	# 3. Split IDs into Train and Test
	unique_ids_list = list(unique_ids)
	random.shuffle(unique_ids_list)

	print(f"\nSequence: {seq_name} | Total IDs: {len(unique_ids_list)}")

	# 4. Processing Images
	# We loop through the dictionary keys (frames)
	sorted_frames  = sorted(frame_map.keys())
	max_global_pid = id_offset
	list_image     = sorted(glob.glob(os.path.join(img_flir_dir, "*.png")))
	pbar           = tqdm(total=len(list_image))
	# extract crops and save
	for frame_idx, img_flir_path in zip(sorted_frames, list_image):
		pbar.set_description(f"Processing {seq_name}")

		# Check if image exists
		basename     = os.path.basename(img_flir_path)
		img_rgb_path = os.path.join(img_rgb_dir, basename)
		if not os.path.exists(img_flir_path) or not os.path.exists(img_rgb_path):
			continue

		img_flir = cv2.imread(img_flir_path)
		img_rgb  = cv2.imread(img_rgb_path)
		if img_flir is None or img_rgb is None:
			continue

		img_h, img_w, _ = img_flir.shape
		detections = frame_map[frame_idx]


		for det in detections:
			pid = det['pid']
			raw_x, raw_y, raw_w, raw_h = det['bbox']

			# Clamp BBox
			x, y, w, h = clamp_bbox(raw_x, raw_y, raw_w, raw_h, img_w, img_h)

			# Skip invalid crops
			if w <= 5 or h <= 5: continue

			# Crop
			crop_bbox_flir = img_flir[y:y+h, x:x+w]
			crop_bbox_rgb  = img_rgb[y:y+h, x:x+w]

			# Check size crop
			if crop_bbox_flir.size == 0 or  crop_bbox_rgb.size == 0:
				continue

			# Check if crop has full person keypoints and get segmentation polygon
			is_valid, polygon, keypoints = check_crop_have_full_person_keypoints(model_pose, model_seg, crop_bbox_rgb)
			if not is_valid:
				continue

			# Global ID adjustment
			global_pid     = pid + id_offset
			max_global_pid = max(max_global_pid, global_pid)

			# Filename Generation
			filename = f"{global_pid:08d}_c{camera_id}_{frame_idx:08d}_00.jpg"

			# polygon is a numpy array of shape [N, 2] (or None if seg failed)
			if polygon is not None:
				# Example: draw polygon on the crop
				crop_bbox_rgb_draw = crop_bbox_rgb.copy()
				cv2.polylines(crop_bbox_rgb_draw, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

				# Example: create a binary mask from polygon
				mask = np.zeros(crop_bbox_rgb.shape[:2], dtype=np.uint8)
				cv2.fillPoly(mask, [polygon], 255)

				# WRITE GB bounding box with polygon drawn
				save_path_rgb_polygon = os.path.join(output_root, 'bbox_rgb_polygon', filename)
				cv2.imwrite(save_path_rgb_polygon, crop_bbox_rgb_draw)

				# WRITE Instance mask (if polygon exists)
				save_path_mask = os.path.join(output_root, 'instance_mask', filename)
				cv2.imwrite(save_path_mask, mask)

			# Draw keypoints
			if keypoints is not None:
				crop_bbox_rgb_keypoints = crop_bbox_rgb.copy()
				draw_pose_on_image(crop_bbox_rgb_keypoints, keypoints)

				# WRITE RGB bounding box with keypoints drawn
				save_path_rgb_keypoint = os.path.join(output_root, 'bbox_rgb_keypoint', filename)
				cv2.imwrite(save_path_rgb_keypoint, crop_bbox_rgb_keypoints)

			# WRITE 2 main crop
			# Flir bounding box
			save_path_flir_bbox = os.path.join(output_root, 'bbox_flir', filename)
			cv2.imwrite(save_path_flir_bbox, crop_bbox_flir)
			# RGB bounding box
			save_path_rgb_bbox = os.path.join(output_root, 'bbox_rgb', filename)
			cv2.imwrite(save_path_rgb_bbox, crop_bbox_rgb)


		pbar.update(1)
	pbar.close()

	return max_global_pid


def main_convert_mot_to_reid_tmot_dataset(
		OUTPUT_DATASET_PATH,
		model_pose,
		model_seg,
		id_offset = -1,
		camera_id = -1
):
	# id_offset = -1  # for unique global IDs across sequences
	# camera_id = -1
	for split in ['train']:
		# This function can be used for command-line execution if needed.
		ANNOTATION_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/annotations/{split}"
		IMAGE_FLIR_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/images/{split}"
		IMAGE_RGB_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/images/{split}"
		# OUTPUT_DATASET_PATH   = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/re_identification/train/tmot/"

		# list of sequences to process
		sequences = os.listdir(os.path.join(ANNOTATION_DATASET_PATH))
		pbar      = tqdm(total=len(sequences))
		for seq in sequences:
			pbar.set_description(f"Processing {seq}")

			# increase camera_id based on seq
			# camera_id = int(seq.replace("seq", ""))
			camera_id +=1

			id_offset = convert_mot_to_reid_tmot_dataset(
				annotation_root = ANNOTATION_DATASET_PATH,
				image_flir_root = IMAGE_FLIR_DATASET_PATH,
				image_rgb_root  = IMAGE_RGB_DATASET_PATH,
				output_root     = OUTPUT_DATASET_PATH,
				seq_name        = seq,
				camera_id       = camera_id,
				id_offset       = id_offset + 1,
				model_pose      = model_pose,
				model_seg       = model_seg
			)
			pbar.update(1)

		pbar.close()

	return id_offset, camera_id


def convert_mot_to_reid_vtmot_dataset(
		annotation_root,
		image_flir_root,
		image_rgb_root,
		output_root,
		seq_name,
		camera_id   = 1,
		id_offset   = 0,
		model_pose  = None,
		model_seg   = None
):
	# 1. Setup Paths
	gt_path = os.path.join(annotation_root, seq_name, f'gt/gt.txt')
	img_flir_dir = os.path.join(image_flir_root, seq_name, 'infrared')
	img_rgb_dir  = os.path.join(image_rgb_root, seq_name, 'visible')

	if not os.path.exists(gt_path):
		print(f"Error: gt.txt not found at {gt_path}")
		return

	# Create output directories
	create_dir_structure(output_root)

	# 2. Parse Ground Truth Manually
	# Data Structure: { frame_number: [ (pid, x, y, w, h), ... ] }
	frame_map  = defaultdict(list)
	unique_ids = set()

	with open(gt_path, 'r') as f_read:
		for line in f_read:
			line = line.strip()

			if not line or "nan" in line:  # Skip empty or invalid lines
				continue

			parts = line.split(',')
			# MOT Format: frame, id, left, top, w, h, conf, class, vis
			# We need at least 6 columns
			if len(parts) < 6:
				continue

			frame_idx  = int(parts[0])
			pid        = int(float(parts[1]))
			x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

			# Handling optional columns (conf, class, vis)
			# Defaults: class=1 (pedestrian), vis=1.0 (visible)
			class_id = int(parts[7]) if len(parts) >= 8 else 1
			vis      = float(parts[8]) if len(parts) >= 9 else 1.0

			# Filter: Only keep pedestrians (Class 1) and visible objects
			if class_id == 1:
				frame_map[frame_idx].append({
					'pid': pid,
					'bbox': (x, y, w, h)
				})
				unique_ids.add(pid)

	# 3. Split IDs into Train and Test
	unique_ids_list = list(unique_ids)
	random.shuffle(unique_ids_list)

	print(f"\nSequence: {seq_name} | Total IDs: {len(unique_ids_list)}")

	# 4. Processing Images
	# We loop through the dictionary keys (frames)
	sorted_frames  = sorted(frame_map.keys())
	max_global_pid = id_offset
	list_image     = sorted(glob.glob(os.path.join(img_flir_dir, "*.jpg")))
	pbar           = tqdm(total=len(list_image))
	# extract crops and save
	for frame_idx, img_flir_path in zip(sorted_frames, list_image):
		pbar.set_description(f"Processing {seq_name}")

		# Check if image exists
		basename     = os.path.basename(img_flir_path)
		img_rgb_path = os.path.join(img_rgb_dir, basename)
		if not os.path.exists(img_flir_path) or not os.path.exists(img_rgb_path):
			continue

		img_flir = cv2.imread(img_flir_path)
		img_rgb  = cv2.imread(img_rgb_path)
		if img_flir is None or img_rgb is None:
			continue

		img_h, img_w, _ = img_flir.shape
		detections = frame_map[frame_idx]


		for det in detections:
			pid = det['pid']
			raw_x, raw_y, raw_w, raw_h = det['bbox']

			# Clamp BBox
			x, y, w, h = clamp_bbox(raw_x, raw_y, raw_w, raw_h, img_w, img_h)

			# Skip invalid crops
			if w <= 100 or h <= 150:
				continue

			# Crop
			crop_bbox_flir = img_flir[y:y+h, x:x+w]
			crop_bbox_rgb  = img_rgb[y:y+h, x:x+w]

			# Check if crop has full person keypoints and get segmentation polygon
			is_valid, polygon, keypoints = check_crop_have_full_person_keypoints(model_pose, model_seg, crop_bbox_rgb)
			if not is_valid:
				continue

			# Global ID adjustment
			global_pid     = pid + id_offset
			max_global_pid = max(max_global_pid, global_pid)

			# Filename Generation
			filename = f"{global_pid:08d}_c{camera_id}_{frame_idx:08d}_00.jpg"

			# polygon is a numpy array of shape [N, 2] (or None if seg failed)
			if polygon is not None:
				# Example: draw polygon on the crop
				crop_bbox_rgb_draw = crop_bbox_rgb.copy()
				cv2.polylines(crop_bbox_rgb_draw, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

				# Example: create a binary mask from polygon
				mask = np.zeros(crop_bbox_rgb.shape[:2], dtype=np.uint8)
				cv2.fillPoly(mask, [polygon], 255)

				# WRITE GB bounding box with polygon drawn
				save_path_rgb_polygon = os.path.join(output_root, 'bbox_rgb_polygon', filename)
				cv2.imwrite(save_path_rgb_polygon, crop_bbox_rgb_draw)

				# WRITE Instance mask (if polygon exists)
				save_path_mask = os.path.join(output_root, 'instance_mask', filename)
				cv2.imwrite(save_path_mask, mask)

			# Draw keypoints
			if keypoints is not None:
				crop_bbox_rgb_keypoints = crop_bbox_rgb.copy()
				draw_pose_on_image(crop_bbox_rgb_keypoints, keypoints)

				# WRITE RGB bounding box with keypoints drawn
				save_path_rgb_keypoint = os.path.join(output_root, 'bbox_rgb_keypoint', filename)
				cv2.imwrite(save_path_rgb_keypoint, crop_bbox_rgb_keypoints)

			# WRITE 2 main crop
			# Flir bounding box
			save_path_flir_bbox = os.path.join(output_root, 'bbox_flir', filename)
			cv2.imwrite(save_path_flir_bbox, crop_bbox_flir)
			# RGB bounding box
			save_path_rgb_bbox = os.path.join(output_root, 'bbox_rgb', filename)
			cv2.imwrite(save_path_rgb_bbox, crop_bbox_rgb)


		pbar.update(1)
	pbar.close()

	return max_global_pid


def main_convert_mot_to_reid_vtmot_dataset(
		OUTPUT_DATASET_PATH,
		model_pose,
		model_seg,
		id_offset = -1,
		camera_id = -1
):
	# id_offset = -1  # for unique global IDs across sequences
	# camera_id = -1
	for split in ['train', 'test']:
		# This function can be used for command-line execution if needed.
		ANNOTATION_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"
		IMAGE_FLIR_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"
		IMAGE_RGB_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"

		# list of sequences to process
		sequences = os.listdir(os.path.join(ANNOTATION_DATASET_PATH))
		pbar      = tqdm(total=len(sequences))
		for seq in sequences:
			pbar.set_description(f"Processing {seq}")

			# increase camera_id based on seq
			camera_id +=1

			id_offset = convert_mot_to_reid_vtmot_dataset(
				annotation_root = ANNOTATION_DATASET_PATH,
				image_flir_root = IMAGE_FLIR_DATASET_PATH,
				image_rgb_root  = IMAGE_RGB_DATASET_PATH,
				output_root     = OUTPUT_DATASET_PATH,
				seq_name        = seq,
				camera_id       = camera_id,
				id_offset       = id_offset + 1,
				model_pose      = model_pose,
				model_seg       = model_seg
			)
			pbar.update(1)

		pbar.close()

	return id_offset, camera_id


def main():
	model_pose           = YOLO("models_zoo/pbvs26_tmot/yolo11x-pose.pt")
	model_seg            = YOLO("models_zoo/pbvs26_tmot/yolo11x-seg.pt")
	id_offset, camera_id = -1, -1
	OUTPUT_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/instance_extraction/tmot/"
	id_offset, camera_id = main_convert_mot_to_reid_tmot_dataset(OUTPUT_DATASET_PATH, model_pose, model_seg, id_offset, camera_id)
	print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")

	OUTPUT_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/instance_extraction/vtmot/"
	id_offset, camera_id = main_convert_mot_to_reid_vtmot_dataset(OUTPUT_DATASET_PATH, model_pose, model_seg, id_offset, camera_id)
	print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")

if __name__ == "__main__":
	main()

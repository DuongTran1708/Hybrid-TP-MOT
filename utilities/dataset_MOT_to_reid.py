import glob
import os
import shutil

import cv2
import random
import math
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO

def create_dir_structure(root_path):
	"""Creates the necessary Market-1501 directory structure."""
	dirs = ['bounding_box_train', 'bounding_box_test', 'query']
	for d in dirs:
		os.makedirs(os.path.join(root_path, d), exist_ok=True)

def clamp_bbox(x, y, w, h, img_width, img_height):
	"""Ensures bbox coordinates do not go outside image boundaries."""
	x = max(0, int(x))
	y = max(0, int(y))
	w = min(int(w), img_width - x)
	h = min(int(h), img_height - y)
	return x, y, w, h

def convert_mot_to_reid_tmot_dataset(
		annotation_root,
		image_root,
		output_root,
		seq_name,
		camera_id   = 1,
		train_ratio = 0.5,
		id_offset   = 0
):
	# 1. Setup Paths
	gt_path = os.path.join(annotation_root, seq_name, f'thermal/{seq_name}_thermal.txt')
	img_dir = os.path.join(image_root, seq_name, 'thermal')

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

	num_train = int(len(unique_ids_list) * train_ratio)
	num_test  = len(unique_ids_list) - num_train
	train_ids = set(unique_ids_list[:num_train])
	test_ids  = set(unique_ids_list[num_train:])

	# DEBUG:
	print(f"\nSequence: {seq_name} | Total IDs: {len(unique_ids_list)} | Train: {len(train_ids)} | Test + Query: {len(test_ids)} ")

	# 4. Processing Images
	# We loop through the dictionary keys (frames)
	sorted_frames  = sorted(frame_map.keys())
	max_global_pid = id_offset
	sqe_id = int(seq_name.replace("seq", ""))
	list_image = sorted(glob.glob(os.path.join(img_dir, "*.png")))
	pbar = tqdm(total=len(list_image))
	# extract crops and save
	for frame_idx, img_path in zip(sorted_frames, list_image):
		pbar.set_description(f"Processing {seq_name}")

		# Check if image exists
		if not os.path.exists(img_path):
			continue

		image = cv2.imread(img_path)
		if image is None:
			continue

		img_h, img_w, _ = image.shape
		detections = frame_map[frame_idx]


		for det in detections:
			pid = det['pid']
			raw_x, raw_y, raw_w, raw_h = det['bbox']

			# Clamp BBox
			x, y, w, h = clamp_bbox(raw_x, raw_y, raw_w, raw_h, img_w, img_h)

			# Skip invalid crops
			if w <= 5 or h <= 5: continue

			# Crop
			crop = image[y:y+h, x:x+w]

			# Global ID adjustment
			global_pid     = pid + id_offset
			max_global_pid = max(max_global_pid, global_pid)

			# Filename Generation
			filename = f"{global_pid:08d}_c{camera_id}_{frame_idx:08d}_00.jpg"

			# Save Logic
			if pid in train_ids:
				save_path = os.path.join(output_root, 'bounding_box_train', filename)
				cv2.imwrite(save_path, crop)
			elif pid in test_ids:
				save_path = os.path.join(output_root, 'bounding_box_test', filename)
				cv2.imwrite(save_path, crop)

		pbar.update(1)
	pbar.close()

	# distribution for test and query with ratio 95:5
	for test_id in test_ids:
		list_crop_imgs = glob.glob(os.path.join(output_root, 'bounding_box_test', f"{test_id + id_offset:08d}_c{camera_id}_*.jpg"))

		num_crop_query      = math.ceil(len(list_crop_imgs) * 5 / 100)  # ratio 95:5
		selected_query_imgs = random.sample(list_crop_imgs, num_crop_query)

		# DEBUG:
		print("*" * 50)
		print(f"Total crop for Test + Query = {len(list_crop_imgs)}")
		print(f"Remain test = {len(list_crop_imgs) - len(selected_query_imgs)}")
		print(f"Query = {len(selected_query_imgs)}")
		print("*" * 50)

		if len(list_crop_imgs) < 3:
			# for img_path in list_crop_imgs:
			# os.remove(img_path)
			pass
			print("Done have query")
			print("^" * 50)
		else:
			for query_img_path in selected_query_imgs:
				filename  = os.path.basename(query_img_path)
				filename  = filename.replace(f"_c{camera_id}_", f"_c{9999-camera_id}_")  # chung per_id thi duoc, nhung phai khac cam_id giua 2 thu muc
				save_path = os.path.join(output_root, 'query', filename)
				if not os.path.exists(save_path):
					shutil.move(query_img_path, save_path)

	return max_global_pid


def main_convert_mot_to_reid_tmot_dataset(OUTPUT_DATASET_PATH, id_offset = -1, camera_id = -1):
	# id_offset = -1  # for unique global IDs across sequences
	# camera_id = -1
	for split in ['train']:
		# This function can be used for command-line execution if needed.
		ANNOTATION_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/annotations/{split}"
		IMAGE_DATASET_PATH      = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/images/{split}"
		# OUTPUT_DATASET_PATH     = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/re_identification/train/tmot/"

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
				image_root      = IMAGE_DATASET_PATH,
				output_root     = OUTPUT_DATASET_PATH,
				seq_name        = seq,
				camera_id       = camera_id,
				train_ratio     = 0.5,
				id_offset       = id_offset + 1
			)
			pbar.update(1)

		pbar.close()

	return id_offset, camera_id


def check_crop_have_full_person_keypoints(model_pose, crop_img_rgb):
	# YOLO11 pose model, there are 17 keypoints
	try:
		results = model_pose(crop_img_rgb, conf=0.25, verbose=False)
	except Exception:
		return False

	# YOLO11 pose has 17 keypoints
	REQUIRED_KEYPOINTS = 17
	VISIBILITY_THRESHOLD = 0.5  # Confidence threshold for visibility

	for result in results:
		if result.keypoints is None or len(result.keypoints) == 0:
			continue

		kpts = result.keypoints.data  # Shape: [num_persons, 17, 3] (x, y, confidence)

		# Process first detected person
		if len(kpts) > 0:
			keypoints = kpts[0]  # Shape: [17, 3]

			# Check if all 17 keypoints are detected and visible
			visible_count = 0
			for i in range(REQUIRED_KEYPOINTS):
				x, y, conf = keypoints[i]
				# Keypoint is visible if confidence > threshold and coordinates are valid
				if conf > VISIBILITY_THRESHOLD and x > 0 and y > 0:
					visible_count += 1

			# Return True only if all keypoints are visible
			if visible_count == REQUIRED_KEYPOINTS:
				return True

	return False


def convert_mot_to_reid_vtmot_dataset(
		annotation_root,
		image_flir_root,
		image_rgb_root,
		output_root,
		seq_name,
		camera_id   = 1,
		train_ratio = 0.5,
		id_offset   = 0,
		model_pose  = None
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

	num_train = int(len(unique_ids_list) * train_ratio)
	num_test  = len(unique_ids_list) - num_train
	train_ids = set(unique_ids_list[:num_train])
	test_ids  = set(unique_ids_list[num_train:])

	# DEBUG:
	print(f"\nSequence: {seq_name} | Total IDs: {len(unique_ids_list)} | Train: {len(train_ids)} | Test + Query: {len(test_ids)} ")

	# 4. Processing Images
	# We loop through the dictionary keys (frames)
	sorted_frames  = sorted(frame_map.keys())
	max_global_pid = id_offset
	list_image     = sorted(glob.glob(os.path.join(img_flir_dir, "*.jpg")))
	pbar           = tqdm(total= len(list_image))
	# extract crops and save
	for frame_idx, img_flir_path in zip(sorted_frames, list_image):
		pbar.set_description(f"Processing {seq_name}")

		# Check if image exists
		basename = os.path.basename(img_flir_path)
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
			crop_flir = img_flir[y:y+h, x:x+w]
			crop_rgb  = img_rgb[y:y+h, x:x+w]

			if not check_crop_have_full_person_keypoints(model_pose, crop_rgb):
				continue

			# Global ID adjustment
			global_pid     = pid + id_offset
			max_global_pid = max(max_global_pid, global_pid)

			# Filename Generation
			filename = f"{global_pid:08d}_c{camera_id}_{frame_idx:08d}_00.jpg"

			# Save Logic
			if pid in train_ids:
				save_path = os.path.join(output_root, 'bounding_box_train', filename)
				cv2.imwrite(save_path, crop_flir)
			elif pid in test_ids:
				save_path = os.path.join(output_root, 'bounding_box_test', filename)
				cv2.imwrite(save_path, crop_flir)
		pbar.update(1)
	pbar.close()

	# distribution for test and query with ratio 95:5
	for test_id in test_ids:
		list_crop_imgs = glob.glob(os.path.join(output_root, 'bounding_box_test', f"{test_id + id_offset:08d}_c{camera_id}_*.jpg"))

		num_crop_query      = math.ceil(len(list_crop_imgs) * 5 / 100)  # ratio 95:5
		selected_query_imgs = random.sample(list_crop_imgs, num_crop_query)

		if len(list_crop_imgs) < 3:
			pass
		else:
			for query_img_path in selected_query_imgs:
				filename  = os.path.basename(query_img_path)
				filename  = filename.replace(f"_c{camera_id}_", f"_c{9999-camera_id}_")  # chung per_id thi duoc, nhung phai khac cam_id giua 2 thu muc
				save_path = os.path.join(output_root, 'query', filename)
				if not os.path.exists(save_path):
					shutil.move(query_img_path, save_path)

	return max_global_pid


def main_convert_mot_to_reid_vtmot_dataset(model_pose, OUTPUT_DATASET_PATH, id_offset = -1, camera_id = -1):
	for split in ['train', 'test']:
		# This function can be used for command-line execution if needed.
		ANNOTATION_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"
		IMAGE_FLIR_DATASET_PATH = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"
		IMAGE_RGB_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/images/{split}"
		# OUTPUT_DATASET_PATH     = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/re_identification/train/tmot/"

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
				train_ratio     = 0.5,
				id_offset       = id_offset + 1,
				model_pose      = model_pose
			)
			pbar.update(1)

		pbar.close()

	return id_offset, camera_id

def main():
	model_pose           = YOLO("models_zoo/pbvs26_tmot/yolo11x-pose.pt")
	id_offset, camera_id = -1, -1
	OUTPUT_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/tmot_dataset_after_checked/re_identification/train/tmot/"
	id_offset, camera_id = main_convert_mot_to_reid_tmot_dataset(OUTPUT_DATASET_PATH, id_offset, camera_id)

	print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")

	OUTPUT_DATASET_PATH  = f"/media/vsw-ws-05/SSD_2/2_dataset/VTMOT/re_identification/train/vtmot/"
	id_offset, camera_id = main_convert_mot_to_reid_vtmot_dataset(model_pose, OUTPUT_DATASET_PATH, id_offset, camera_id)

	print(f"\nFinal id_offset: {id_offset}, camera_id: {camera_id}\n")


# --- Example Usage ---
if __name__ == "__main__":
	main()


"""
In the context of Person Re-Identification (ReID), these three folders represent the standard **Train/Test split protocol**.

Think of the entire process like a **police search**:

### 1. `bounding_box_train` (The Study Material)

* **Role:** This is the data the model uses to **learn**.
* **What happens:** The model looks at these images, reads the file names (e.g., "This is Person A", "This is Person B"), and adjusts its weights to learn what makes Person A look different from Person B.
* **Constraint:** The people (identities) in this folder **must never** appear in the other two folders. If Person #001 is here, they cannot be in `test` or `query`.

### 2. `query` (The "Needle")

* **Role:** These are the **Probe** images or the "questions" you ask the model.
* **Analogy:** This is the photo of a suspect the police are holding up, asking, "Have we seen this person before?"
* **What happens:** During testing, the model takes an image from this folder and tries to find the same person in the `bounding_box_test` folder.

### 3. `bounding_box_test` (The "Haystack" / The Gallery)

* **Role:** This is the **Gallery** or **Database**.
* **Analogy:** This is the massive database of surveillance footage the police are searching through.
* **What happens:** When you run a test, the model compares the `query` image against **every single image** in this folder to rank them by similarity.
* **Content:** This folder contains images of the people in `query` (the correct matches) plus images of random people (distractors) to make the search harder.

---

### How they interact (The Workflow)

1. **Training Phase:**
* **Input:** Only `bounding_box_train`.
* **Goal:** The model learns to generate a numeric "fingerprint" (feature vector) for every image.


2. **Testing Phase (Evaluation):**
* **Step A:** The model creates a fingerprint for a specific image in `query`.
* **Step B:** The model creates fingerprints for **all** images in `bounding_box_test`.
* **Step C:** It calculates the distance (similarity) between the Query fingerprint and all Test fingerprints.
* **Step D:** It sorts the Test images from "most similar" to "least similar."
* **Result:** If the correct match (same person) is at the top of the list, the model is accurate (Rank-1 Accuracy).



### Summary Table

| Folder               | Name          | Purpose                          | Identities                              |
| ---                  | ---           | ---                              | ---                                     |
| `bounding_box_train` | **Train Set** | Learning features/weights.       | **Set A** (e.g., IDs 1-500)             |
| `query`              | **Probe**     | The image to *search for*.       | **Set B** (e.g., IDs 501-1000)          |
| `bounding_box_test`  | **Gallery**   | The database to *search inside*. | **Set B** (Must contain IDs from Query) |

**Key Rule:** Set A (Train) and Set B (Test/Query) must be **disjoint**. The model cannot be tested on people it has already "memorized" during training.
"""
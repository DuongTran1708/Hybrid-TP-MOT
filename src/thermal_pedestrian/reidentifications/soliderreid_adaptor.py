from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

from thermal_pedestrian.core.factory.builder import IDENTIFICATIONS
from thermal_pedestrian.reidentifications.basereid_adaptor import BaseReID_Adapter
from thermal_pedestrian.core.utils.bbox import bbox_xywh_to_xyxy
from solider_reid.model import make_model
from solider_reid.config import cfg


__all__ = [
	'SOLIDER_ReID_Adaptor'
]

# MARK: - SOLIDER ReID Adaptor

@IDENTIFICATIONS.register(name="solider_reid")
class SOLIDER_ReID_Adaptor(BaseReID_Adapter):
	"""Adaptor for SOLIDER model as encoder for re-identification.

	Attributes:
		encoder (SOLIDER_ReID): The SOLIDER model for re-identification.
		buffer_size (int): Size of the buffer for storing recent embeddings.

	"""

	def __init__(self,  model_cfg: str, weight: str, ratio: float, **kwargs):
		"""Initialize adaptor for re-identification.

		Args:
			model (list): List of SOLIDER models for re-identification.
			configs (list): List of config models for re-identification.
			buffer_size (int): Size of the buffer for storing recent embeddings.
		"""
		super().__init__(**kwargs)
		self.weight    = weight
		self.model_cfg = model_cfg
		self.ratio 	   = ratio
		self.init_models()

	def init_models(self):
		self.model = SOLIDER_REID(config_file = self.model_cfg, weights_path=self.weight)

	def extract_feature(self, img: np.ndarray, xywhs: np.ndarray) -> list[Any | np.ndarray]:
		"""Extract feature embeddings from image and detections."""
		img_h, img_w = img.shape[:2]
		img  = Image.fromarray(img)
		dets = bbox_xywh_to_xyxy(xywhs)
		return self.model(dets, img)


class SOLIDER_REID():

	def __init__(self, config_file: str, device: torch.device | str | None = None, weights_path: str | None = None):
		if config_file:
			cfg.merge_from_file(config_file)
		cfg.freeze()
		self.cfg = cfg

		self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.model = make_model(
			self.cfg,
			num_class=1,
			camera_num=0,
			view_num=0,
			semantic_weight=self.cfg.MODEL.SEMANTIC_WEIGHT
		)

		ckpt_path = weights_path or self.cfg.TEST.WEIGHT
		if ckpt_path:
			self.model.load_param(ckpt_path)

		self.model.to(self.device)
		self.model.eval()

		self.preproc = T.Compose([
			T.Resize(self.cfg.INPUT.SIZE_TEST),
			T.ToTensor(),
			T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
		])

	def __call__(self, x: np.ndarray | torch.Tensor, frame: Image.Image | None = None) -> np.ndarray:
		return self.forward(x, frame)

	def preprocessing(self, xyxys: np.ndarray, img: Image.Image) -> torch.Tensor:
		crops = []
		for box in xyxys:
			x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			crop = img.crop((max(0, x1), max(0, y1), x2, y2))
			crops.append(self.preproc(crop))

		if not crops:
			return torch.empty((0, 3, *self.cfg.INPUT.SIZE_TEST), device=self.device)

		batch = torch.stack(crops).to(self.device)
		# Ensure dtype matches model weights
		batch = batch.type_as(next(self.model.parameters()))
		return batch

	def forward(self, x: np.ndarray | torch.Tensor, frame: Image.Image | None = None) -> np.ndarray:
		if frame is not None:
			x = self.preprocessing(x, frame)

		with torch.no_grad():
			features = self.model(x)

		return self.postprocessing(features)

	@staticmethod
	def postprocessing(features: tuple | torch.Tensor) -> np.ndarray:
		if isinstance(features, (tuple, list)):
			features = features[0]
		feats = features
		feats[torch.isinf(feats)] = 1.0
		feats = F.normalize(feats, p=2, dim=1)
		return feats.cpu().numpy()

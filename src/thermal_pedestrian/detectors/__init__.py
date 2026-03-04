#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detector classes.
"""

from __future__ import annotations

from loguru import logger

from .basedetector import *
try:
	from .yolov8_adaptor import *
except ImportError as e:
	# logger.warning(f"YOLOv8 import failed: {e}")
	pass
try:
	from .yolov11_adaptor import *
except ImportError as e:
	logger.warning(f"YOLOv11 import failed: {e}")

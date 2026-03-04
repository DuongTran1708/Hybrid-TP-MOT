# ==================================================================== #
# File name: __init__.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 19/07/2023
#
# ``tracker`` API consists of several trackers that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #

from __future__ import annotations
from loguru import logger

from .basetracker_adapter import BaseTracker_Adapter
try:
	from .sort_adapter import SORT
except ImportError as e:
	# logger.warning(f"SORT import failed: {e}")
	pass
try:
	from .bytetrack_adapter import ByteTrack_Adapter
except ImportError as e:
	# logger.warning(f"ByteTrack import failed: {e}")
	pass
try:
	from .botsort_adapter import BOTSORT_Adapter
except ImportError as e:
	# logger.warning(f"BOTSORT import failed: {e}")
	pass
try:
	from .ocsort_adapter import OCSORT_Adapter
except ImportError as e:
	# logger.warning(f"OCSORT import failed: {e}")
	pass
try:
	from .hybridsort_adapter import HybridSORT_Adapter
except ImportError as e:
	# logger.warning(f"HybridSORT import failed: {e}")
	pass
try:
	from .hybridtpmot_adapter import HybridTPMOT_Adapter
except ImportError as e:
	logger.warning(f"HybridTPMOT import failed: {e}")
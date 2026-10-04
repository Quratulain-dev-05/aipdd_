"""Shared data containers passed between pipeline modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single decoded video frame."""
    image: np.ndarray          # H x W x 3, uint8, BGR
    frame_id: int
    timestamp: datetime
    source_id: str = "cam0"


@dataclass(frozen=True)
class PreprocessedFrame:
    """Frame converted to model input format plus the info needed to undo it."""
    tensor: np.ndarray                 # 1 x 3 x H x W, float32, range [0, 1]
    original_shape: Tuple[int, int]    # (height, width) of the source frame
    scale: float                       # resize ratio used
    pad: Tuple[int, int]               # (pad_x, pad_y) added by letterboxing
    frame: Frame


@dataclass(frozen=True)
class Detection:
    """One detected object, in original-frame pixel coordinates."""
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)


@dataclass(frozen=True)
class Alert:
    """An alert raised by a rule."""
    rule_name: str
    severity: str                      # "info" | "warning" | "critical"
    message: str
    frame_id: int
    timestamp: datetime
    detections: List[Detection] = field(default_factory=list)


Config = Dict[str, object]

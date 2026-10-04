"""Module 2 - ImagePreprocessor: converts raw frames to model-ready tensors."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .data_types import Frame, PreprocessedFrame


class ImagePreprocessor:
    """Letterbox resize, BGR->RGB, scale to [0, 1], HWC->NCHW."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640),
                 pad_value: int = 114) -> None:
        self.target_size = target_size      # (height, width)
        self.pad_value = pad_value

    def process(self, frame: Frame) -> PreprocessedFrame:
        """Prepare one frame for inference."""
        import cv2
        h, w = frame.image.shape[:2]
        th, tw = self.target_size
        scale = min(th / h, tw / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x, pad_y = (tw - new_w) // 2, (th - new_h) // 2
        canvas = np.full((th, tw, 3), self.pad_value, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = canvas[:, :, ::-1]
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return PreprocessedFrame(tensor[None], (h, w), scale, (pad_x, pad_y), frame)

    def process_batch(self, frames: list) -> list:
        """Preprocess a list of frames."""
        return [self.process(f) for f in frames]

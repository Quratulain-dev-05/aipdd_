"""Module 3 - ModelInferenceEngine: loads a detector and returns detections."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .data_types import Detection, PreprocessedFrame


class BaseInferenceEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Load weights into memory (and onto the device)."""

    @abstractmethod
    def predict(self, item: PreprocessedFrame) -> List[Detection]:
        """Run detection on one preprocessed frame."""

    @abstractmethod
    def warmup(self, runs: int = 3) -> None:
        """Run dummy inferences so the first real frame is not slow."""


class ModelInferenceEngine(BaseInferenceEngine):
    """Ultralytics YOLO wrapper. Returns boxes mapped back to original frame pixels."""

    def __init__(self, weights_path: str = "models/yolov8n.pt",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def load_model(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)

    def warmup(self, runs: int = 3) -> None:
        import numpy as np
        if self._model is None:
            self.load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(runs):
            self._model.predict(dummy, verbose=False, device=self.device)

    def predict(self, item: PreprocessedFrame) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("Call load_model() before predict().")
        # Ultralytics does its own letterbox, so the original BGR frame is passed.
        result = self._model.predict(
            item.frame.image, conf=self.conf_threshold, iou=self.iou_threshold,
            device=self.device, verbose=False)[0]
        names = result.names
        detections: List[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0])
            detections.append(Detection(cls, names[cls], float(box.conf[0]), (x1, y1, x2, y2)))
        return detections

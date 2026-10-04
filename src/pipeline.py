"""Wires the four modules together: ingest -> preprocess -> infer -> alert/log."""
from __future__ import annotations

from typing import Optional

from .alert_logger import AlertLogger
from .data_ingestion import BaseDataIngestion
from .image_preprocessor import ImagePreprocessor
from .model_inference_engine import BaseInferenceEngine


class DetectionPipeline:
    def __init__(self, ingestion: BaseDataIngestion, preprocessor: ImagePreprocessor,
                 engine: BaseInferenceEngine, alert_logger: AlertLogger) -> None:
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engine = engine
        self.alert_logger = alert_logger

    def run(self, max_frames: Optional[int] = None) -> int:
        """Process frames until the source ends or max_frames is reached.
        Returns the number of frames processed."""
        if not self.ingestion.open():
            raise RuntimeError("Could not open video source.")
        self.engine.load_model()
        self.engine.warmup()
        count = 0
        try:
            for frame in self.ingestion.frames():
                item = self.preprocessor.process(frame)
                detections = self.engine.predict(item)
                self.alert_logger.handle(frame, detections)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            self.ingestion.close()
            self.alert_logger.close()
        return count

"""Module 1 - DataIngestion: acquires frames from a camera, RTSP stream or file."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator, Optional, Union

from .data_types import Frame


class BaseDataIngestion(ABC):
    """Interface every frame source must implement."""

    @abstractmethod
    def open(self) -> bool:
        """Connect to the source. Returns True on success."""

    @abstractmethod
    def read_frame(self) -> Optional[Frame]:
        """Return the next Frame, or None if the stream ended or failed."""

    @abstractmethod
    def close(self) -> None:
        """Release the source and any resources."""

    @abstractmethod
    def is_opened(self) -> bool:
        """True while the source is connected."""

    def frames(self) -> Iterator[Frame]:
        """Iterate over frames until the source ends."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame


class DataIngestion(BaseDataIngestion):
    """OpenCV-backed ingestion for webcams (int), video files and RTSP URLs (str)."""

    def __init__(self, source: Union[int, str], source_id: str = "cam0",
                 reconnect_attempts: int = 3) -> None:
        self.source = source
        self.source_id = source_id
        self.reconnect_attempts = reconnect_attempts
        self._cap = None
        self._frame_id = 0

    def open(self) -> bool:
        import cv2  # imported lazily so the module can be imported without OpenCV
        self._cap = cv2.VideoCapture(self.source)
        return self._cap.isOpened()

    def read_frame(self) -> Optional[Frame]:
        if self._cap is None:
            raise RuntimeError("Call open() before read_frame().")
        for _ in range(self.reconnect_attempts + 1):
            ok, image = self._cap.read()
            if ok:
                self._frame_id += 1
                return Frame(image, self._frame_id, datetime.now(), self.source_id)
            self.close()
            if not self.open():
                continue
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

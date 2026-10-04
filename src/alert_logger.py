"""Module 4 - AlertLogger: evaluates alert rules, stores events, sends notifications."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Callable, List, Optional

from .data_types import Alert, Detection, Frame

NotifyFn = Callable[[Alert], None]


class AlertLogger:
    """Applies simple label/confidence rules, logs events to SQLite and a log file."""

    def __init__(self, db_path: str = "logs/events.db", log_file: str = "logs/app.log",
                 watch_labels: Optional[List[str]] = None, min_confidence: float = 0.5,
                 notifier: Optional[NotifyFn] = None) -> None:
        self.watch_labels = set(watch_labels or ["person"])
        self.min_confidence = min_confidence
        self.notifier = notifier
        self._logger = logging.getLogger("alert_logger")
        if not self._logger.handlers:
            self._logger.addHandler(logging.FileHandler(log_file))
            self._logger.setLevel(logging.INFO)
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, frame_id INTEGER, "
            "rule TEXT, severity TEXT, message TEXT, detections TEXT)")
        self._db.commit()

    def evaluate(self, frame: Frame, detections: List[Detection]) -> List[Alert]:
        """Return the alerts triggered by this frame's detections."""
        hits = [d for d in detections
                if d.label in self.watch_labels and d.confidence >= self.min_confidence]
        if not hits:
            return []
        labels = ", ".join(sorted({d.label for d in hits}))
        return [Alert("watchlist_object", "warning",
                      f"{len(hits)} object(s) detected: {labels}",
                      frame.frame_id, datetime.now(), hits)]

    def log_event(self, alert: Alert) -> int:
        """Persist an alert; returns the new row id."""
        payload = json.dumps([d.__dict__ for d in alert.detections])
        cur = self._db.execute(
            "INSERT INTO events (ts, frame_id, rule, severity, message, detections) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (alert.timestamp.isoformat(), alert.frame_id, alert.rule_name,
             alert.severity, alert.message, payload))
        self._db.commit()
        self._logger.info("%s | %s | %s", alert.severity, alert.rule_name, alert.message)
        return int(cur.lastrowid)

    def send_notification(self, alert: Alert) -> bool:
        """Call the configured notifier. Returns True if one was called."""
        if self.notifier is None:
            return False
        self.notifier(alert)
        return True

    def handle(self, frame: Frame, detections: List[Detection]) -> List[Alert]:
        """Evaluate rules, log and notify. Convenience entry point for the pipeline."""
        alerts = self.evaluate(frame, detections)
        for a in alerts:
            self.log_event(a)
            self.send_notification(a)
        return alerts

    def close(self) -> None:
        self._db.close()

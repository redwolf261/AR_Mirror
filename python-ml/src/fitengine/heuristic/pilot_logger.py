"""
Append-only JSONL pilot session logger.

Logs every FitEngine session from the moment the widget is shown through
to the final recommendation.  Ground truth (purchased size, return outcome)
is appended later via annotate().

Every logged session with a return outcome becomes brand-calibration data.
The flywheel starts the moment the first user takes a photo.

Log schema
----------
Per funnel-event record:
{
    "record_type":            "event",
    "session_id":             "uuid4",
    "timestamp":              "ISO-8601",
    "event":                  "widget_shown" | "flow_started" | "photo1_captured"
                               | "photo2_captured" | "recommendation_shown"
                               | "recommendation_accepted" | "recommendation_rejected"
                               | "measurement_accepted" | "measurement_rejected",
    "time_to_complete_seconds": float | null   # photo2 only
}

Per session summary record (written after recommendation_shown):
{
    "record_type":            "session",
    "session_id":             "...",
    "timestamp":              "...",
    "kp_front":               [[x,y,c], ...],   # [33,3]
    "kp_side":                [[x,y,c], ...],   # [33,3]
    "height_cm":              178,
    "height_source":          "user_input",
    "measurement_ratios":     { ... },          # from BodyProxyMeasurements
    "predicted_size": {
        "collar":          "16.0",
        "jacket":          "42R",
        "trouser_waist":   "34",
        "confidence_level": "heuristic"
    },
    "actual_size_purchased":  null,   # filled post-purchase via annotate()
    "returned":               null    # filled post-return window
}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PilotLogger:
    """
    Thread-safe append-only JSONL logger for FitEngine pilot sessions.

    Args:
        path : path to the .jsonl log file.
               Creates parent directories if needed.
    """

    def __init__(self, path: str | Path = "data/pilot_log.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session_starts: dict[str, float] = {}  # session_id → widget_shown time

    # ── Funnel event logging ──────────────────────────────────────────────

    def log_event(
        self,
        session_id: str,
        event: str,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Log a single funnel event.

        Events: widget_shown | flow_started | photo1_captured |
                photo2_captured | recommendation_shown | recommendation_accepted
        """
        record: dict = {
            "record_type": "event",
            "session_id":  session_id,
            "timestamp":   _now_iso(),
            "event":       event,
        }

        # Track time-to-complete
        if event == "widget_shown":
            self._session_starts[session_id] = time.monotonic()

        if event == "photo2_captured":
            t0 = self._session_starts.get(session_id)
            record["time_to_complete_seconds"] = (
                round(time.monotonic() - t0, 2) if t0 else None
            )

        if extra:
            record.update(extra)

        self._append(record)

    # ── Session summary logging ───────────────────────────────────────────

    def log_session(
        self,
        session_id: str,
        kp_front: np.ndarray,
        kp_side: np.ndarray,
        height_cm: float,
        predicted_size: dict,
        measurement_ratios: Optional[dict] = None,
        height_source: str = "user_input",
    ) -> None:
        """
        Write a full session summary record after the recommendation is shown.

        Args:
            session_id        : UUID generated at widget_shown.
            kp_front          : [33, 3] front keypoints (normalised).
            kp_side           : [33, 3] side keypoints (normalised).
            height_cm         : user-reported height.
            predicted_size    : {collar, jacket, trouser_waist, confidence_level}.
            measurement_ratios: optional ratio dict from BodyProxyMeasurements.
            height_source     : 'user_input' | 'camera_estimate'.
        """
        record = {
            "record_type":         "session",
            "session_id":          session_id,
            "timestamp":           _now_iso(),
            "kp_front":            kp_front.tolist() if isinstance(kp_front, np.ndarray) else kp_front,
            "kp_side":             kp_side.tolist()  if isinstance(kp_side,  np.ndarray) else kp_side,
            "height_cm":           float(height_cm),
            "height_source":       height_source,
            "measurement_ratios":  measurement_ratios or {},
            "predicted_size":      predicted_size,
            "actual_size_purchased": None,
            "returned":              None,
        }
        self._append(record)

    # ── Ground truth annotation ───────────────────────────────────────────

    def annotate(
        self,
        session_id: str,
        actual_size_purchased: Optional[str],
        returned: Optional[bool],
    ) -> bool:
        """
        Back-fill ground truth on a logged session.

        Reads the entire file, updates the matching session record, rewrites.
        Designed for low-volume post-purchase annotation (not hot path).

        Args:
            session_id            : identifies the session to update.
            actual_size_purchased : e.g. "42R", "16.0", "34".
            returned              : True if item was returned.

        Returns:
            True if session was found and updated, False otherwise.
        """
        records = self._read_all()
        found = False
        for r in records:
            if r.get("record_type") == "session" and r.get("session_id") == session_id:
                r["actual_size_purchased"] = actual_size_purchased
                r["returned"]              = returned
                found = True
                break

        if found:
            self._rewrite(records)
            logger.debug("Annotated session %s → purchased=%s returned=%s",
                         session_id, actual_size_purchased, returned)
        else:
            logger.warning("annotate(): session_id '%s' not found in log.", session_id)

        return found

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def new_session_id() -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        records = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _rewrite(self, records: list[dict]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

"""
Data Flywheel Session Logger — Phase 4 Moat Infrastructure

Every try-on session generates a structured log tuple:
  body_measurements → garment tried → size selected → purchase? → return? → fit_rating

After 10,000+ sessions this dataset becomes the uncopiable moat:
  - Indian body proportion data (no Western company has it)
  - Per-SKU bias corrections learned from real returns
  - Demographic-specific size curves

Schema follows docs/data-strategy.md exactly.
Logs are written as JSONL to logs/fit_measurements.jsonl and
logs/user_feedback.jsonl (append-only, one JSON object per line).

Usage:
    logger = SessionLogger()
    session_id = logger.start_session(sku="SKU-001", size_label="M")
    logger.log_measurements(session_id, body_measurements, fit_decision)
    logger.log_feedback(session_id, purchased=True, returned=False, fit_rating=4)
    logger.close_session(session_id)
"""

from __future__ import annotations

import json
import hashlib
import time
import uuid
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import threading

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
_MEASUREMENTS_LOG = _LOGS_DIR / "fit_measurements.jsonl"
_FEEDBACK_LOG = _LOGS_DIR / "user_feedback.jsonl"
_FAILURES_LOG = _LOGS_DIR / "failures.jsonl"

# Thread-safe write lock (multiple sessions can run concurrently)
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GarmentMetadata:
    sku: str
    size_label: str
    brand: str = ""
    category: str = ""          # t-shirt | shirt | jacket | ...
    shoulder_cm: float = 0.0
    chest_cm: float = 0.0
    length_cm: float = 0.0
    hip_cm: float = 0.0
    inseam_cm: float = 0.0     # trousers/skirts


@dataclass
class BodyMeasurements:
    shoulder_width_cm: float
    chest_width_cm: float
    torso_length_cm: float
    hip_width_cm: float = 0.0
    inseam_cm: float = 0.0
    height_cm: float = 0.0     # filled by Depth Anything V2 when available
    confidence: float = 0.0    # pose confidence (0-1)
    depth_backend: str = "geometric"   # geometric | midas | depth_anything_v2


@dataclass
class FitDecision:
    overall: str                         # TIGHT | GOOD | LOOSE
    shoulder: str = "UNKNOWN"
    chest: str = "UNKNOWN"
    length: str = "UNKNOWN"
    hip: str = "UNKNOWN"


@dataclass
class FrameMetadata:
    distance_proxy: float = 0.0         # depth at torso centre (metres if metric)
    pose_confidence: float = 0.0
    lighting_mean: float = 0.0
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class SessionState:
    session_id: str
    user_id_hash: str
    sku: str
    size_label: str
    started_at: float
    garment_meta: Optional[GarmentMetadata] = None
    body_meas: Optional[BodyMeasurements] = None
    fit_decision: Optional[FitDecision] = None
    frame_meta: Optional[FrameMetadata] = None
    closed: bool = False


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class SessionLogger:
    """
    Append-only JSONL session logger.

    One instance can manage multiple concurrent sessions (thread-safe).
    """

    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or _LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

        logger.info(f"✓ SessionLogger ready — logs → {self.logs_dir}")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        sku: str,
        size_label: str,
        user_id: Optional[str] = None,
        garment_meta: Optional[GarmentMetadata] = None,
    ) -> str:
        """
        Begin a new try-on session.

        Args:
            sku:          Product SKU
            size_label:   "S" | "M" | "L" | "XL" | ...
            user_id:      Optional user identifier (will be hashed for privacy)
            garment_meta: Optional full garment spec

        Returns:
            session_id: UUID string — pass to all subsequent log calls
        """
        session_id = str(uuid.uuid4())
        uid_hash = hashlib.sha256((user_id or session_id).encode()).hexdigest()[:16]

        state = SessionState(
            session_id=session_id,
            user_id_hash=uid_hash,
            sku=sku,
            size_label=size_label,
            started_at=time.time(),
            garment_meta=garment_meta,
        )

        with self._lock:
            self._sessions[session_id] = state

        logger.debug(f"Session started: {session_id[:8]}… sku={sku} size={size_label}")
        return session_id

    def log_measurements(
        self,
        session_id: str,
        body_meas: BodyMeasurements,
        fit_decision: FitDecision,
        frame_meta: Optional[FrameMetadata] = None,
    ) -> None:
        """
        Log body measurements + fit decision for a session.
        Writes to fit_measurements.jsonl immediately.
        """
        state = self._get_session(session_id)
        if state is None:
            return

        state.body_meas = body_meas
        state.fit_decision = fit_decision
        state.frame_meta = frame_meta or FrameMetadata()

        record: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "session_id": session_id,
            "user_id_hash": state.user_id_hash,
            "sku": state.sku,
            "garment_metadata": asdict(state.garment_meta) if state.garment_meta else {},
            "body_measurements": asdict(body_meas),
            "fit_decision": asdict(fit_decision),
            "frame_metadata": asdict(state.frame_meta),
        }

        self._append(_MEASUREMENTS_LOG, record)
        logger.debug(f"Measurements logged for {session_id[:8]}…")

    def log_feedback(
        self,
        session_id: str,
        purchased: bool,
        size_selected: Optional[str] = None,
        returned: bool = False,
        return_reason: Optional[str] = None,
        fit_rating: Optional[int] = None,          # 1-5
        fit_comment: Optional[str] = None,         # "slightly_loose" | "perfect" | ...
    ) -> None:
        """
        Log post-purchase/return outcome.
        This is the ground-truth signal that trains the size correction model.
        Writes to user_feedback.jsonl immediately.
        """
        state = self._get_session(session_id)
        if state is None:
            return

        record: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "session_id": session_id,
            "user_id_hash": state.user_id_hash,
            "sku": state.sku,
            "predicted_fit": state.fit_decision.overall if state.fit_decision else "UNKNOWN",
            "predicted_size": state.size_label,
            "actual_outcome": {
                "purchased": purchased,
                "size_selected": size_selected or state.size_label,
                "returned": returned,
                "return_reason": return_reason,
                "fit_rating": fit_rating,
                "fit_comment": fit_comment,
            },
            "body_measurements": asdict(state.body_meas) if state.body_meas else {},
            "time_since_measurement_sec": time.time() - state.started_at,
        }

        self._append(_FEEDBACK_LOG, record)
        logger.debug(f"Feedback logged for {session_id[:8]}… purchased={purchased} returned={returned}")

    def log_failure(
        self,
        session_id: str,
        failure_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a pipeline failure event (pose detection fail, low light, etc.)"""
        state = self._get_session(session_id)
        record: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "session_id": session_id,
            "user_id_hash": state.user_id_hash if state else "unknown",
            "failure_type": failure_type,
            "context": context or {},
        }
        self._append(_FAILURES_LOG, record)

    def close_session(self, session_id: str) -> None:
        """Release session state from memory."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].closed = True
                del self._sessions[session_id]

    # ------------------------------------------------------------------
    # Stats / analytics helpers
    # ------------------------------------------------------------------

    def session_count(self) -> int:
        """Number of measurement records logged (size of moat so far)."""
        return self._count_lines(_MEASUREMENTS_LOG)

    def feedback_count(self) -> int:
        """Number of purchase/return feedbacks logged."""
        return self._count_lines(_FEEDBACK_LOG)

    def return_rate_for_sku(self, sku: str) -> Optional[float]:
        """
        Compute return rate for a given SKU from the feedback log.
        Returns None if < 10 records (insufficient data).
        """
        total, returned = 0, 0
        feedback_path = self.logs_dir / "user_feedback.jsonl"
        if not feedback_path.exists():
            return None
        try:
            with open(feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("sku") == sku and rec.get("actual_outcome", {}).get("purchased"):
                        total += 1
                        if rec["actual_outcome"].get("returned"):
                            returned += 1
        except Exception:
            return None

        if total < 10:
            return None
        return returned / total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            logger.warning(f"Unknown session_id: {session_id}")
        return state

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        """Thread-safe append of a JSON record to a JSONL file."""
        line = json.dumps(record, ensure_ascii=False)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Module-level singleton (import and use directly)
# ---------------------------------------------------------------------------
_default_logger: Optional[SessionLogger] = None


def get_session_logger() -> SessionLogger:
    """Return the module-level singleton SessionLogger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = SessionLogger()
    return _default_logger


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sl = SessionLogger(logs_dir=Path("/tmp/ar_mirror_test_logs"))

    sid = sl.start_session(
        sku="TSH-001",
        size_label="M",
        user_id="test_user_42",
        garment_meta=GarmentMetadata(
            sku="TSH-001", size_label="M", brand="DemoWear",
            category="t-shirt", shoulder_cm=44.0, chest_cm=50.0, length_cm=65.0,
        ),
    )

    sl.log_measurements(
        sid,
        body_meas=BodyMeasurements(
            shoulder_width_cm=42.3, chest_width_cm=46.8, torso_length_cm=63.5,
            confidence=0.87, depth_backend="depth_anything_v2",
        ),
        fit_decision=FitDecision(overall="GOOD", shoulder="GOOD", chest="GOOD", length="GOOD"),
        frame_meta=FrameMetadata(distance_proxy=1.8, pose_confidence=0.89, lighting_mean=142.3),
    )

    sl.log_feedback(sid, purchased=True, returned=False, fit_rating=4, fit_comment="slightly_loose")
    sl.close_session(sid)

    print(f"Sessions logged : {sl.session_count()}")
    print(f"Feedbacks logged: {sl.feedback_count()}")

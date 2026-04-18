"""
Holistic Tracker (MediaPipe Tasks API)
======================================
Wraps MediaPipe HandLandmarker in a background daemon thread, providing
left/right hand landmarks without blocking the AR render loop.

MediaPipe 0.10+ removed mp.solutions.holistic; this implementation
uses the Tasks API HandLandmarker instead.  Pose landmarks come from
the existing PoseLandmarker in body_aware_fitter — this tracker handles
hands only.

The render loop calls ``get_latest()`` which returns the most recent
result non-blockingly.  A new frame is enqueued via ``enqueue(frame)``
and processed asynchronously.  When ``frame_skip=3``, inference runs
at most every 3rd frame and intermediate frames reuse the last result.

Usage
-----
    from src.core.holistic_tracker import HolisticTracker, HolisticResult

    tracker = HolisticTracker(frame_skip=3)
    tracker.start()

    # inside render loop:
    tracker.enqueue(bgr_frame)
    result = tracker.get_latest()
    if result:
        left_hand  = result.left_hand_landmarks   # list[NormalizedLandmark] or None
        right_hand = result.right_hand_landmarks
"""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).parent.parent.parent   # project root
_HAND_MODEL_CANDIDATES = [
    _ROOT / "hand_landmarker.task",
    _ROOT / "models" / "hand_landmarker.task",
]


def _find_hand_model():
    for p in _HAND_MODEL_CANDIDATES:
        if p.exists():
            return str(p)
    return None


@dataclass
class HolisticResult:
    """Hand landmark results from one HolisticTracker inference call."""
    pose_landmarks:       dict = field(default_factory=dict)   # empty; pose comes from body_aware_fitter
    left_hand_landmarks:  Optional[list[Any]] = None
    right_hand_landmarks: Optional[list[Any]] = None
    confidence: float = 0.0
    frame_idx: int = 0


class HolisticTracker:
    """
    Asynchronous hand landmark tracker using MediaPipe Tasks API HandLandmarker.

    Parameters
    ----------
    frame_skip : int
        Enqueue at most every Nth frame for inference.
    model_complexity : int
        Ignored (kept for API compatibility).
    """

    def __init__(
        self,
        frame_skip: int = 3,
        model_complexity: int = 0,  # kept for API compat, unused
    ) -> None:
        self._frame_skip   = max(1, frame_skip)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)
        self._latest: Optional[HolisticResult]         = None
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread]       = None
        self._running = False
        self._frame_counter = 0
        self._available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the background inference thread. Returns True on success."""
        if self._running:
            return True
        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="holistic_tracker",
        )
        self._running = True
        self._thread.start()
        logger.info("[Holistic] Background tracker started (frame_skip=%d)", self._frame_skip)
        return True

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False
        try:
            self._queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass

    @property
    def available(self) -> bool:
        """True if MediaPipe Holistic loaded successfully."""
        return self._available

    # ------------------------------------------------------------------
    # Public API (called from render loop)
    # ------------------------------------------------------------------

    def enqueue(self, frame_bgr: np.ndarray) -> None:
        """
        Offer a new frame for inference.  Skips frames according to
        frame_skip and drops silently if the queue is full (non-blocking).
        """
        self._frame_counter += 1
        if self._frame_counter % self._frame_skip != 0:
            return
        try:
            self._queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass  # drop frame — prefer fresh over blocking

    def get_latest(self) -> Optional[HolisticResult]:
        """Return the most recent result, or None if no inference has run yet."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        model_path = _find_hand_model()
        if model_path is None:
            logger.warning(
                "[Holistic] hand_landmarker.task not found — hand tracking disabled."
            )
            self._running = False
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core   import base_options as bo
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import \
                VisionTaskRunningMode as RunningMode

            options = HandLandmarkerOptions(
                base_options = bo.BaseOptions(model_asset_path=model_path),
                running_mode = RunningMode.IMAGE,
                num_hands    = 2,
                min_hand_detection_confidence = 0.5,
                min_hand_presence_confidence  = 0.5,
                min_tracking_confidence       = 0.5,
            )
            landmarker = HandLandmarker.create_from_options(options)
            self._available = True
            logger.info("[Holistic] HandLandmarker ready (%s)", model_path)
        except Exception as exc:
            logger.warning("[Holistic] HandLandmarker init failed: %s", exc)
            self._running = False
            return

        frame_idx = 0
        while self._running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img    = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                det    = landmarker.detect(mp_img)
                result = self._parse(det, frame_idx)
                with self._lock:
                    self._latest = result
            except Exception as exc:
                logger.debug("[Holistic] Inference error: %s", exc)

            frame_idx += 1

        try:
            landmarker.close()
        except Exception:
            pass
        logger.info("[Holistic] Background tracker stopped")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, det: Any, frame_idx: int) -> HolisticResult:
        """
        Map HandLandmarker detection to HolisticResult.

        MP Task handedness labels ("Left"/"Right") are from the person's
        perspective, which is mirrored from a camera image.  We swap them
        so left_hand_landmarks = left side of the camera frame.
        """
        left_hand  = None
        right_hand = None

        if det.hand_landmarks and det.handedness:
            for landmarks, handedness_list in zip(det.hand_landmarks, det.handedness):
                if not handedness_list:
                    continue
                label = handedness_list[0].category_name  # "Left" | "Right"
                # MP "Left" = person's left = camera-right; swap for display coords
                if label == "Left":
                    right_hand = list(landmarks)
                else:
                    left_hand = list(landmarks)

        return HolisticResult(
            pose_landmarks       = {},
            left_hand_landmarks  = left_hand,
            right_hand_landmarks = right_hand,
            confidence           = 1.0 if (left_hand or right_hand) else 0.0,
            frame_idx            = frame_idx,
        )

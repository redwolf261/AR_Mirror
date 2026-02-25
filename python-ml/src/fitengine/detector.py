"""
RTMPose keypoint detector via rtmlib (Apache 2.0).

PoseDetector wraps rtmlib.Body and outputs a normalised body-33 keypoint
array.  Model weights are downloaded to ~/.fitengine/models/ on first use.

Usage:
    detector = PoseDetector()
    kp33 = detector.detect(bgr_image)   # np.ndarray [33, 3] (x, y, conf), [0, 1]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils.joints import rtmpose133_to_body33, normalize_keypoints

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.path.expanduser("~/.fitengine/models"))

# RTMPose-m ONNX — downloaded by rtmlib automatically
_RTMLIB_BACKEND = "onnxruntime"
_RTMLIB_DEVICE  = "cuda"   # falls back to cpu if cuda unavailable


class PoseDetector:
    """
    RTMPose-based 2D keypoint detector.

    Detects up to one person per frame and returns body-33 keypoints
    normalised to [0, 1] image coordinates.
    """

    def __init__(
        self,
        backend: str = _RTMLIB_BACKEND,
        device: str = _RTMLIB_DEVICE,
        score_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            backend         : 'onnxruntime' or 'openvino'.
            device          : 'cuda' or 'cpu'.  Gracefully degrades to cpu.
            score_threshold : minimum joint confidence to keep.
        """
        self.score_threshold = score_threshold
        self._model = self._load_model(backend, device)

    def _load_model(self, backend: str, device: str):
        try:
            from rtmlib import Wholebody  # type: ignore
        except ImportError as e:
            raise ImportError(
                "rtmlib is required for PoseDetector. "
                "Install: pip install rtmlib"
            ) from e

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Try GPU first, fall back to CPU.
        # Use Wholebody mode='balanced' (rtmw whole-body, 133 keypoints).
        # Body('balanced') uses body7 which only gives 17 COCO keypoints.
        for dev in (device, "cpu"):
            try:
                model = Wholebody(
                    mode="balanced",
                    to_openpose=False,
                    backend=backend,
                    device=dev,
                )
                logger.info("PoseDetector loaded (backend=%s device=%s)", backend, dev)
                return model
            except Exception as exc:
                if dev == "cpu":
                    raise
                logger.warning("GPU init failed (%s) — falling back to CPU for PoseDetector.", exc)

        raise RuntimeError("PoseDetector: could not load model on any device.")

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run RTMPose on a single BGR image.

        Args:
            img_bgr : [H, W, 3] uint8 BGR image.

        Returns:
            kp33 : [33, 3] float32.  Columns: (x, y, conf), x/y in [0, 1].
                   Joints with conf < score_threshold have x=y=0 and conf=0.
        """
        H, W = img_bgr.shape[:2]
        # rtmlib Body.__call__ returns (keypoints[N,133,2], scores[N,133])
        keypoints, scores = self._model(img_bgr)

        if keypoints is None or len(keypoints) == 0:
            logger.debug("PoseDetector: no person detected.")
            return np.zeros((33, 3), dtype=np.float32)

        # Take the person with the highest mean confidence
        mean_scores = np.mean(scores, axis=-1)   # [N]
        best_idx = int(np.argmax(mean_scores))

        kp_raw = np.array(keypoints[best_idx], dtype=np.float32)  # [133, 2]
        sc     = np.array(scores[best_idx],    dtype=np.float32)  # [133]

        kp133 = np.concatenate([
            kp_raw.reshape(133, 2),
            sc.reshape(133, 1),
        ], axis=-1)  # [133, 3]

        # Map to body-33
        kp33_px = rtmpose133_to_body33(kp133)  # [33, 3] in pixels

        # Normalise
        kp33 = normalize_keypoints(kp33_px, W, H)  # → [0, 1]

        # Zero out low-confidence joints
        low_conf = kp33[:, 2] < self.score_threshold
        kp33[low_conf, :2] = 0.0

        return kp33

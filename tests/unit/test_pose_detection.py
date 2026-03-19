"""Deterministic pose-preprocessing checks without live camera dependencies."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _create_test_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 120), (440, 420), (0, 255, 0), -1)
    cv2.circle(frame, (320, 85), 40, (255, 220, 170), -1)
    return frame


def test_pose_model_artifacts_exist() -> None:
    assert (ROOT / "pose_landmarker_lite.task").exists()
    assert (ROOT / "hand_landmarker.task").exists()


def test_bgr_to_rgb_roundtrip_preserves_shape() -> None:
    bgr = _create_test_frame()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr_roundtrip = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    assert bgr.shape == rgb.shape
    assert bgr_roundtrip.shape == bgr.shape


def test_mirror_flip_keeps_dimensions_and_dtype() -> None:
    frame = _create_test_frame()
    flipped = cv2.flip(frame, 1)

    assert flipped.shape == frame.shape
    assert flipped.dtype == frame.dtype

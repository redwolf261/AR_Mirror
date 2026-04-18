"""Deterministic sizing math tests without camera or heavy ML dependencies."""

import json
import time

import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
REFERENCE_HEAD_CM = 23.0


def _synthetic_landmarks() -> dict[int, dict[str, float]]:
    return {
        0: {"x": 0.50, "y": 0.25, "z": 0.0, "visibility": 0.95},
        11: {"x": 0.42, "y": 0.35, "z": 0.0, "visibility": 0.92},
        12: {"x": 0.58, "y": 0.35, "z": 0.0, "visibility": 0.93},
        23: {"x": 0.43, "y": 0.62, "z": 0.0, "visibility": 0.90},
        24: {"x": 0.57, "y": 0.62, "z": 0.0, "visibility": 0.91},
    }


def _scale_factor(landmarks: dict[int, dict[str, float]]) -> float:
    nose_y = landmarks[0]["y"]
    shoulder_mid_y = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
    head_height_pixels = abs(nose_y - shoulder_mid_y) * FRAME_HEIGHT
    return REFERENCE_HEAD_CM / head_height_pixels


def _shoulder_width_cm(landmarks: dict[int, dict[str, float]], scale: float) -> float:
    left = landmarks[11]
    right = landmarks[12]
    dx = right["x"] - left["x"]
    dy = right["y"] - left["y"]
    shoulder_pixels = np.sqrt(dx ** 2 + dy ** 2) * FRAME_WIDTH
    return float(shoulder_pixels * scale)


def _torso_length_cm(landmarks: dict[int, dict[str, float]], scale: float) -> float:
    shoulder_mid_y = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
    hip_mid_y = (landmarks[23]["y"] + landmarks[24]["y"]) / 2
    torso_pixels = abs(shoulder_mid_y - hip_mid_y) * FRAME_HEIGHT
    return float(torso_pixels * scale)


def _fit_decision(garment_shoulder_cm: float, body_shoulder_cm: float) -> str:
    ease_shoulder = 2.0
    diff = garment_shoulder_cm - body_shoulder_cm
    if diff < ease_shoulder:
        return "TIGHT"
    if diff <= (ease_shoulder + 4.0):
        return "GOOD"
    return "LOOSE"


def test_synthetic_measurements_are_in_expected_range() -> None:
    landmarks = _synthetic_landmarks()
    scale = _scale_factor(landmarks)
    shoulder_cm = _shoulder_width_cm(landmarks, scale)
    torso_cm = _torso_length_cm(landmarks, scale)

    assert 35.0 < shoulder_cm < 55.0
    assert 45.0 < torso_cm < 75.0


def test_scale_factor_is_positive_and_finite() -> None:
    scale = _scale_factor(_synthetic_landmarks())
    assert np.isfinite(scale)
    assert scale > 0


def test_fit_decision_thresholds() -> None:
    body = 40.0
    assert _fit_decision(41.5, body) == "TIGHT"
    assert _fit_decision(44.0, body) == "GOOD"
    assert _fit_decision(47.0, body) == "LOOSE"


def test_log_entry_is_json_serializable() -> None:
    landmarks = _synthetic_landmarks()
    scale = _scale_factor(landmarks)
    shoulder_cm = _shoulder_width_cm(landmarks, scale)
    torso_cm = _torso_length_cm(landmarks, scale)

    entry = {
        "timestamp": time.time(),
        "event_type": "fit_result",
        "data": {
            "decision": _fit_decision(44.0, shoulder_cm),
            "measurements": {
                "shoulder_cm": shoulder_cm,
                "torso_cm": torso_cm,
                "confidence": 0.92,
            },
            "garment": {
                "sku": "TEST-001",
                "size_label": "M",
                "shoulder_cm": 44.0,
            },
        },
    }

    payload = json.dumps(entry)
    assert "fit_result" in payload

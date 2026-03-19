"""Deterministic occlusion-mask checks for hand/forearm layering."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def _load_hand_occluder_class():
    module_path = ROOT / "src" / "core" / "hand_occluder.py"
    spec = importlib.util.spec_from_file_location("hand_occluder_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HandOccluder


@dataclass
class _HandLm:
    x: float
    y: float


def _synthetic_hand(cx: float, cy: float, spread: float = 0.02) -> list[_HandLm]:
    points: list[_HandLm] = []
    for i in range(21):
        ox = ((i % 5) - 2) * spread * 0.5
        oy = ((i // 5) - 2) * spread * 0.5
        points.append(_HandLm(cx + ox, cy + oy))
    return points


def test_hand_occluder_returns_zero_mask_without_landmarks() -> None:
    HandOccluder = _load_hand_occluder_class()
    occluder = HandOccluder()
    mask = occluder.make_mask(
        frame_shape=(480, 640, 3),
        hand_lm_left=None,
        hand_lm_right=None,
        pose_lm={},
    )

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 0


def test_hand_occluder_generates_nonzero_mask_with_pose_and_hand() -> None:
    HandOccluder = _load_hand_occluder_class()
    occluder = HandOccluder()
    pose = {
        13: {"x": 0.40, "y": 0.45, "visibility": 0.95},
        15: {"x": 0.36, "y": 0.58, "visibility": 0.95},
    }
    hand = _synthetic_hand(0.36, 0.60)

    mask = occluder.make_mask(
        frame_shape=(480, 640, 3),
        hand_lm_left=hand,
        hand_lm_right=None,
        pose_lm=pose,
    )

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_hand_occluder_is_stable_for_same_inputs() -> None:
    HandOccluder = _load_hand_occluder_class()
    occluder = HandOccluder()
    pose = {
        14: {"x": 0.62, "y": 0.44, "visibility": 0.96},
        16: {"x": 0.66, "y": 0.57, "visibility": 0.96},
    }
    hand = _synthetic_hand(0.66, 0.59)

    mask_a = occluder.make_mask((480, 640, 3), None, hand, pose)
    mask_b = occluder.make_mask((480, 640, 3), None, hand, pose)

    assert mask_a.shape == mask_b.shape
    assert int(np.abs(mask_a.astype(np.int16) - mask_b.astype(np.int16)).sum()) == 0

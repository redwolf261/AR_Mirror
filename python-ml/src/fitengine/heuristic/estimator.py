"""
Pure geometry heuristic: keypoint ratios → shirt/suit size.

Intentionally imperfect — this is a market probe, not a product.
Phase 1 goal: ship to a known contact and confirm whether real users
complete a 2-photo flow.  Zero ML dependency.

The sizing logic below uses empirical threshold tables that map
scale-invariant body ratios to men's formalwear size classes.
Thresholds are overridable via a brand chart JSON.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..measurements import BodyProxyMeasurements
from ..size_chart import SizeChart

logger = logging.getLogger(__name__)


class HeuristicEstimator:
    """
    Phase 1 heuristic size estimator.

    No ML.  Computes scale-invariant body ratios from 2-view keypoints
    and maps them to collar / jacket / trouser sizes using the
    threshold tables in the brand's JSON chart.

    This estimator is replaced silently by DualViewRegressor in Phase 2.
    The FitEnginePipeline calls self._estimator.predict() — same interface.
    """

    def __init__(self, chart: SizeChart | None = None) -> None:
        self._chart = chart or SizeChart("generic")

    def predict(
        self,
        kp_front: np.ndarray,
        kp_side:  np.ndarray,
        height_cm: float,
    ) -> dict:
        """
        Estimate men's formalwear sizes from two-view keypoints.

        Args:
            kp_front  : [33, 3] (x, y, conf) front view, normalised [0,1].
            kp_side   : [33, 3] side view, normalised [0,1].
            height_cm : user-reported standing height in cm.

        Returns:
            {
                "collar":           "16.0",
                "jacket":           "42R",
                "trouser_waist":    "34",
                "confidence_level": "heuristic",
            }
        """
        m = BodyProxyMeasurements.from_keypoints(kp_front, kp_side)

        if not m.valid:
            logger.warning(
                "HeuristicEstimator: low-confidence keypoints — "
                "returning default medium sizes."
            )
            return {
                "collar":           "15.5",
                "jacket":           "40",
                "trouser_waist":    "32",
                "confidence_level": "heuristic",
            }

        # Adjust hip ratio for height — taller men tend to have longer inseam
        # but similar waist-to-torso ratios; we shift trouser index slightly
        height_adj = _height_trouser_shift(height_cm)

        # Get base size prediction from chart thresholds
        result = self._chart.predict_size(m, logits=None)

        # Apply height adjustment to trouser
        trouser_classes = self._chart._data["trouser_sizes"]["classes"]
        t_idx = self._chart._data["trouser_sizes"]["class_index"].get(
            result["trouser_waist"], 3
        )
        t_idx_adj = max(0, min(t_idx + height_adj, len(trouser_classes) - 1))
        result["trouser_waist"] = str(int(trouser_classes[t_idx_adj]))

        # Jacket length suffix (S / R / L) based on height
        result["jacket"] = _add_jacket_length(result["jacket"], height_cm)

        return result


def estimate_shirt_size(
    kp_front:  np.ndarray,
    kp_side:   np.ndarray,
    height_cm: float,
    chart: Optional[SizeChart] = None,
) -> dict:
    """
    Module-level convenience wrapper around HeuristicEstimator.

    Args:
        kp_front  : np.ndarray [33, 3] (x, y, confidence)
        kp_side   : np.ndarray [33, 3]
        height_cm : float, required for trouser/sleeve sizing

    Returns:
        {
            "collar":           "16.0",
            "jacket":           "42R",
            "trouser_waist":    "34",
            "confidence_level": "heuristic"
        }
    """
    return HeuristicEstimator(chart).predict(kp_front, kp_side, height_cm)


# ── Height-based adjustments ──────────────────────────────────────────────

def _height_trouser_shift(height_cm: float) -> int:
    """
    Return an integer shift to apply to the trouser class index based on height.

    Very tall men (>188cm) tend to need one size larger trouser to account
    for proportional waist.  Very short men (<163cm) shift down by one.
    """
    if height_cm >= 190:
        return +1
    if height_cm >= 183:
        return 0
    if height_cm >= 170:
        return 0
    if height_cm >= 163:
        return -1
    return -1


def _add_jacket_length(jacket_size: str, height_cm: float) -> str:
    """
    Append S / R / L length suffix to jacket size.

    Short  (< 170cm) → S
    Regular (170–182) → R
    Long   (> 182cm) → L
    """
    if height_cm < 170:
        length = "S"
    elif height_cm <= 182:
        length = "R"
    else:
        length = "L"
    # Avoid double-suffix if already present
    clean = jacket_size.rstrip("SRL")
    return f"{clean}{length}"

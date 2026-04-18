"""
Anthropometric β sampler — Phase 2 data generation.

Samples STAR β vectors whose population distribution matches ANSUR II
(US Army Anthropometric Survey, public domain) men's measurements.

DO NOT generate synthetic data until this module passes Gate 1:
    python -m fitengine.data_gen.anthropometrics --check-distribution --n 50000
    # Expected: all jacket size classes within [15%, 40%]
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSUR II men's percentiles (public domain — US Army ANSUR II study 2012)
# Format: [p5, p25, p50, p75, p95]
# ---------------------------------------------------------------------------
ANSUR_MENS_PERCENTILES: dict[str, list[float]] = {
    "stature_cm":         [163.0, 170.0, 175.0, 180.0, 188.0],
    "chest_circ_cm":      [ 87.0,  94.0,  99.0, 106.0, 116.0],
    "waist_circ_cm":      [ 74.0,  82.0,  89.0,  97.0, 112.0],
    "shoulder_width_cm":  [ 38.0,  41.0,  43.0,  46.0,  50.0],
    "hip_circ_cm":        [ 88.0,  95.0,  99.0, 105.0, 115.0],
    "inseam_cm":          [ 72.0,  77.0,  80.0,  83.0,  88.0],
}


class AnthropometricBetaSampler:
    """
    Sample STAR β[10] vectors with population-realistic distribution.

    Sampling strategy:
      50% — population-realistic (ANSUR percentiles → β correlation)
      25% — extremes (p5-p10 and p90-p95 tails)
      15% — correlated body types (tall+broad, short+lean)
      10% — single-dimension extremes (β ±3.0)

    β weighting for men's formalwear:
      β[0]: overall size      (highest weight)
      β[1]: height            (high weight)
      β[2]: chest breadth     (high weight)
      β[3]: shoulder width    (high weight)
      β[4]: waist proportion  (medium weight)
      β[5-9]: secondary shape (lower weight, hip/thigh less critical)
    """

    # TODO (Phase 2, Month 2): implement full β correlation from ANSUR statistics.
    # For now this is a placeholder with the correct interface.

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        raise NotImplementedError(
            "AnthropometricBetaSampler is a Phase 2 implementation target. "
            "Implement after Gate 1 (anthropometric distribution check)."
        )

    def sample(self, n: int) -> np.ndarray:
        """
        Sample n β vectors.

        Returns:
            beta : [n, 10] float32
        """
        raise NotImplementedError

    def check_distribution(self, n: int = 50_000) -> dict:
        """
        Assert all jacket size classes appear in [15%, 40%] range.

        Returns:
            dict with class distribution stats. Raises AssertionError if
            any class is outside the gate.
        """
        raise NotImplementedError


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Anthropometric distribution check")
    parser.add_argument("--check-distribution", action="store_true")
    parser.add_argument("--n", type=int, default=50_000)
    args = parser.parse_args()

    if args.check_distribution:
        sampler = AnthropometricBetaSampler()
        stats = sampler.check_distribution(args.n)
        print("Distribution check passed:", stats)


if __name__ == "__main__":
    _cli()

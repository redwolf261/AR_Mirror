"""
Per-view data augmentation — Phase 2.

Applied independently to each view's keypoints and width curve:
  - per-view jitter (small random translation / scale)
  - horizontal flip (front view only, with L/R swap)
  - random joint dropout

# TODO (Phase 2, Month 2)
"""

from __future__ import annotations
import numpy as np


class ViewAugmenter:
    """Phase 2 augmentation stub."""

    def augment_front(self, kp33: np.ndarray, width32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def augment_side(self, kp33: np.ndarray, width32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

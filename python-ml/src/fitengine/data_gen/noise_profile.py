"""
RTMPose-calibrated noise profile — Phase 2 data generation.

Write and validate this module BEFORE generating any synthetic data.
Noise model calibrated from real RTMPose error distributions on video data.

Validation gate (Gate 2):
    python -m fitengine.data_gen.noise_profile --visualize
    # Expected: Laplace tails visible vs Gaussian, swap events at low-conf joints
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class LaplaceJointNoise:
    """
    Laplace-distributed 2D keypoint noise.

    Heavier tails than Gaussian — matches RTMPose real error distribution.

    Params:
        scale_high_conf = 1.5 px  (conf > 0.7)
        scale_low_conf  = 5.0 px  (conf < 0.3)
    """
    # TODO (Phase 2, Month 2)
    scale_high_conf: float = 1.5
    scale_low_conf:  float = 5.0

    def apply(self, kp33: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SymmetricSwapNoise:
    """
    Left/Right joint confusion (e.g. left knee ↔ right knee).

    Applied when joint confidence < 0.3 with probability prob_at_low_conf.

    Params:
        prob_at_low_conf = 0.12
    """
    # TODO (Phase 2, Month 2)
    prob_at_low_conf: float = 0.12

    _SWAP_PAIRS: list[tuple[int, int]] = [
        (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    ]

    def apply(self, kp33: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TruncationAugment:
    """
    Zeros out lower body keypoints — simulates badly framed photos.

    Params:
        prob = 0.25
    """
    # TODO (Phase 2, Month 2)
    prob: float = 0.25

    _LOWER_BODY_JOINTS: list[int] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 28, 29, 30, 31]

    def apply(self, kp33: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SideViewBias:
    """
    Random forward lean — users rarely stand perfectly orthogonal.

    Params:
        lean_std = 8.0 degrees
    """
    # TODO (Phase 2, Month 2)
    lean_std: float = 8.0

    def apply(self, kp33: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CameraDistortion:
    """
    Radial distortion on 2D coordinates.

    Params:
        k1_std = 0.05
    """
    # TODO (Phase 2, Month 2)
    k1_std: float = 0.05

    def apply(self, kp33: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SilhouetteNoise:
    """
    ±3% torso-width noise — simulates thick fabric, loose clothing.

    Params:
        width_std = 0.03
    """
    # TODO (Phase 2, Month 2)
    width_std: float = 0.03

    def apply(self, width_curve: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoiseProfile:
    """
    Composite noise pipeline applied to a single view's keypoints + width curve.

    Usage:
        profile = NoiseProfile()
        kp33_noisy, width_noisy = profile.apply(kp33, width_curve32, view="front")
    """

    def __init__(self) -> None:
        self.joint_noise      = LaplaceJointNoise()
        self.swap_noise       = SymmetricSwapNoise()
        self.truncation       = TruncationAugment()
        self.side_bias        = SideViewBias()
        self.distortion       = CameraDistortion()
        self.silhouette_noise = SilhouetteNoise()

    def apply(
        self,
        kp33: np.ndarray,
        width_curve32: np.ndarray,
        view: Literal["front", "side"] = "front",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply full noise pipeline to keypoints and width curve.

        Args:
            kp33          : [33, 3] — normalised keypoints (x, y, conf).
            width_curve32 : [32]    — normalised width curve.
            view          : 'front' or 'side'.

        Returns:
            (kp33_noisy, width32_noisy)
        """
        # TODO (Phase 2, Month 2): chain all noise transformers here
        raise NotImplementedError


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.visualize:
        # TODO: show Laplace vs Gaussian comparison plot
        raise NotImplementedError("Visualisation not yet implemented (Phase 2).")


if __name__ == "__main__":
    _cli()

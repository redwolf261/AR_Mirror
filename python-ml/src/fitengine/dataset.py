"""
HDF5 PoseDataset — Phase 2 training data loader.

Loads synthetic (and Blender) samples from HDF5 files.

HDF5 schema per record:
    kp_front[33,3]    normalised keypoints, front view
    kp_side[33,3]     normalised keypoints, side view
    width_front[32]   silhouette width curve, front
    width_side[32]    silhouette width curve, side
    height_norm[1]    (height_cm - 175.0) / 12.0
    theta[72]         STAR pose parameters
    beta[10]          STAR shape parameters
    collar_class      int 0..7
    jacket_class      int 0..7
    trouser_class     int 0..9

# TODO (Phase 2, Month 2): implement after Gate 3 (10k smoke test passes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    PyTorch Dataset wrapping an HDF5 file of synthetic pose samples.

    # TODO (Phase 2, Month 2)
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        val_split: float = 0.05,
        height_dropout: float = 0.05,
        seed: int = 42,
    ) -> None:
        raise NotImplementedError(
            "PoseDataset is a Phase 2 implementation target. "
            "Implement after Gate 3 (10k HDF5 smoke test passes)."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                "x":            torch.Tensor [263],
                "beta":         torch.Tensor [10],
                "theta":        torch.Tensor [72],
                "collar_class": torch.LongTensor [],
                "jacket_class": torch.LongTensor [],
                "trouser_class":torch.LongTensor [],
            }
        """
        raise NotImplementedError

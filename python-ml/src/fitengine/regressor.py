"""
DualViewRegressor — Phase 2 ML model.

Input:  263-dim vector
  kp_front   [99]  = 33 joints × 3 (x, y, conf), normalised
  kp_side    [99]  = 33 joints × 3 (x, y, conf), normalised
  width_front[32]  = silhouette width curve, front view
  width_side [32]  = silhouette width curve, side view
  height_norm[ 1]  = (height_cm - 175.0) / 12.0

Architecture:
  Linear(263, 1024) + BatchNorm1d + ReLU + Dropout(0.2)
  Linear(1024, 1024) + BatchNorm1d + ReLU + Dropout(0.2)
  Linear(1024, 512)  + BatchNorm1d + ReLU
  head_beta:  Linear(512, 10)    → β intermediate
  head_theta: Linear(512, 72)    → θ (reprojection loss only)

Parameters: ~9M
VRAM at inference: ~36MB fp32 / ~18MB fp16

# TODO (Phase 2, Month 2): implement after Gate 3 (10k sample smoke test passes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualViewRegressor(nn.Module):
    """
    Phase 2 dual-view body shape regressor.

    Architecture frozen per plan (do not change without updating configs/default.yaml).
    """

    INPUT_DIM = 263

    def __init__(
        self,
        input_dim:  int = 263,
        hidden:     list[int] | None = None,
        beta_dim:   int = 10,
        theta_dim:  int = 72,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        hidden = hidden or [1024, 1024, 512]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i, h in enumerate(hidden):
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head_beta  = nn.Linear(in_dim, beta_dim)
        self.head_theta = nn.Linear(in_dim, theta_dim)

        # TODO (Phase 2, Month 2): implement forward(), add to trainer.py

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : [B, 263] input feature vector.

        Returns:
            beta  : [B, 10]
            theta : [B, 72]
        """
        # TODO (Phase 2, Month 2)
        raise NotImplementedError("DualViewRegressor.forward() is a Phase 2 target.")

    @staticmethod
    def build_input(
        kp_front:    "torch.Tensor",   # [B, 33, 3]
        kp_side:     "torch.Tensor",   # [B, 33, 3]
        width_front: "torch.Tensor",   # [B, 32]
        width_side:  "torch.Tensor",   # [B, 32]
        height_norm: "torch.Tensor",   # [B, 1]
    ) -> "torch.Tensor":
        """Concatenate all inputs into a 263-dim feature vector."""
        return torch.cat([
            kp_front.reshape(-1, 99),
            kp_side.reshape(-1, 99),
            width_front,
            width_side,
            height_norm,
        ], dim=-1)


class DualViewRegressorEstimator:
    """
    Phase 2 estimator wrapper — same interface as HeuristicEstimator.
    Swapped in via FitEnginePipeline.swap_estimator().

    # TODO (Phase 2, Month 3): implement after training gates pass.
    """

    def __init__(self, checkpoint_path: str) -> None:
        raise NotImplementedError("DualViewRegressorEstimator is a Phase 2 target.")

    def predict(self, kp_front, kp_side, height_cm: float) -> dict:
        raise NotImplementedError

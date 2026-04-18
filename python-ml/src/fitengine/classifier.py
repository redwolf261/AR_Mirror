"""
Three independent classifier heads — Phase 2.

Each head maps β[10] → size class logits.
Independently detachable for brand-specific calibration.

CollarClassifier:  Linear(10, 64) + ReLU + Dropout(0.1) + Linear(64, 8)
  Classes: 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5

JacketClassifier:  Linear(10, 64) + ReLU + Dropout(0.1) + Linear(64, 8)
  Classes: 36, 38, 40, 42, 44, 46, 48, 50

TrouserClassifier: Linear(10, 64) + ReLU + Dropout(0.1) + Linear(64, 10)
  Classes: 28, 30, 32, 34, 36, 38, 40, 42, 44, 46

Brand calibration: sklearn LogisticRegression fitted on top of the β head
(see size_chart.SizeChart.calibrate()).  No GPU, no retraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _head(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


class CollarClassifier(nn.Module):
    """Collar size classifier: β[10] → logits[8]."""
    CLASSES = [14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]

    def __init__(self) -> None:
        super().__init__()
        self.net = _head(10, len(self.CLASSES))

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        return self.net(beta)


class JacketClassifier(nn.Module):
    """Jacket size classifier: β[10] → logits[8]."""
    CLASSES = [36, 38, 40, 42, 44, 46, 48, 50]

    def __init__(self) -> None:
        super().__init__()
        self.net = _head(10, len(self.CLASSES))

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        return self.net(beta)


class TrouserClassifier(nn.Module):
    """Trouser waist classifier: β[10] → logits[10]."""
    CLASSES = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46]

    def __init__(self) -> None:
        super().__init__()
        self.net = _head(10, len(self.CLASSES))

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        return self.net(beta)


class FitEngineClassifierBundle(nn.Module):
    """
    Three heads bundled for joint training.

    # TODO (Phase 2, Month 2): used by trainer.py after Gate 3.
    """

    def __init__(self) -> None:
        super().__init__()
        self.collar  = CollarClassifier()
        self.jacket  = JacketClassifier()
        self.trouser = TrouserClassifier()

    def forward(self, beta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "collar":  self.collar(beta),
            "jacket":  self.jacket(beta),
            "trouser": self.trouser(beta),
        }

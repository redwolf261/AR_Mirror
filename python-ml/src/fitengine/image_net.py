"""
image_net.py — DualStreamCNN: two-stream MobileNetV3-Large over stick figures.

Architecture
------------
  front_img [1,128,128] ──┐
                           ├──▶ shared MobileNetV3-Large backbone ──▶ 960-d each
  side_img  [1,128,128] ──┘
                           ▼
             concat [1920] + height_norm [1] → [1921]
                           ▼
             Dense(1921→1024) → BN → ReLU → Dropout(0.3)
                           ▼
             3 classification heads:
               head_collar  : Linear(1024, 8)
               head_jacket  : Linear(1024, 8)
               head_trouser : Linear(1024, 10)

Design notes:
  • Weights are SHARED between the two streams (not doubled) — forces
    learning rotation-invariant body-proportion patterns.
  • No ImageNet pre-training (stick figures ≠ natural images).
  • Single-channel input → first conv layer is 1-ch, not 3-ch.
  • Total trainable params: ~5.5M, VRAM@batch=64 ≈ 2.2 GB (safe on RTX 2050 4GB).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

from fitengine.heuristic.ratio_regressor import (
    COLLAR_CLASSES,
    JACKET_CLASSES,
    TROUSER_CLASSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone helper
# ---------------------------------------------------------------------------

def _make_mobilenet_backbone() -> nn.Sequential:
    """
    MobileNetV3-Large, adapted:
      • 1-channel input  (first Conv2d: 3 → 1)
      • no classification head  (features only, output = 960-d avg-pooled)
      • random init  (no ImageNet weights)
    """
    net = tvm.mobilenet_v3_large(weights=None)

    # Patch first convolution: 3-ch → 1-ch
    first_conv = net.features[0][0]          # Conv2d(3, 16, ...)
    new_conv   = nn.Conv2d(
        1, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
    net.features[0][0] = new_conv

    # Return only the feature extractor + adaptive pool (drop classifier)
    return nn.Sequential(net.features, net.avgpool)   # output: [B, 960, 1, 1]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualStreamCNN(nn.Module):
    """
    Two-stream stick-figure CNN with shared backbone.

    Args:
        dropout: dropout probability before each head.
    """

    FEATURE_DIM = 960   # MobileNetV3-Large avgpool output channels

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = _make_mobilenet_backbone()   # shared

        # Fusion MLP
        fusion_in = self.FEATURE_DIM * 2 + 1          # front + side + height_norm
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_collar  = nn.Linear(1024, len(COLLAR_CLASSES))
        self.head_jacket  = nn.Linear(1024, len(JACKET_CLASSES))
        self.head_trouser = nn.Linear(1024, len(TROUSER_CLASSES))

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B,1,H,W] → [B,960]"""
        return self.backbone(img).flatten(1)

    def forward(
        self,
        front: torch.Tensor,         # [B, 1, H, W]
        side:  torch.Tensor,         # [B, 1, H, W]
        height_norm: torch.Tensor,   # [B, 1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            collar_logits  [B, 8]
            jacket_logits  [B, 8]
            trouser_logits [B, 10]
        """
        feat_f = self._encode(front)
        feat_s = self._encode(side)
        x      = torch.cat([feat_f, feat_s, height_norm], dim=1)  # [B, 1921]
        h      = self.fusion(x)
        return self.head_collar(h), self.head_jacket(h), self.head_trouser(h)


# ---------------------------------------------------------------------------
# Estimator — drop-in wrapper for pipeline.swap_estimator()
# ---------------------------------------------------------------------------

class DualStreamEstimator:
    """
    Wraps DualStreamCNN for production inference.
    Implements the same .predict() interface as HeuristicEstimator.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        self._model = DualStreamCNN()
        self._model.load_state_dict(ckpt["model"])
        self._model.to(self._device)
        self._model.eval()
        self._size      = ckpt.get("img_size", 128)
        self._val_acc   = ckpt.get("val_adj_acc", {})
        logger.info(
            "DualStreamEstimator loaded from %s (val adj_acc: %s)",
            checkpoint_path, self._val_acc,
        )

    def predict(
        self,
        kp_front:  np.ndarray,
        kp_side:   np.ndarray,
        height_cm: float,
    ) -> dict:
        from fitengine.render import render_body_image
        from fitengine.measurements import BodyProxyMeasurements
        from fitengine.heuristic.estimator import _add_jacket_length

        m = BodyProxyMeasurements.from_keypoints(kp_front, kp_side)
        if not m.valid:
            logger.warning("DualStreamEstimator: invalid keypoints — default medium sizes")
            return {
                "collar":           "15.5",
                "jacket":           "40R",
                "trouser_waist":    "32",
                "confidence_level": "Low",
            }

        img_f = render_body_image(kp_front, size=self._size)  # [1,H,W] float32
        img_s = render_body_image(kp_side,  size=self._size)

        front  = torch.tensor(img_f).unsqueeze(0).to(self._device)   # [1,1,H,W]
        side   = torch.tensor(img_s).unsqueeze(0).to(self._device)
        hn     = torch.tensor(
            [[(height_cm - 175.0) / 12.0]], dtype=torch.float32
        ).to(self._device)                                             # [1,1]

        with torch.no_grad():
            c_l, j_l, t_l = self._model(front, side, hn)

        c_p = torch.softmax(c_l[0], 0).cpu().numpy()
        j_p = torch.softmax(j_l[0], 0).cpu().numpy()
        t_p = torch.softmax(t_l[0], 0).cpu().numpy()

        collar_str  = str(round(COLLAR_CLASSES [int(c_p.argmax())], 1))
        jacket_raw  = str(JACKET_CLASSES [int(j_p.argmax())])
        trouser_str = str(TROUSER_CLASSES[int(t_p.argmax())])

        def _conf(probs: np.ndarray) -> str:
            s = np.sort(probs)
            m = float(s[-1] - s[-2])
            return "High" if m >= 0.40 else "Medium" if m >= 0.18 else "Low"

        rank = {"High": 2, "Medium": 1, "Low": 0}
        overall = min([_conf(c_p), _conf(j_p), _conf(t_p)], key=lambda x: rank[x])

        return {
            "collar":           collar_str,
            "jacket":           _add_jacket_length(jacket_raw, height_cm),
            "trouser_waist":    trouser_str,
            "confidence_level": overall,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_CKPT = Path(__file__).parents[3] / "models" / "dual_stream_cnn.pt"


def load_dual_stream(
    checkpoint_path: Optional[str | Path] = None,
) -> Optional[DualStreamEstimator]:
    """Load DualStreamEstimator or return None if checkpoint not found."""
    path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
    if not path.exists():
        return None
    try:
        return DualStreamEstimator(path)
    except Exception as exc:
        logger.warning("Failed to load DualStreamCNN from %s: %s", path, exc)
        return None

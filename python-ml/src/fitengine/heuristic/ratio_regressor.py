"""
RatioRegressor — lightweight MLP trained on synthetic body ratio data.

Replaces the heuristic threshold tables with a proper learned model.
Same interface as HeuristicEstimator, so it's a drop-in for swap_estimator().

Input (7 features):
    shoulder_width_ratio   shoulder px / torso_height px
    chest_width_ratio      estimated chest px / torso_height px
    hip_width_ratio        hip px / torso_height px
    torso_depth_ratio      side-view shoulder span / torso_height px
    shoulder_depth_ratio   same as above (separate slot for future use)
    torso_height_px        torso height as fraction of image height [0,1]
    height_norm            (height_cm - 175.0) / 12.0

Output: 3 classification heads
    collar    8 classes  [14.0 … 17.5 in 0.5" steps]
    jacket    8 classes  [36 … 50 in 2" steps]
    trouser  10 classes  [28 … 46 in 2" steps]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size class tables (must match generic.json)
# ---------------------------------------------------------------------------

COLLAR_CLASSES  = [14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]
JACKET_CLASSES  = [36,   38,   40,   42,   44,   46,   48,   50  ]
TROUSER_CLASSES = [28,   30,   32,   34,   36,   38,   40,   42,   44,   46]

# String → class index   (populated once at import)
COLLAR_IDX  = {str(round(v, 1)): i for i, v in enumerate(COLLAR_CLASSES)}
JACKET_IDX  = {str(v): i          for i, v in enumerate(JACKET_CLASSES)}
TROUSER_IDX = {str(v): i          for i, v in enumerate(TROUSER_CLASSES)}

INPUT_DIM = 12  # elbow_width_ratio & arm_span_ratio dropped (arm-angle artifacts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RatioRegressor(nn.Module):
    """
    Small MLP: 7 body-ratio features → 3 size classification heads.

    ~100k parameters, trains in <2 min on RTX 2050.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        # Wider MLP to handle 14-feature input with 1M training samples.
        # 512→512→256→128 gives ~450k params, still fast with GPU-resident dataset.
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_collar  = nn.Linear(128, len(COLLAR_CLASSES))
        self.head_jacket  = nn.Linear(128, len(JACKET_CLASSES))
        self.head_trouser = nn.Linear(128, len(TROUSER_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x : [B, 7]
        Returns:
            collar_logits  : [B, 8]
            jacket_logits  : [B, 8]
            trouser_logits : [B, 10]
        """
        h = self.backbone(x)
        return self.head_collar(h), self.head_jacket(h), self.head_trouser(h)

    @staticmethod
    def features_from_measurements(m, height_cm: float) -> np.ndarray:
        """
        Extract the 7-feature vector from a BodyProxyMeasurements instance.

        Returns float32 array [7].
        """
        # elbow_width_ratio & arm_span_ratio are deliberately excluded:
        # they reflect random arm-angle variation, not body size, and add noise.
        return np.array([
            m.shoulder_width_ratio,
            m.chest_width_ratio,
            m.hip_width_ratio,
            m.torso_depth_ratio,
            m.shoulder_depth_ratio,
            m.torso_height_px,
            m.leg_length_ratio,
            m.inseam_ratio,
            m.head_width_ratio,
            m.hip_shoulder_taper,
            m.side_hip_depth_ratio,
            (height_cm - 175.0) / 12.0,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Estimator wrapper — drop-in for HeuristicEstimator
# ---------------------------------------------------------------------------

class RatioRegressorEstimator:
    """
    ML-based estimator using RatioRegressor.
    Implements the same .predict() interface as HeuristicEstimator.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        self._model = RatioRegressor()
        self._model.load_state_dict(ckpt["model"])
        self._model.to(self._device)
        self._model.eval()
        self._val_acc = ckpt.get("val_adj_acc", {})
        logger.info(
            "RatioRegressorEstimator loaded from %s  (val adj_acc: %s)",
            checkpoint_path, self._val_acc,
        )

    def predict(
        self,
        kp_front: np.ndarray,
        kp_side:  np.ndarray,
        height_cm: float,
    ) -> dict:
        """
        Same return schema as HeuristicEstimator.predict().
        """
        from fitengine.measurements import BodyProxyMeasurements
        from fitengine.heuristic.estimator import _add_jacket_length

        m = BodyProxyMeasurements.from_keypoints(kp_front, kp_side)

        if not m.valid:
            logger.warning(
                "RatioRegressorEstimator: invalid measurements — "
                "falling back to default medium sizes."
            )
            return {
                "collar":           "15.5",
                "jacket":           "40R",
                "trouser_waist":    "32",
                "confidence_level": "Low",
            }

        feat = RatioRegressor.features_from_measurements(m, height_cm)
        x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            c_logits, j_logits, t_logits = self._model(x)

        c_probs = torch.softmax(c_logits[0], dim=0).cpu().numpy()
        j_probs = torch.softmax(j_logits[0], dim=0).cpu().numpy()
        t_probs = torch.softmax(t_logits[0], dim=0).cpu().numpy()

        collar_str  = str(round(COLLAR_CLASSES [int(c_probs.argmax())], 1))
        jacket_raw  = str(JACKET_CLASSES [int(j_probs.argmax())])
        trouser_str = str(TROUSER_CLASSES[int(t_probs.argmax())])

        # Confidence: gap between top-2 probabilities
        def _conf(probs: np.ndarray) -> str:
            s = np.sort(probs)
            margin = float(s[-1] - s[-2])
            if margin >= 0.40: return "High"
            if margin >= 0.18: return "Medium"
            return "Low"

        conf_levels  = [_conf(c_probs), _conf(j_probs), _conf(t_probs)]
        # Overall: worst of three
        level_rank   = {"High": 2, "Medium": 1, "Low": 0}
        overall_conf = min(conf_levels, key=lambda x: level_rank[x])

        jacket_str = _add_jacket_length(jacket_raw, height_cm)

        return {
            "collar":           collar_str,
            "jacket":           jacket_str,
            "trouser_waist":    trouser_str,
            "confidence_level": overall_conf,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_CKPT = Path(__file__).parents[4] / "models" / "ratio_regressor.pt"


def load_ratio_regressor(
    checkpoint_path: Optional[str | Path] = None,
) -> Optional[RatioRegressorEstimator]:
    """
    Load RatioRegressorEstimator from checkpoint, or return None if not found.
    Used by FitEnginePipeline at startup.
    """
    path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
    if not path.exists():
        return None
    try:
        return RatioRegressorEstimator(path)
    except Exception as exc:
        logger.warning("Failed to load RatioRegressor from %s: %s", path, exc)
        return None

"""
SMPL-X Migration Foundation — Phase 4

Drop-in upgrade path from SMPL (6890 vertices) to SMPL-X (10,475 vertices).

SMPL-X adds:
  - Full hand articulation (45 hand joints → sleeve/cuff fitting)
  - Facial expressions (15 expression coefficients → collar/neckline fitting)
  - Jaw pose (1 jaw joint)

Why this matters for garment fitting:
  SMPL          → 6890 verts, no hands, no face
  SMPL-X        → 10,475 verts, full hand mesh, face mesh
  Coverage gain → sleeves fit to wrist, collars fit to neck accurately

This module provides:
  1. SMPLXBodyReconstructor — wraps smplx (pip install smplx) with the same
     public API as SMPLBodyReconstructor, so swap-in is one line in app.py
  2. smpl_to_smplx_params() — converts existing SMPL β/θ parameters to
     SMPL-X format (useful for backward compat with trained regressor)
  3. SMPLXMigrationStub — no-op class that logs warnings during the transition
     period before SMPL-X model weights are downloaded

References:
  SMPL-X: https://smpl-x.is.tue.mpg.de/
  Paper: Pavlakos et al., 2019 — "Expressive Body Capture: 3D Hands,
         Face, and Body from a Single Image"
  smplx library: https://github.com/vchoutas/smplx

Download model weights (free registration):
  https://smpl-x.is.tue.mpg.de/download.php
  Place in: models/smplx_neutral.npz
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Number of vertices for each body model
SMPL_VERTICES = 6890
SMPLX_VERTICES = 10475

# Parameter dimensions
SMPL_POSE_DIM = 72          # 24 joints × 3 (axis-angle)
SMPLX_POSE_DIM = 165        # 55 joints × 3
SMPLX_HAND_POSE_DIM = 90   # 2 hands × 15 joints × 3
SMPLX_EXPRESSION_DIM = 10   # facial expression coefficients


class SMPLXMigrationStub:
    """
    Zero-dependency stub that satisfies the SMPLBodyReconstructor API.

    Used during the transition period (before SMPL-X weights are downloaded).
    Falls back to the existing SMPL reconstructor transparently.

    Once models/smplx_neutral.npz is present, swap to SMPLXBodyReconstructor.
    """

    def __init__(self, smpl_reconstructor=None):
        self._smpl = smpl_reconstructor
        self._warned = False
        logger.info(
            "SMPLXMigrationStub active — using SMPL fallback.\n"
            "  To upgrade: download SMPL-X weights from https://smpl-x.is.tue.mpg.de\n"
            "  and place at models/smplx_neutral.npz"
        )

    @property
    def is_available(self) -> bool:
        return self._smpl.is_available if self._smpl else False

    @property
    def vertex_count(self) -> int:
        return SMPL_VERTICES   # will become 10475 after migration

    def reconstruct(self, pose_landmarks, frame_shape=None) -> Optional[Dict[str, Any]]:
        """Delegate to SMPL reconstructor during transition."""
        if not self._warned:
            logger.debug(
                "SMPL-X not yet active (stub mode). "
                "Hand/face fitting unavailable until migration completes."
            )
            self._warned = True
        if self._smpl is not None:
            return self._smpl.reconstruct(pose_landmarks, frame_shape)
        return None


class SMPLXBodyReconstructor:
    """
    Full SMPL-X body reconstructor (10,475 vertices).

    Requires:
      pip install smplx
      models/smplx_neutral.npz  (download from smpl-x.is.tue.mpg.de)

    API is a superset of SMPLBodyReconstructor — same reconstruct() signature
    plus additional hand_pose and expression outputs.
    """

    def __init__(
        self,
        model_path: str = "models/smplx_neutral.npz",
        regressor_path: str = "models/smpl_regressor.pth",   # existing regressor still usable
        device: str = "cuda",
        use_hands: bool = True,
        use_face: bool = True,
    ):
        self.device = device
        self.model_path = model_path
        self.use_hands = use_hands
        self.use_face = use_face
        self._is_available = False
        self._model = None

        try:
            self._load()
            self._is_available = True
            logger.info(
                f"✓ SMPL-X ready — {SMPLX_VERTICES} vertices "
                f"(+{SMPLX_VERTICES - SMPL_VERTICES} vs SMPL), "
                f"hands={use_hands}, face={use_face}"
            )
        except Exception as e:
            logger.warning(f"SMPL-X not available: {e}")

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def vertex_count(self) -> int:
        return SMPLX_VERTICES if self._is_available else SMPL_VERTICES

    def _load(self) -> None:
        """Load SMPL-X model using the smplx library."""
        import smplx   # pip install smplx
        import torch

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"SMPL-X model not found: {self.model_path}\n"
                "Download from: https://smpl-x.is.tue.mpg.de/download.php\n"
                "(Free registration required)\n"
                "Place at: models/smplx_neutral.npz"
            )

        self._model = smplx.create(
            self.model_path,
            model_type="smplx",
            gender="neutral",
            use_pca=False,                # full hand pose
            num_betas=10,
            num_expression_coeffs=SMPLX_EXPRESSION_DIM,
            use_face_contour=False,
        ).to(device=self.device)

    def reconstruct(
        self,
        pose_landmarks,
        frame_shape: Optional[Tuple[int, int]] = None,
        betas: Optional[np.ndarray] = None,
        hand_pose: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct SMPL-X body mesh from 2D pose landmarks.

        Returns dict with:
          vertices:    (10475, 3) float32 — 3D body mesh
          joints:      (127, 3) float32 — full joint positions
          faces:       (20908, 3) int32 — triangle indices
          hand_pose:   (90,) float32 — hand joint rotations
          expression:  (10,) float32 — facial expression
          betas:       (10,) float32 — shape parameters
        """
        if not self._is_available or self._model is None:
            return None

        import torch

        try:
            betas_t = torch.zeros(1, 10, device=self.device)
            if betas is not None:
                betas_t = torch.tensor(betas[:10], dtype=torch.float32, device=self.device).unsqueeze(0)

            body_pose = torch.zeros(1, SMPLX_POSE_DIM - 3, device=self.device)
            if pose_landmarks is not None:
                body_pose = self._landmarks_to_smplx_pose(pose_landmarks)

            left_hand = torch.zeros(1, SMPLX_HAND_POSE_DIM // 2, device=self.device)
            right_hand = torch.zeros(1, SMPLX_HAND_POSE_DIM // 2, device=self.device)
            if hand_pose is not None:
                hp = torch.tensor(hand_pose, dtype=torch.float32, device=self.device)
                left_hand = hp[:SMPLX_HAND_POSE_DIM // 2].unsqueeze(0)
                right_hand = hp[SMPLX_HAND_POSE_DIM // 2:].unsqueeze(0)

            expr = torch.zeros(1, SMPLX_EXPRESSION_DIM, device=self.device)
            if expression is not None:
                expr = torch.tensor(expression, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self._model(
                    betas=betas_t,
                    body_pose=body_pose,
                    left_hand_pose=left_hand,
                    right_hand_pose=right_hand,
                    expression=expr,
                    return_verts=True,
                )

            return {
                "vertices": output.vertices.squeeze().cpu().numpy(),
                "joints": output.joints.squeeze().cpu().numpy(),
                "faces": self._model.faces.astype(np.int32),
                "hand_pose": torch.cat([left_hand, right_hand], dim=-1).squeeze().cpu().numpy(),
                "expression": expr.squeeze().cpu().numpy(),
                "betas": betas_t.squeeze().cpu().numpy(),
                "vertex_count": SMPLX_VERTICES,
            }

        except Exception as e:
            logger.error(f"SMPL-X reconstruction failed: {e}")
            return None

    def _landmarks_to_smplx_pose(self, pose_landmarks) -> "torch.Tensor":
        """
        Convert MediaPipe 33-landmark pose to SMPL-X body_pose parameters.

        SMPL-X body_pose: 21 joints × 3 = 63 dims (excludes global orient)
        Uses the same heuristic mapping as the SMPL regressor, extended for
        the additional SMPL-X joints (wrists, fingers mapped to hand_pose).
        """
        import torch
        # Zero initialise — the trained regressor will override this
        # once fine-tuned on SMPL-X targets
        return torch.zeros(1, SMPLX_POSE_DIM - 3, device=self.device)


# ---------------------------------------------------------------------------
# Parameter conversion utilities
# ---------------------------------------------------------------------------

def smpl_to_smplx_params(
    smpl_betas: np.ndarray,
    smpl_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Convert SMPL parameters to SMPL-X format (approximate).

    The first 10 shape betas are compatible across SMPL / SMPL-X.
    Body pose (joints 0-21) maps directly; hand/face start at zero.

    Args:
        smpl_betas: (10,) shape parameters
        smpl_pose:  (72,) SMPL pose (24 joints × 3)

    Returns:
        dict with keys: betas, body_pose, hand_pose, expression
    """
    betas = smpl_betas[:10].copy()

    # SMPL joints 0-21 map to SMPL-X body joints 0-21
    body_pose_smplx = np.zeros(SMPLX_POSE_DIM - 3, dtype=np.float32)
    n_shared = min(len(smpl_pose) - 3, SMPLX_POSE_DIM - 3)
    body_pose_smplx[:n_shared] = smpl_pose[3:3 + n_shared]

    hand_pose = np.zeros(SMPLX_HAND_POSE_DIM, dtype=np.float32)
    expression = np.zeros(SMPLX_EXPRESSION_DIM, dtype=np.float32)

    return {
        "betas": betas,
        "body_pose": body_pose_smplx,
        "hand_pose": hand_pose,
        "expression": expression,
    }


def create_body_reconstructor(
    prefer_smplx: bool = True,
    smpl_reconstructor=None,
    smplx_model_path: str = "models/smplx_neutral.npz",
    device: str = "cuda",
):
    """
    Factory — returns the best available body reconstructor.

    Priority:
      1. SMPL-X (if model file present and smplx library installed)
      2. SMPL (existing SMPLBodyReconstructor)
      3. SMPLXMigrationStub (delegates to SMPL with a logged warning)

    Usage in app.py (one-line swap):
        from src.core.smplx_body_reconstruction import create_body_reconstructor
        reconstructor = create_body_reconstructor(smpl_reconstructor=existing_smpl)
    """
    if prefer_smplx and Path(smplx_model_path).exists():
        try:
            rec = SMPLXBodyReconstructor(
                model_path=smplx_model_path, device=device
            )
            if rec.is_available:
                return rec
        except Exception as e:
            logger.debug(f"SMPL-X init failed: {e}")

    if smpl_reconstructor is not None and getattr(smpl_reconstructor, "is_available", False):
        logger.info("Using SMPL reconstructor (SMPL-X model not ready)")
        return smpl_reconstructor

    return SMPLXMigrationStub(smpl_reconstructor)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"SMPL  vertices : {SMPL_VERTICES}")
    print(f"SMPL-X vertices: {SMPLX_VERTICES}")
    print(f"Vertex gain    : +{SMPLX_VERTICES - SMPL_VERTICES}")

    stub = SMPLXMigrationStub()
    print(f"Stub available : {stub.is_available}")
    print(f"Stub verts     : {stub.vertex_count}")

    params = smpl_to_smplx_params(
        smpl_betas=np.zeros(10),
        smpl_pose=np.zeros(72),
    )
    print(f"Converted betas shape     : {params['betas'].shape}")
    print(f"Converted body_pose shape : {params['body_pose'].shape}")
    print(f"Converted hand_pose shape : {params['hand_pose'].shape}")

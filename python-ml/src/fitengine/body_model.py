"""
STAR body model wrapper (MIT licence).

STARBodyModel auto-downloads STAR weights to ~/.fitengine/star/ on first use and
exposes a forward() interface compatible with the SMPL θ/β convention:

    vertices [B, 6890, 3]  (metres)
    joints   [B, 24, 3]    (metres)

STAR reference: https://github.com/ahmedosman/STAR
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.path.expanduser("~/.fitengine/star"))
_MODEL_URL = (
    "https://github.com/ahmedosman/STAR/releases/download/v1.0/STAR_NEUTRAL.npz"
)
_MODEL_FILENAME = "STAR_NEUTRAL.npz"

# STAR has 6890 vertices, 24 joints, β dim=10, θ dim=72
_N_VERTS = 6890
_N_JOINTS = 24
_BETA_DIM = 10
_THETA_DIM = 72


def _download_star(dest: Path) -> None:
    """Download STAR neutral model weights to dest."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading STAR weights → %s", dest)
    urllib.request.urlretrieve(_MODEL_URL, str(dest))
    logger.info("STAR download complete.")


class STARBodyModel(nn.Module):
    """
    Minimal STAR body model.

    Parameters are loaded from the official STAR .npz (MIT licence).
    This is not the full differentiable STAR implementation — it is a
    forward-pass-only wrapper sufficient for:
      - synthetic data generation (Phase 2)
      - scale-invariant proxy measurements (Phase 1)
      - reprojection loss during regression training (Phase 2)

    For full differentiable STAR (e.g. pose optimisation), install the
    official `star` package: https://github.com/ahmedosman/STAR
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        super().__init__()

        if model_path is None:
            model_path = _CACHE_DIR / _MODEL_FILENAME

        model_path = Path(model_path)
        if not model_path.exists():
            _download_star(model_path)

        data = np.load(str(model_path), allow_pickle=True)
        self._load_params(data)
        logger.info("STAR body model loaded from %s", model_path)

    def _load_params(self, data: np.lib.npyio.NpzFile) -> None:
        """Register STAR model parameters as buffers."""

        def _buf(key: str, fallback_shape=None) -> torch.Tensor:
            if key in data:
                return torch.from_numpy(data[key].astype(np.float32))
            if fallback_shape is not None:
                logger.warning("STAR key '%s' not found — using zeros %s", key, fallback_shape)
                return torch.zeros(fallback_shape, dtype=torch.float32)
            raise KeyError(f"STAR model missing required key: '{key}'")

        # Shape blend shapes [6890*3, 10]
        self.register_buffer("shapedirs",   _buf("shapedirs",   (_N_VERTS * 3, _BETA_DIM)))
        # Pose blend shapes  [6890*3, 207] (STAR uses 207 = 23×9)
        self.register_buffer("posedirs",    _buf("posedirs",    (_N_VERTS * 3, 207)))
        # Joint regressor    [24, 6890]
        self.register_buffer("J_regressor", _buf("J_regressor", (_N_JOINTS, _N_VERTS)))
        # Template vertices  [6890, 3]
        self.register_buffer("v_template",  _buf("v_template",  (_N_VERTS, 3)))
        # Skinning weights   [6890, 24]
        self.register_buffer("weights",     _buf("weights",     (_N_VERTS, _N_JOINTS)))
        # Kinematic tree     [24, 2]
        if "kintree_table" in data:
            kt = torch.from_numpy(data["kintree_table"].astype(np.int64))
            self.register_buffer("kintree_table", kt)
        else:
            # default SMPL/STAR kinematic tree
            default_kt = torch.tensor([
                [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ], dtype=torch.int64)
            self.register_buffer("kintree_table", default_kt)

    @torch.no_grad()
    def forward(
        self,
        theta: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            theta : [B, 72]  — pose parameters (axis-angle, 24 joints × 3).
            beta  : [B, 10]  — shape parameters.

        Returns:
            vertices : [B, 6890, 3]  in metres.
            joints   : [B, 24, 3]   in metres.
        """
        B = beta.shape[0]
        device = beta.device

        # 1. Shape blend shapes
        v_shaped = self.v_template + (
            self.shapedirs @ beta.T
        ).T.reshape(B, _N_VERTS, 3)  # [B, 6890, 3]

        # 2. Pose blend shapes (rotation matrix representation)
        #    Convert axis-angle to rotation matrices
        theta_mat = _batch_rodrigues(theta.reshape(-1, 3)).reshape(B, _N_JOINTS, 3, 3)
        # Pose blend shape features: (R - I) for joints 1:24
        pose_feature = (theta_mat[:, 1:] - torch.eye(3, device=device)).reshape(B, -1)  # [B, 207]
        v_posed = v_shaped + (self.posedirs @ pose_feature.T).T.reshape(B, _N_VERTS, 3)

        # 3. Joint locations
        J = torch.einsum("jv,bvk->bjk", self.J_regressor, v_posed)  # [B, 24, 3]

        # 4. LBS
        vertices = _lbs(v_posed, theta_mat, J, self.weights, self.kintree_table)

        return vertices, J

    def get_height_cm(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Estimate standing height (cm) from beta.

        Forward pass with zero pose, then take max Y − min Y of vertices.
        Args:
            beta : [B, 10]
        Returns:
            height_cm : [B]
        """
        device = beta.device
        B = beta.shape[0]
        theta_zero = torch.zeros(B, _THETA_DIM, device=device)
        verts, _ = self.forward(theta_zero, beta)
        height_m = verts[:, :, 1].max(dim=1).values - verts[:, :, 1].min(dim=1).values
        return height_m * 100.0  # → cm


# ---------------------------------------------------------------------------
# LBS helpers
# ---------------------------------------------------------------------------

def _batch_rodrigues(rvecs: torch.Tensor) -> torch.Tensor:
    """
    Batch axis-angle → rotation matrices.
    rvecs : [N, 3]
    Returns: [N, 3, 3]
    """
    angle = rvecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
    axis  = rvecs / angle                                       # [N, 3]
    angle = angle.squeeze(-1)                                   # [N]

    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1.0 - c

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = torch.stack([
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ], dim=-1).reshape(-1, 3, 3)

    return R


def _lbs(
    v_posed: torch.Tensor,
    theta_mat: torch.Tensor,
    J: torch.Tensor,
    weights: torch.Tensor,
    kintree_table: torch.Tensor,
) -> torch.Tensor:
    """
    Linear Blend Skinning.

    Args:
        v_posed       : [B, V, 3]
        theta_mat     : [B, 24, 3, 3]
        J             : [B, 24, 3]
        weights       : [V, 24]
        kintree_table : [2, 24]

    Returns:
        vertices : [B, V, 3]
    """
    B, V = v_posed.shape[:2]
    device = v_posed.device

    parent = kintree_table[0]  # [24]

    # Build global transforms [B, 24, 4, 4]
    T = torch.zeros(B, 24, 4, 4, device=device)
    T[:, :, :3, :3] = theta_mat
    T[:, :, :3,  3] = J
    T[:, :,  3,  3] = 1.0

    # Subtract parent joint to get local
    for i in range(1, 24):
        p = parent[i].long()
        J_parent = J[:, p, :]  # [B, 3]
        T[:, i, :3, 3] -= T[:, i, :3, :3] @ J_parent.unsqueeze(-1).squeeze(-1)

    # Forward kinematics
    G = T.clone()
    for i in range(1, 24):
        p = parent[i].long()
        G[:, i] = G[:, p] @ T[:, i]

    # Remove rest-pose
    rest = torch.zeros(B, 24, 4, 1, device=device)
    rest[:, :, :3, 0] = J
    rest[:, :,  3, 0] = 1.0
    G_rest = G - (G @ rest).expand_as(G)
    # Correct: subtract the rest-pose offset properly
    zeros = torch.zeros(B, 24, 1, 4, device=device)
    Gp = G.clone()
    Jh = torch.cat([J, torch.ones(B, 24, 1, device=device)], dim=-1)  # [B,24,4]
    offset = torch.einsum("bjkl,bjl->bjk", G, Jh)[:, :, :, None]     # [B,24,4,1]
    # Standard LBS rest pose correction
    G_corrected = G.clone()
    for b in range(B):
        for j in range(24):
            G_corrected[b, j, :3, 3] -= (G_corrected[b, j, :3, :3] @ J[b, j])

    # Skin
    W = weights.unsqueeze(0).expand(B, -1, -1)         # [B, V, 24]
    T_skin = torch.einsum("bvj,bjkl->bvkl", W, G_corrected)  # [B, V, 4, 4]

    v_h = torch.cat([v_posed, torch.ones(B, V, 1, device=device)], dim=-1)  # [B,V,4]
    v_out = torch.einsum("bvkl,bvl->bvk", T_skin, v_h)[:, :, :3]           # [B,V,3]

    return v_out

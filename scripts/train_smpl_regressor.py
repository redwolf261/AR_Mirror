#!/usr/bin/env python3
"""
Train SMPL Regressor: 2D MediaPipe landmarks → SMPL parameters (β shape, θ pose)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW TO RUN  (paste into terminal, no timeout):

  .\\ar\\Scripts\\python.exe scripts\\train_smpl_regressor.py

Optional flags:
  --epochs 60            (default 60, ~20-25 min on RTX 2050)
  --batch-size 256       (default 256, reduce to 128 if OOM)
  --lr 1e-3              (default 1e-3)
  --hidden-dim 512       (default 512)
  --num-blocks 4         (default 4 iterative regression blocks)
  --num-iterations 3     (default 3 refinement passes per block)
  --resume               (resume from output/logs/smpl_regressor_latest.pth)
  --dry-run              (5 batches only, to verify setup)
  --n-samples 100000     (number of synthetic samples to generate)

Output:
  models/smpl_regressor.pth          ← final checkpoint (load in production)
  output/logs/smpl_training.log      ← full training log
  output/logs/smpl_regressor_latest.pth  ← latest checkpoint (for --resume)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data source: Commercial-safe synthetic generation
  - Unlimited SMPL parameter variations with perfect ground truth
  - MediaPipe-compatible keypoint format
  - Domain randomization for robust training
  - Zero research dataset dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── output dirs ──────────────────────────────────────────────────────────────
LOG_DIR   = ROOT / "output" / "logs"
MODEL_DIR = ROOT / "models"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "smpl_training.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("train_smpl")

# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Single residual block: Linear → BN → ReLU → Linear → BN + skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class IterativeSMPLRegressor(nn.Module):
    """
    Iterative regression network: 2D landmarks → SMPL (β, θ).

    Architecture mirrors IEF (Iterative Error Feedback) from HMR (Kanazawa et al.).
    Each iteration refines the current parameter estimate:
        x_t = x_{t-1} - f(concat(landmarks, x_{t-1}))

    Input:  (batch, input_dim)   — flattened 2D keypoints + optionally visibility
    Output: (batch, shape_dim)   — β shape params (10)
            (batch, pose_dim)    — θ pose params (72)

    Parameters:
        input_dim     : 33 * 3 = 99  (x, y, visibility per MediaPipe landmark)
        hidden_dim    : 512
        num_blocks    : 4 residual blocks per refinement step
        num_iterations: 3 refinement passes
        shape_dim     : 10  (SMPL β)
        pose_dim      : 72  (SMPL θ, 24 joints × 3 axis-angle)
    """

    def __init__(
        self,
        input_dim: int = 99,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        num_iterations: int = 3,
        shape_dim: int = 10,
        pose_dim: int = 72,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.shape_dim = shape_dim
        self.pose_dim  = pose_dim
        param_dim = shape_dim + pose_dim  # 82

        # Encoder: project landmarks + current params into hidden space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + param_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        # Output heads
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, shape_dim),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, pose_dim),
        )

        # Initial parameter estimate (learnable mean)
        self.register_parameter(
            "init_params",
            nn.Parameter(torch.zeros(1, param_dim))
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for the output heads (stable early training)
        for head in [self.shape_head, self.pose_head]:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            keypoints: (batch, input_dim)
        Returns:
            shape_params: (batch, 10)
            pose_params : (batch, 72)
        """
        B = keypoints.shape[0]
        params = self.init_params.expand(B, -1)  # (B, 82)

        for _ in range(self.num_iterations):
            x = torch.cat([keypoints, params], dim=1)   # (B, input_dim + 82)
            h = self.encoder(x)
            for block in self.blocks:
                h = block(h)
            delta_shape = self.shape_head(h)
            delta_pose  = self.pose_head(h)
            delta = torch.cat([delta_shape, delta_pose], dim=1)
            params = params + delta                       # residual update

        shape = params[:, :self.shape_dim]
        pose  = params[:, self.shape_dim:]
        return shape, pose


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 100_000, seed: int = 42) -> TensorDataset:
    """
    Generate synthetic (keypoints, β, θ) pairs.

    Since we don't have a labelled 2D→SMPL dataset, we generate plausible
    correspondences by:
    1. Sampling random SMPL parameters from their natural distributions
    2. Projecting a simplified kinematic chain to 2D keypoints
    3. Adding realistic noise (landmark jitter, partial occlusion)

    This gives the regressor enough signal to learn the structural relationship
    between pose/shape and 2D projections.
    """
    rng = np.random.default_rng(seed)
    log.info(f"Generating {n_samples:,} synthetic samples (seed={seed})…")

    # ── Sample SMPL parameters ───────────────────────────────────────────────
    # β: shape — ~N(0, 1) per principal component (typical range ±3)
    betas = rng.normal(0, 1, (n_samples, 10)).astype(np.float32)

    # theta: pose - most joints near zero, some variance for natural poses
    # 24 joints x 3 axis-angle = 72 dims.  One sigma per joint (broadcast x3).
    pose_sigma_per_joint = np.array([
        0.3,  # 0  global orient
        0.5,  # 1  L hip
        0.5,  # 2  R hip
        0.1,  # 3  spine1
        0.7,  # 4  L knee
        0.7,  # 5  R knee
        0.1,  # 6  spine2
        0.3,  # 7  L ankle
        0.3,  # 8  R ankle
        0.1,  # 9  spine3
        0.1,  # 10 L foot
        0.1,  # 11 R foot
        0.1,  # 12 neck
        0.3,  # 13 L collar
        0.3,  # 14 R collar
        0.1,  # 15 head
        0.6,  # 16 L shoulder
        0.6,  # 17 R shoulder
        0.5,  # 18 L elbow
        0.5,  # 19 R elbow
        0.3,  # 20 L wrist
        0.3,  # 21 R wrist
        0.1,  # 22 L hand
        0.1,  # 23 R hand
    ], dtype=np.float32)  # exactly 24 joints
    pose_sigma_full = np.repeat(pose_sigma_per_joint, 3)  # (72,)
    thetas = rng.normal(0, pose_sigma_full, (n_samples, 72)).astype(np.float32)

    # ── Project to 2D keypoints (simplified kinematic skeleton) ─────────────
    # 33 MediaPipe landmarks, each (x, y, visibility)
    # We use a rough centred skeleton and apply pose offsets
    _MEDIAPIPE_SKELETON_NORM = np.array([
        # (x, y) in normalised image coords [centre=0.5]
        [0.50, 0.08],  # 0  nose
        [0.47, 0.07],  # 1  L eye inner
        [0.46, 0.07],  # 2  L eye
        [0.45, 0.07],  # 3  L eye outer
        [0.53, 0.07],  # 4  R eye inner
        [0.54, 0.07],  # 5  R eye
        [0.55, 0.07],  # 6  R eye outer
        [0.44, 0.09],  # 7  L ear
        [0.56, 0.09],  # 8  R ear
        [0.48, 0.11],  # 9  L mouth
        [0.52, 0.11],  # 10 R mouth
        [0.39, 0.22],  # 11 L shoulder
        [0.61, 0.22],  # 12 R shoulder
        [0.33, 0.38],  # 13 L elbow
        [0.67, 0.38],  # 14 R elbow
        [0.28, 0.52],  # 15 L wrist
        [0.72, 0.52],  # 16 R wrist
        [0.25, 0.55],  # 17 L pinky
        [0.75, 0.55],  # 18 R pinky
        [0.26, 0.55],  # 19 L index
        [0.74, 0.55],  # 20 R index
        [0.27, 0.55],  # 21 L thumb
        [0.73, 0.55],  # 22 R thumb
        [0.42, 0.55],  # 23 L hip
        [0.58, 0.55],  # 24 R hip
        [0.41, 0.72],  # 25 L knee
        [0.59, 0.72],  # 26 R knee
        [0.40, 0.88],  # 27 L ankle
        [0.60, 0.88],  # 28 R ankle
        [0.38, 0.92],  # 29 L heel
        [0.62, 0.92],  # 30 R heel
        [0.39, 0.94],  # 31 L foot index
        [0.61, 0.94],  # 32 R foot index
    ], dtype=np.float32)  # (33, 2)

    # Apply pose-driven deformation (simplified: linear blend of joint angles)
    # Shoulder width scales with beta[0]; overall height scales with beta[1]
    width_scale  = 1.0 + 0.10 * betas[:, 0:1]   # (N, 1)
    height_scale = 1.0 + 0.05 * betas[:, 1:2]   # (N, 1)

    # Broadcast skeleton to all samples
    kp_xy = np.tile(_MEDIAPIPE_SKELETON_NORM, (n_samples, 1, 1))  # (N, 33, 2)

    # Scale around centre
    kp_xy[:, :, 0] = 0.5 + (kp_xy[:, :, 0] - 0.5) * width_scale
    kp_xy[:, :, 1] = 0.5 + (kp_xy[:, :, 1] - 0.5) * height_scale

    # Add pose offsets (dominant joints: shoulders, elbows, hips, knees)
    kp_xy[:, 11, 0] += thetas[:, 48] * 0.04   # L shoulder x
    kp_xy[:, 12, 0] += thetas[:, 51] * 0.04   # R shoulder x
    kp_xy[:, 13, 0] += thetas[:, 54] * 0.06   # L elbow x
    kp_xy[:, 14, 0] += thetas[:, 57] * 0.06   # R elbow x
    kp_xy[:, 25, 1] += thetas[:, 12] * 0.04   # L knee y (bend)
    kp_xy[:, 26, 1] += thetas[:, 15] * 0.04   # R knee y

    # Add realistic noise
    jitter = rng.normal(0, 0.008, kp_xy.shape).astype(np.float32)
    kp_xy = np.clip(kp_xy + jitter, 0.0, 1.0)

    # Visibility: mostly visible, with random occlusion
    visibility = rng.uniform(0.6, 1.0, (n_samples, 33, 1)).astype(np.float32)
    # Randomly occlude ~15% of landmarks
    occlude_mask = rng.random((n_samples, 33, 1)) < 0.15
    visibility[occlude_mask] = rng.uniform(0.0, 0.3, occlude_mask.sum())

    # Stack into (N, 33, 3) then flatten to (N, 99)
    kp = np.concatenate([kp_xy, visibility], axis=-1).reshape(n_samples, 99)

    # Convert to tensors
    kp_t     = torch.tensor(kp,     dtype=torch.float32)
    betas_t  = torch.tensor(betas,  dtype=torch.float32)
    thetas_t = torch.tensor(thetas, dtype=torch.float32)

    log.info(f"  kp:     {kp_t.shape}  range [{kp_t.min():.3f}, {kp_t.max():.3f}]")
    log.info(f"  betas:  {betas_t.shape}  range [{betas_t.min():.3f}, {betas_t.max():.3f}]")
    log.info(f"  thetas: {thetas_t.shape}  range [{thetas_t.min():.3f}, {thetas_t.max():.3f}]")

    return TensorDataset(kp_t, betas_t, thetas_t)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation (commercial-safe approach)
# ─────────────────────────────────────────────────────────────────────────────


def _joints3d_to_pseudo_params(
    joints3d: np.ndarray,   # (N, J, 3)  world/camera-space 3-D joints
    rng: np.random.Generator,
) -> tuple:
    """
    Convert 3-D joint positions to pseudo SMPL (β, θ) parameters.

    β (shape): derived from limb-length ratios relative to a mean skeleton.
    θ (pose): derived from unit-vector directions of each bone.

    Returns:
        betas  (N, 10) float32
        thetas (N, 72) float32
    """
    N = joints3d.shape[0]

    # ── β: encode body proportions ──────────────────────────────────────────
    # Use 7 H3.6M-style limb lengths that are present in both H3WB(body) and H3.6M.
    # Indices here reference the joint arrays after the caller has extracted a
    # 17-joint body subset in H3.6M order (0=hip, 1=Rhip, … 16=Rwrist).

    def _safe_bone(a, b):
        """Length of bone a→b for each frame, shape (N,)."""
        if a < joints3d.shape[1] and b < joints3d.shape[1]:
            return np.linalg.norm(joints3d[:, a] - joints3d[:, b], axis=-1)
        return np.ones(N, dtype=np.float32)

    upper_leg_L = _safe_bone(4, 5)   # L hip → L knee
    upper_leg_R = _safe_bone(1, 2)   # R hip → R knee
    lower_leg_L = _safe_bone(5, 6)   # L knee → L ankle
    lower_leg_R = _safe_bone(2, 3)   # R knee → R ankle
    upper_arm_L = _safe_bone(11, 12) # L shoulder → L elbow
    upper_arm_R = _safe_bone(14, 15) # R shoulder → R elbow
    torso       = _safe_bone(0, 8)   # hip → thorax  (if J>=9)

    # Normalise each by its own mean (removes absolute scale)
    def _norm(x):
        mu = np.mean(x) + 1e-8
        return (x - mu) / (mu * 0.3 + 1e-8)   # ~N(0,1) range

    # β[:, 0] ≈ overall height, β[:, 1] ≈ leg length, β[:, 2] ≈ arm length,
    # β[:, 3] ≈ symmetry, rest are small noise
    leg_mean  = (upper_leg_L + upper_leg_R + lower_leg_L + lower_leg_R) / 4
    arm_mean  = (upper_arm_L + upper_arm_R) / 2
    leg_sym   = _norm(upper_leg_L - upper_leg_R)
    arm_sym   = _norm(upper_arm_L - upper_arm_R)

    betas = np.zeros((N, 10), dtype=np.float32)
    betas[:, 0] = _norm(leg_mean + arm_mean)      # ~height
    betas[:, 1] = _norm(leg_mean)                  # leg length
    betas[:, 2] = _norm(arm_mean)                  # arm length
    betas[:, 3] = leg_sym                          # L/R leg symmetry
    betas[:, 4] = arm_sym                          # L/R arm symmetry
    betas[:, 5:] = rng.normal(0, 0.1, (N, 5)).astype(np.float32)
    betas = np.clip(betas, -3.0, 3.0)

    # ── θ: encode joint rotations from bone directions ───────────────────────
    # SMPL has 24 joints × 3 axis-angle = 72 dims.
    # We compute 6 key rotations from bone direction vectors; pad rest with noise.

    def _bone_dir(a, b):
        """Unit vector from joint a → joint b, shape (N, 3)."""
        if a < joints3d.shape[1] and b < joints3d.shape[1]:
            v = joints3d[:, b] - joints3d[:, a]
            norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
            return v / norm
        return np.tile([0., -1., 0.], (N, 1)).astype(np.float32)

    # T-pose canonical directions  (these define θ=0)
    _T_LEFT_ARM  = np.array([-1., 0., 0.], dtype=np.float32)
    _T_RIGHT_ARM = np.array([ 1., 0., 0.], dtype=np.float32)
    _T_LEG_DOWN  = np.array([ 0.,-1., 0.], dtype=np.float32)

    def _axis_angle(canonical, actual):
        """Rotation from `canonical` to `actual` as axis-angle (N, 3)."""
        # axis = canonical × actual
        axis = np.cross(canonical, actual)   # (N, 3)
        sin_a = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8
        cos_a = np.clip(np.einsum('j,nj->n', canonical, actual), -1, 1)[:, None]
        angle = np.arctan2(sin_a, cos_a)     # (N, 1)
        ax_n  = axis / sin_a                 # (N, 3)
        return (ax_n * angle).astype(np.float32)  # (N, 3)

    thetas = rng.normal(0, 0.05, (N, 72)).astype(np.float32)  # base: small noise

    # Pelvis (joint 0) = global orient — vertical trunk
    if joints3d.shape[1] >= 9:
        trunk_dir = _bone_dir(0, 8)
        thetas[:, 0:3]  = _axis_angle(np.array([0., 1., 0.]), trunk_dir)

    # L hip (joint 1), R hip (joint 4)
    thetas[:, 3:6]  = _axis_angle(_T_LEG_DOWN, _bone_dir(4, 5))    # L hip
    thetas[:, 12:15] = _axis_angle(_T_LEG_DOWN, _bone_dir(1, 2))   # R hip
    # L knee (joint 5), R knee (joint 2)
    thetas[:, 15:18] = _axis_angle(_T_LEG_DOWN, _bone_dir(5, 6))   # L knee
    thetas[:, 6:9]   = _axis_angle(_T_LEG_DOWN, _bone_dir(2, 3))   # R knee
    # L shoulder (joint 16), R shoulder (joint 17)
    thetas[:, 48:51] = _axis_angle(_T_LEFT_ARM,  _bone_dir(11, 12)) # L shoulder
    thetas[:, 51:54] = _axis_angle(_T_RIGHT_ARM, _bone_dir(14, 15)) # R shoulder
    # L elbow (joint 18), R elbow (joint 19)
    thetas[:, 54:57] = _axis_angle(_T_LEFT_ARM,  _bone_dir(12, 13)) # L elbow
    thetas[:, 57:60] = _axis_angle(_T_RIGHT_ARM, _bone_dir(15, 16)) # R elbow

    return betas, thetas


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    path: Path,
    config: dict,
):
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler else None,
        "val_loss":     val_loss,
        "model_config": config,
    }, path)
    log.info(f"  Checkpoint saved → {path}  (val_loss={val_loss:.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
) -> float:
    model.train()
    loss_meter  = AverageMeter("loss")
    shape_meter = AverageMeter("shape_loss")
    pose_meter  = AverageMeter("pose_loss")

    log_every = max(1, len(loader) // 10)  # log ~10 times per epoch
    t_epoch   = time.perf_counter()

    for i, (kp, gt_shape, gt_pose) in enumerate(loader):
        kp        = kp.to(device,       non_blocking=True)
        gt_shape  = gt_shape.to(device, non_blocking=True)
        gt_pose   = gt_pose.to(device,  non_blocking=True)

        pred_shape, pred_pose = model(kp)

        # Shape loss: MSE on β
        loss_shape = nn.functional.mse_loss(pred_shape, gt_shape)

        # Pose loss: MSE on θ, scaled down (θ values are larger)
        loss_pose = nn.functional.mse_loss(pred_pose, gt_pose)

        # Total loss: weight shape more heavily (matters for garment size)
        loss = 2.0 * loss_shape + 1.0 * loss_pose

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping (prevents occasional spikes)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        B = kp.shape[0]
        loss_meter.update(loss.item(), B)
        shape_meter.update(loss_shape.item(), B)
        pose_meter.update(loss_pose.item(), B)

        if (i + 1) % log_every == 0 or (i + 1) == len(loader):
            elapsed = time.perf_counter() - t_epoch
            pct     = (i + 1) / len(loader) * 100
            log.info(
                f"  Epoch {epoch:3d}  [{pct:5.1f}%]  "
                f"loss={loss_meter.avg:.5f}  "
                f"shape={shape_meter.avg:.5f}  "
                f"pose={pose_meter.avg:.5f}  "
                f"elapsed={elapsed:.1f}s"
            )

        if args.dry_run and i >= 4:
            log.info("  [dry-run] stopping after 5 batches")
            break

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    shape_meter = AverageMeter("val_shape")
    pose_meter  = AverageMeter("val_pose")

    for kp, gt_shape, gt_pose in loader:
        kp        = kp.to(device)
        gt_shape  = gt_shape.to(device)
        gt_pose   = gt_pose.to(device)

        pred_shape, pred_pose = model(kp)
        shape_meter.update(nn.functional.mse_loss(pred_shape, gt_shape).item(), kp.shape[0])
        pose_meter.update(nn.functional.mse_loss(pred_pose, gt_pose).item(), kp.shape[0])

    val_loss = 2.0 * shape_meter.avg + 1.0 * pose_meter.avg
    return val_loss, shape_meter.avg, pose_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SMPL Regressor (2D keypoints → SMPL β, θ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--hidden-dim",     type=int,   default=512)
    p.add_argument("--num-blocks",     type=int,   default=4)
    p.add_argument("--num-iterations", type=int,   default=3)
    p.add_argument("--n-samples",      type=int,   default=100_000)
    p.add_argument("--val-split",      type=float, default=0.1)
    p.add_argument("--workers",        type=int,   default=0,
                   help="DataLoader workers (0 = main thread, safe on Windows)")
    p.add_argument("--resume",         action="store_true",
                   help="Resume from output/logs/smpl_regressor_latest.pth")
    p.add_argument("--dry-run",        action="store_true",
                   help="5 batches only — verifies setup without full training")
    p.add_argument("--cpu",            action="store_true",
                   help="Force CPU (useful for debugging)")
    p.add_argument("--synthetic",       action="store_true", default=True,
                   help="Generate synthetic training data (always enabled for commercial safety)")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available — training on CPU (will be slow)")

    # ── Banner ────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════")
    log.info("  SMPL Regressor Training")
    log.info(f"  device         : {device}")
    log.info(f"  epochs         : {args.epochs}")
    log.info(f"  batch_size     : {args.batch_size}")
    log.info(f"  lr             : {args.lr}")
    log.info(f"  hidden_dim     : {args.hidden_dim}")
    log.info(f"  num_blocks     : {args.num_blocks}")
    log.info(f"  num_iterations : {args.num_iterations}")
    log.info(f"  n_samples      : {args.n_samples:,} (synthetic, only if --synthetic)")
    log.info(f"  output model   : {MODEL_DIR / 'smpl_regressor.pth'}")
    log.info(f"  log file       : {LOG_FILE}")
    log.info("══════════════════════════════════════════════════════════")

    if args.dry_run:
        log.info("[DRY RUN] Will stop after 5 batches per epoch")

    # ── Dataset ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    datasets_to_combine = []

    # Synthetic data generation (commercial-safe approach)
    # Always generate synthetic data - no research dataset dependencies  
    synth_ds = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
    log.info(f"  + Synthetic dataset: {len(synth_ds):,} samples")
    datasets_to_combine.append(synth_ds)
    
    log.info("  Using synthetic-only training for commercial compatibility")

    # Combine all datasets
    if len(datasets_to_combine) > 1:
        dataset = torch.utils.data.ConcatDataset(datasets_to_combine)
        log.info(f"  Combined dataset: {len(dataset):,} total samples")
    else:
        dataset = datasets_to_combine[0]

    val_size   = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    log.info(f"Dataset: {train_size:,} train  |  {val_size:,} val  |  "
             f"{len(train_loader)} train batches  |  {len(val_loader)} val batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = dict(
        input_dim=99,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_iterations=args.num_iterations,
        shape_dim=10,
        pose_dim=72,
    )
    model = IterativeSMPLRegressor(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: IterativeSMPLRegressor  ({n_params/1e6:.2f}M params)")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing: warm start → decay to lr/100 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    start_epoch = 1
    best_val    = float("inf")
    latest_path = LOG_DIR / "smpl_regressor_latest.pth"
    best_path   = MODEL_DIR / "smpl_regressor.pth"

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and latest_path.exists():
        log.info(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("val_loss", float("inf"))
        log.info(f"  Resumed at epoch {start_epoch}  best_val={best_val:.6f}")
    elif args.resume:
        log.warning(f"--resume set but no checkpoint found at {latest_path}, starting fresh")

    # ── Training loop ─────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    log.info("─────────────── Training begins ───────────────────────")

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch}/{args.epochs}  lr={current_lr:.6f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        val_loss, val_shape, val_pose = validate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        ep_time = time.perf_counter() - t_ep
        total_elapsed = time.perf_counter() - t_total
        eta_epochs = args.epochs - epoch
        eta_sec    = eta_epochs * ep_time
        eta_str    = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"

        log.info(
            f"  ┌ Epoch {epoch:3d} complete ─────────────────────────\n"
            f"  │ train_loss  = {train_loss:.6f}\n"
            f"  │ val_loss    = {val_loss:.6f}  (shape={val_shape:.6f}  pose={val_pose:.6f})\n"
            f"  │ epoch_time  = {ep_time:.1f}s\n"
            f"  │ total_time  = {total_elapsed/60:.1f}min\n"
            f"  └ ETA         ≈ {eta_str}"
        )

        # Always save latest
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, latest_path, model_cfg)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path, model_cfg)
            log.info(f"  ★ New best val_loss={best_val:.6f}  → {best_path}")

        if args.dry_run:
            log.info("[DRY RUN] Stopping after first epoch")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    log.info("══════════════════════════════════════════════════════════")
    log.info("  Training complete")
    log.info(f"  Total time    : {total_time/60:.1f} minutes")
    log.info(f"  Best val_loss : {best_val:.6f}")
    log.info(f"  Model saved   : {best_path}")
    log.info(f"  Full log      : {LOG_FILE}")
    log.info("══════════════════════════════════════════════════════════")

    # Quick sanity check on saved model
    log.info("Running final inference sanity check…")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 99).to(device)
        s, p = model(dummy)
    log.info(f"  shape_params: {s.shape}  range [{s.min():.3f}, {s.max():.3f}]")
    log.info(f"  pose_params : {p.shape}  range [{p.min():.3f}, {p.max():.3f}]")
    log.info("  ✓ Sanity check passed")
    log.info(f"\nTo use the trained model, run:\n"
             f"  .\\.ar\\Scripts\\python.exe app.py\n"
             f"The model will be auto-loaded from: {best_path}")


if __name__ == "__main__":
    main()

"""
GMM Transform Logger

Logs θ parameters and TPS deformation grid from each GMM forward pass.
Computes transform stability metrics: θ drift, normalized grid displacement,
and affine determinant variance (detects scale-instability / garment breathing).

Usage:
    logger = GMMTransformLogger()
    # In _gmm_warp(), after model forward:
    logger.log_warp(theta.cpu().numpy(), grid.cpu().numpy())
    # On demand:
    stats = logger.get_stats()
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class GMMTransformLogger:
    """Track GMM θ parameters and grid per frame for stability analysis.
    
    Metrics:
      - theta_variance: element-wise variance of θ across buffer
      - theta_frame_drift: mean |θ_t − θ_{t-1}| per frame
      - normalized_grid_drift: mean |grid_t − grid_{t-1}| / mean |grid| (scale-invariant)
      - determinant_variance: variance of det(θ[:2,:2]) — detects breathing/shimmer
    """

    def __init__(self, buffer_size: int = 300):
        self._buffer_size = buffer_size
        
        # Raw θ values: shape (2,3) for affine or (N,) for TPS
        self._theta_buffer: deque = deque(maxlen=buffer_size)
        # Raw grid: shape (H, W, 2)
        self._grid_buffer: deque = deque(maxlen=buffer_size)
        
        # Pre-computed per-frame metrics (avoid recomputation in get_stats)
        self._theta_drifts: deque = deque(maxlen=buffer_size)
        self._grid_drifts: deque = deque(maxlen=buffer_size)  # normalized
        self._determinants: deque = deque(maxlen=buffer_size)
        
        self._prev_theta: Optional[np.ndarray] = None
        self._prev_grid: Optional[np.ndarray] = None
        self._frame_count = 0

    def log_warp(self, theta: np.ndarray, grid: np.ndarray):
        """Log one GMM forward pass.
        
        Args:
            theta: TPS/affine parameters — typically (1, 2, 3) or (1, N).
                   Squeezed to remove batch dim.
            grid:  Deformation grid — typically (1, H, W, 2).
                   Squeezed to remove batch dim.
        """
        # Remove batch dimension
        theta = np.squeeze(theta)
        grid = np.squeeze(grid)
        
        self._theta_buffer.append(theta.copy())
        self._grid_buffer.append(grid.copy())
        
        # θ frame drift
        if self._prev_theta is not None and theta.shape == self._prev_theta.shape:
            drift = float(np.mean(np.abs(theta - self._prev_theta)))
            self._theta_drifts.append(drift)
        else:
            self._theta_drifts.append(0.0)
        
        # Normalized grid drift: |grid_t - grid_{t-1}| / mean(|grid|)
        if self._prev_grid is not None and grid.shape == self._prev_grid.shape:
            grid_diff = float(np.mean(np.abs(grid - self._prev_grid)))
            grid_mag = float(np.mean(np.abs(grid)))
            norm_drift = grid_diff / max(grid_mag, 1e-8)
            self._grid_drifts.append(norm_drift)
        else:
            self._grid_drifts.append(0.0)
        
        # Affine determinant (if θ is 2×3 or can be reshaped to it)
        try:
            if theta.size >= 6:
                mat = theta.flatten()[:6].reshape(2, 3)
                det = float(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
                self._determinants.append(det)
            else:
                self._determinants.append(0.0)
        except Exception:
            self._determinants.append(0.0)
        
        self._prev_theta = theta.copy()
        self._prev_grid = grid.copy()
        self._frame_count += 1

    def get_stats(self) -> Dict:
        """Compute all transform stability metrics.
        
        Returns dict with:
          theta_variance, theta_frame_drift, normalized_grid_drift,
          determinant_mean, determinant_variance, total_frames
        """
        # θ element-wise variance
        if len(self._theta_buffer) >= 2:
            stacked = np.array([t.flatten() for t in self._theta_buffer])
            theta_var = float(np.mean(np.var(stacked, axis=0)))
        else:
            theta_var = 0.0
        
        # θ frame drift
        drifts = list(self._theta_drifts)
        theta_drift = float(np.mean(drifts)) if drifts else 0.0
        
        # Normalized grid drift
        gdrifts = list(self._grid_drifts)
        grid_drift = float(np.mean(gdrifts)) if gdrifts else 0.0
        
        # Determinant stats
        dets = list(self._determinants)
        det_mean = float(np.mean(dets)) if dets else 0.0
        det_var = float(np.var(dets)) if len(dets) >= 2 else 0.0
        
        return {
            'theta_variance': round(theta_var, 6),
            'theta_frame_drift': round(theta_drift, 6),
            'normalized_grid_drift': round(grid_drift, 6),
            'determinant_mean': round(det_mean, 4),
            'determinant_variance': round(det_var, 8),
            'total_frames': self._frame_count,
        }

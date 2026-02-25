"""
Landmark Stability Logger

Circular-buffer logger that records per-frame landmark positions (pixel space)
and computes jitter statistics broken down by joint and by static/moving state.

Usage:
    logger = LandmarkStabilityLogger()
    # Each frame, after pose detection:
    logger.log_frame(landmarks, frame_shape=(480, 640))
    # On demand:
    stats = logger.get_stats()
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, List

# MediaPipe landmark indices
_DIRECT_JOINTS = {
    'shoulder_L': 11,
    'shoulder_R': 12,
    'hip_L': 23,
    'hip_R': 24,
}
# Derived joints: midpoint of two indices
_DERIVED_JOINTS = {
    'neck': (11, 12),
    'mid_hip': (23, 24),
}

ALL_JOINT_NAMES = list(_DIRECT_JOINTS.keys()) + list(_DERIVED_JOINTS.keys())


class LandmarkStabilityLogger:
    """Track landmark positions and compute stability metrics.
    
    Stores last `buffer_size` frames of per-joint pixel positions.
    Computes:
      - mean_displacement_all: frame-to-frame L2 across all joints, all frames
      - mean_displacement_static_only: same but only during static-pose windows
      - max_displacement_static: worst-case jitter under static pose
      - variance_per_joint: positional variance per joint over buffer
      - per_joint_frame_drift: mean frame-to-frame displacement per joint
    """

    def __init__(self, buffer_size: int = 300, static_threshold_px: float = 3.0):
        """
        Args:
            buffer_size: Number of frames to retain (~10s at 30 FPS).
            static_threshold_px: Mean joint displacement below this marks frame as 'static'.
        """
        self._buffer_size = buffer_size
        self._static_threshold = static_threshold_px
        
        # Per-frame: {joint_name: (x_px, y_px)}
        self._positions: deque = deque(maxlen=buffer_size)
        # Per-frame: mean displacement from previous frame (float)
        self._frame_displacements: deque = deque(maxlen=buffer_size)
        # Per-frame: is_static flag
        self._static_flags: deque = deque(maxlen=buffer_size)
        # Per-frame: per-joint displacement from previous frame
        self._per_joint_displacements: deque = deque(maxlen=buffer_size)
        
        self._prev_positions: Optional[Dict[str, Tuple[float, float]]] = None
        self._frame_count = 0

    def log_frame(self, landmarks, frame_shape: Tuple[int, int]):
        """Record one frame of landmark data.
        
        Args:
            landmarks: MediaPipe NormalizedLandmark list (0-1 coords).
            frame_shape: (height, width) of the frame in pixels.
        """
        h, w = frame_shape
        positions: Dict[str, Tuple[float, float]] = {}
        
        # Direct joints
        for name, idx in _DIRECT_JOINTS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                positions[name] = (lm.x * w, lm.y * h)
        
        # Derived joints (midpoints)
        for name, (idx_a, idx_b) in _DERIVED_JOINTS.items():
            if idx_a < len(landmarks) and idx_b < len(landmarks):
                a, b = landmarks[idx_a], landmarks[idx_b]
                positions[name] = (
                    (a.x + b.x) * 0.5 * w,
                    (a.y + b.y) * 0.5 * h,
                )
        
        self._positions.append(positions)
        
        # Compute frame-to-frame displacement
        if self._prev_positions is not None:
            joint_disps: Dict[str, float] = {}
            disp_values: List[float] = []
            
            for name in ALL_JOINT_NAMES:
                if name in positions and name in self._prev_positions:
                    dx = positions[name][0] - self._prev_positions[name][0]
                    dy = positions[name][1] - self._prev_positions[name][1]
                    d = float(np.sqrt(dx * dx + dy * dy))
                    joint_disps[name] = d
                    disp_values.append(d)
            
            mean_disp = float(np.mean(disp_values)) if disp_values else 0.0
            is_static = mean_disp < self._static_threshold
            
            self._frame_displacements.append(mean_disp)
            self._static_flags.append(is_static)
            self._per_joint_displacements.append(joint_disps)
        else:
            self._frame_displacements.append(0.0)
            self._static_flags.append(True)
            self._per_joint_displacements.append({})
        
        self._prev_positions = positions
        self._frame_count += 1

    def get_last_displacement(self) -> float:
        """Return the most recent frame-to-frame displacement (for GPD gating)."""
        return self._frame_displacements[-1] if self._frame_displacements else 0.0

    def get_stats(self) -> Dict:
        """Compute all stability metrics over the current buffer.
        
        Returns dict with:
          mean_displacement_all, mean_displacement_static_only,
          max_displacement_static, variance_per_joint, per_joint_frame_drift,
          static_frame_ratio, total_frames
        """
        all_disps = list(self._frame_displacements)
        static_flags = list(self._static_flags)
        
        # All-frame displacement
        mean_all = float(np.mean(all_disps)) if all_disps else 0.0
        
        # Static-only displacement
        static_disps = [d for d, s in zip(all_disps, static_flags) if s]
        mean_static = float(np.mean(static_disps)) if static_disps else -1.0
        max_static = float(np.max(static_disps)) if static_disps else -1.0
        
        # Static frame ratio
        static_ratio = len(static_disps) / len(static_flags) if static_flags else 0.0
        
        # Per-joint positional variance
        variance_per_joint: Dict[str, float] = {}
        per_joint_drift: Dict[str, float] = {}
        
        for name in ALL_JOINT_NAMES:
            # Positional variance over buffer
            xs = [pos[name][0] for pos in self._positions if name in pos]
            ys = [pos[name][1] for pos in self._positions if name in pos]
            if len(xs) >= 2:
                variance_per_joint[name] = float(np.var(xs) + np.var(ys))
            else:
                variance_per_joint[name] = 0.0
            
            # Mean frame-to-frame drift per joint
            joint_d = [jd.get(name, 0.0) for jd in self._per_joint_displacements if name in jd]
            per_joint_drift[name] = float(np.mean(joint_d)) if joint_d else 0.0
        
        return {
            'mean_displacement_all': round(mean_all, 3),
            'mean_displacement_static_only': round(mean_static, 3),
            'max_displacement_static': round(max_static, 3),
            'variance_per_joint': {k: round(v, 3) for k, v in variance_per_joint.items()},
            'per_joint_frame_drift': {k: round(v, 3) for k, v in per_joint_drift.items()},
            'static_frame_ratio': round(static_ratio, 3),
            'total_frames': self._frame_count,
        }

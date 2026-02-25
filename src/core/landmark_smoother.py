"""
Pre-warp Landmark Smoother (Phase B1)

Per-joint weighted exponential moving average (EMA) filter
applied *before* landmarks are used for body measurement
extraction or GMM agnostic construction.

Design driven by Phase A data:
  - Hips jitter 2.3× more than neck → heavier smoothing on hips
  - Static jitter 1.5px → target <0.5px post-smoothing
  - Must not lag behind real movement → velocity-adaptive α
"""

import numpy as np
from collections import deque
from typing import Optional


# MediaPipe landmark indices for key joints
_JOINT_GROUPS = {
    'neck':       [0],        # Nose (proxy for head/neck)
    'shoulder_L': [11],
    'shoulder_R': [12],
    'elbow_L':    [13],
    'elbow_R':    [14],
    'wrist_L':    [15],
    'wrist_R':    [16],
    'hip_L':      [23],
    'hip_R':      [24],
    'knee_L':     [25],
    'knee_R':     [26],
    'ankle_L':    [27],
    'ankle_R':    [28],
}

# Per-joint base α (higher = less smoothing, faster response)
# Driven by Phase A per_joint_frame_drift data
_JOINT_ALPHA = {
    'neck':       0.60,   # 5.3px drift — most stable, light smoothing
    'shoulder_L': 0.50,   # 7.4px drift — moderate
    'shoulder_R': 0.50,   # 7.8px drift — moderate
    'elbow_L':    0.55,   # arms need responsiveness
    'elbow_R':    0.55,
    'wrist_L':    0.65,   # hands must track quickly
    'wrist_R':    0.65,
    'hip_L':      0.30,   # 9.3px drift — heavy smoothing
    'hip_R':      0.25,   # 12.4px drift — heaviest smoothing
    'knee_L':     0.35,
    'knee_R':     0.35,
    'ankle_L':    0.40,
    'ankle_R':    0.40,
}


class LandmarkSmoother:
    """Per-joint weighted EMA with velocity-adaptive alpha.
    
    When the user is nearly still, alpha is low (heavy smoothing).
    When they move quickly, alpha increases (responsive tracking).
    
    smoothed[t] = α * raw[t] + (1 - α) * smoothed[t-1]
    """
    
    def __init__(
        self,
        velocity_threshold: float = 5.0,
        velocity_boost: float = 2.0,
        max_alpha: float = 0.95,
    ):
        """
        Args:
            velocity_threshold: Pixel displacement above which we boost α.
            velocity_boost: Multiplier for α when velocity exceeds threshold.
            max_alpha: Hard cap on α to prevent raw pass-through.
        """
        self.velocity_threshold = velocity_threshold
        self.velocity_boost = velocity_boost
        self.max_alpha = max_alpha
        
        # State: smoothed positions per landmark index → (x, y, z)
        self._smoothed: dict[int, np.ndarray] = {}
        self._prev_raw: dict[int, np.ndarray] = {}
        
        # Build index → alpha lookup
        self._alpha: dict[int, float] = {}
        for group_name, indices in _JOINT_GROUPS.items():
            alpha = _JOINT_ALPHA.get(group_name, 0.5)
            for idx in indices:
                self._alpha[idx] = alpha
        
        # Default alpha for unlisted landmarks
        self._default_alpha = 0.5
        
        self.frame_count = 0
    
    def smooth(self, landmarks, frame_shape: tuple[int, int] = (720, 1280)):
        """Smooth landmarks in-place and return them.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList (list-like of landmarks
                       with .x, .y, .z, .visibility attributes).
            frame_shape: (height, width) for pixel-space velocity computation.
            
        Returns:
            The same landmark list with smoothed x, y, z values.
            Original values are NOT preserved — this modifies in place.
        """
        h, w = frame_shape
        self.frame_count += 1
        
        for idx in range(len(landmarks)):
            lm = landmarks[idx]
            raw = np.array([lm.x, lm.y, lm.z], dtype=np.float64)
            
            if idx not in self._smoothed:
                # First frame: initialize with raw
                self._smoothed[idx] = raw.copy()
                self._prev_raw[idx] = raw.copy()
                continue
            
            # Compute velocity in pixel space
            prev = self._prev_raw[idx]
            dx = (raw[0] - prev[0]) * w
            dy = (raw[1] - prev[1]) * h
            velocity = np.sqrt(dx * dx + dy * dy)
            
            # Adaptive alpha
            base_alpha = self._alpha.get(idx, self._default_alpha)
            if velocity > self.velocity_threshold:
                alpha = min(base_alpha * self.velocity_boost, self.max_alpha)
            else:
                alpha = base_alpha
            
            # EMA update
            smoothed = alpha * raw + (1.0 - alpha) * self._smoothed[idx]
            self._smoothed[idx] = smoothed
            self._prev_raw[idx] = raw.copy()
            
            # Write back to landmark (in-place)
            lm.x = float(smoothed[0])
            lm.y = float(smoothed[1])
            lm.z = float(smoothed[2])
        
        return landmarks
    
    def get_stats(self) -> dict:
        """Return smoother state for diagnostics."""
        return {
            'frame_count': self.frame_count,
            'tracked_landmarks': len(self._smoothed),
        }
    
    def reset(self):
        """Clear all state (e.g., when person leaves frame)."""
        self._smoothed.clear()
        self._prev_raw.clear()
        self.frame_count = 0

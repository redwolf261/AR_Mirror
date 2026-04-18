"""
Phase 3: Temporal Stabilization - Implementation Skeleton
Optical flow, motion compensation, and advanced EMA smoothing
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
from collections import deque


class OpticalFlowEstimator:
    """
    Estimate optical flow between consecutive frames using Farnebäck algorithm
    
    GPU-optimized version using CuPy for fast computation
    """
    
    def __init__(self, method: str = "farneback", use_gpu: bool = True):
        """
        Initialize optical flow estimator
        
        Args:
            method: "farneback" or "lucas_kanade"
            use_gpu: Use GPU acceleration if available
        """
        self.method = method
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Farnebäck parameters
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.n_poly = 5
        self.sigma = 1.2
        
        # State
        self.prev_gray = None
    
    def estimate_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow between two frames
        
        Args:
            prev_frame: Previous frame (H, W, 3)
            curr_frame: Current frame (H, W, 3)
        
        Returns:
            flow: (H, W, 2) motion vectors in pixels
        """
        if self.method == "farneback":
            return self._farneback_flow(prev_frame, curr_frame)
        else:
            return self._lucas_kanade_flow(prev_frame, curr_frame)
    
    def _farneback_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Farnebäck optical flow (recommended)"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,  # type: ignore
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.n_poly,
            poly_sigma=self.sigma,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
        return flow  # (H, W, 2)
    
    def _lucas_kanade_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Lucas-Kanade optical flow (sparse, for landmarks)"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        
        if corners is None:
            return np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2), dtype=np.float32)
        
        # Track corners
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None  # type: ignore
        )
        
        # Build dense flow from sparse corners via interpolation
        h, w = prev_frame.shape[:2]
        good_mask = status.ravel() == 1
        if not np.any(good_mask):
            return np.zeros((h, w, 2), dtype=np.float32)

        old_pts = corners[good_mask].reshape(-1, 2)
        new_pts = new_corners[good_mask].reshape(-1, 2)
        displacements = new_pts - old_pts  # (N, 2)

        # Grid coordinates for dense output
        grid_y, grid_x = np.mgrid[0:h, 0:w]

        try:
            from scipy.interpolate import griddata
            flow_x = griddata(old_pts, displacements[:, 0],
                              (grid_x, grid_y), method='linear', fill_value=0.0)
            flow_y = griddata(old_pts, displacements[:, 1],
                              (grid_x, grid_y), method='linear', fill_value=0.0)
        except ImportError:
            # Fallback: nearest-neighbor via OpenCV remap (no scipy)
            flow_x = np.zeros((h, w), dtype=np.float32)
            flow_y = np.zeros((h, w), dtype=np.float32)
            for pt, disp in zip(old_pts, displacements):
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                if 0 <= iy < h and 0 <= ix < w:
                    flow_x[iy, ix] = disp[0]
                    flow_y[iy, ix] = disp[1]

        flow = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
        return flow


class TemporalLandmarkStabilization:
    """
    Stabilize landmark positions using optical flow
    
    Combines optical flow motion estimation with EMA smoothing
    """
    
    def __init__(self, alpha: float = 0.6):
        """
        Initialize landmark stabilization
        
        Args:
            alpha: EMA smoothing factor (0.0-1.0)
        """
        self.alpha = alpha
        self.prev_landmarks = None
        self.prev_frame = None
        self.optical_flow = OpticalFlowEstimator(method="farneback")
    
    def stabilize(self, landmarks: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
        """
        Stabilize landmarks using optical flow
        
        Args:
            landmarks: (N, 2) landmark positions
            current_frame: (H, W, 3) RGB frame
        
        Returns:
            stable_landmarks: (N, 2) stabilized positions
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            self.prev_frame = current_frame.copy()
            return landmarks
        
        # Estimate optical flow
        if self.prev_frame is not None:
            flow = self.optical_flow.estimate_flow(self.prev_frame, current_frame)
        else:
            flow = np.zeros((current_frame.shape[0], current_frame.shape[1], 2), dtype=np.float32)
        
        # Get motion vectors for landmarks
        motion_vectors = self._get_landmark_motion(self.prev_landmarks, flow)
        
        # Predict landmark positions
        predicted_landmarks = self.prev_landmarks + motion_vectors * 0.8
        
        # Apply EMA smoothing
        stable_landmarks = self.alpha * landmarks + (1 - self.alpha) * predicted_landmarks
        
        # Update state
        self.prev_landmarks = stable_landmarks.copy()
        self.prev_frame = current_frame.copy()
        
        return stable_landmarks
    
    def _get_landmark_motion(self, landmarks: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Get motion vectors for each landmark"""
        motion_vectors = np.zeros_like(landmarks)
        
        for i, (x, y) in enumerate(landmarks):
            ix, iy = int(np.round(x)), int(np.round(y))
            
            # Boundary check
            if 0 <= iy < flow.shape[0] and 0 <= ix < flow.shape[1]:
                motion_vectors[i] = flow[iy, ix]
        
        return motion_vectors


class WarpingTemporalStabilization:
    """
    Stabilize warping grid using motion compensation
    
    Prevents warping grid from oscillating due to landmark jitter
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize warping stabilization
        
        Args:
            alpha: EMA smoothing factor
        """
        self.alpha = alpha
        self.prev_warp_grid = None
        self.optical_flow = OpticalFlowEstimator()
    
    def stabilize(self, warp_grid: np.ndarray, frame: np.ndarray, 
                 landmarks: np.ndarray) -> np.ndarray:
        """
        Stabilize warping grid
        
        Args:
            warp_grid: (H//K, W//K, 2) grid
            frame: (H, W, 3) RGB frame
            landmarks: (N, 2) body landmarks
        
        Returns:
            stable_grid: Stabilized warp grid
        """
        if self.prev_warp_grid is None:
            self.prev_warp_grid = warp_grid.copy()
            return warp_grid
        
        # Estimate motion scale
        motion_scale = self._estimate_motion_scale(landmarks)
        
        # Apply predicted smoothing
        stable_grid = (self.alpha * warp_grid + 
                      (1 - self.alpha) * self.prev_warp_grid * motion_scale)
        
        # Apply deformation constraints
        stable_grid = self._apply_constraints(stable_grid)
        
        self.prev_warp_grid = stable_grid.copy()
        
        return stable_grid
    
    def _estimate_motion_scale(self, landmarks: np.ndarray) -> float:
        """Estimate global motion scale from landmarks.
        
        Compares current landmarks to previously stored landmarks
        and returns a scale factor proportional to the mean displacement.
        """
        if not hasattr(self, '_prev_landmarks') or self._prev_landmarks is None:
            self._prev_landmarks = landmarks.copy()
            return 1.0
        
        # Mean displacement magnitude (in pixels)
        displacements = np.linalg.norm(landmarks - self._prev_landmarks, axis=-1)
        mean_disp = float(np.mean(displacements))
        
        self._prev_landmarks = landmarks.copy()
        
        # Scale: 1.0 = stationary, > 1.0 = moving fast, clamped to [0.8, 1.2]
        scale = 1.0 + np.clip(mean_disp / 50.0, -0.2, 0.2)
        return float(scale)
    
    def _apply_constraints(self, grid: np.ndarray) -> float:
        """Apply deformation constraints to prevent extreme grid distortion.
        
        Clamps per-cell displacement relative to an identity grid so that
        no single cell moves more than max_deformation fraction of the grid size.
        """
        max_deformation = 0.15  # max displacement as fraction of grid size
        
        # Build an identity grid for comparison
        h, w = grid.shape[:2]
        identity_y = np.linspace(0.0, 1.0, h).reshape(-1, 1).repeat(w, axis=1)
        identity_x = np.linspace(0.0, 1.0, w).reshape(1, -1).repeat(h, axis=0)
        identity = np.stack([identity_x, identity_y], axis=-1).astype(np.float32)
        
        # Scale identity to match grid range
        grid_min = grid.min(axis=(0, 1), keepdims=True)
        grid_max = grid.max(axis=(0, 1), keepdims=True)
        grid_range = np.maximum(grid_max - grid_min, 1e-6)
        identity_scaled = identity * grid_range + grid_min
        
        # Clamp displacement
        displacement = grid - identity_scaled
        max_abs = max_deformation * grid_range
        displacement = np.clip(displacement, -max_abs, max_abs)
        
        return identity_scaled + displacement


class AdaptiveEMA:
    """
    Exponential Moving Average with adaptive alpha
    
    Adapts smoothing factor based on motion magnitude
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize adaptive EMA
        
        Args:
            window_size: Number of frames to track motion
        """
        self.prev_value = None
        self.motion_history = deque(maxlen=window_size)
        self.window_size = window_size
    
    def smooth(self, current_value: np.ndarray, motion_magnitude: float) -> np.ndarray:
        """
        Apply adaptive EMA smoothing
        
        Args:
            current_value: Current value
            motion_magnitude: Magnitude of detected motion
        
        Returns:
            smoothed_value: EMA-smoothed value
        """
        if self.prev_value is None:
            self.prev_value = current_value.copy()
            return current_value
        
        # Calculate adaptive alpha
        alpha = self._adaptive_alpha(motion_magnitude)
        
        # Apply EMA
        smoothed = alpha * current_value + (1 - alpha) * self.prev_value
        
        # Track motion
        self.motion_history.append(motion_magnitude)
        
        self.prev_value = smoothed.copy()
        
        return smoothed
    
    def _adaptive_alpha(self, motion_magnitude: float) -> float:
        """Calculate alpha based on motion magnitude"""
        if motion_magnitude < 0.5:
            return 0.80  # High smoothing for low motion
        elif motion_magnitude < 1.0:
            return 0.70
        elif motion_magnitude < 2.0:
            return 0.50
        else:
            return 0.30  # Minimal smoothing for high motion
    
    def get_average_motion(self) -> float:
        """Get average motion from history"""
        if len(self.motion_history) == 0:
            return 0.0
        return float(np.mean(list(self.motion_history)))


class TemporalFilter:
    """
    Advanced multi-scale temporal filtering
    
    Combines multiple smoothing scales for balanced quality
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Initialize temporal filter
        
        Args:
            buffer_size: Number of frames to buffer
        """
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
    
    def apply_filter(self, current_value: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale temporal filtering
        
        Args:
            current_value: Current frame value
        
        Returns:
            filtered_value: Temporally filtered value
        """
        self.frame_buffer.append(current_value.copy())
        
        if len(self.frame_buffer) == 0:
            return current_value
        
        # Calculate weighted average
        # Recent frames weighted more heavily
        weights = np.linspace(0.5, 1.0, len(self.frame_buffer))
        weights /= weights.sum()
        
        filtered = np.zeros_like(current_value, dtype=np.float32)
        
        for i, (frame, weight) in enumerate(zip(self.frame_buffer, weights)):
            filtered += frame.astype(np.float32) * weight
        
        return filtered.astype(current_value.dtype)
    
    def clear(self):
        """Clear buffer"""
        self.frame_buffer.clear()


class TemporalStabilizationPipeline:
    """
    Complete temporal stabilization pipeline
    
    Combines optical flow, motion compensation, and adaptive smoothing
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize temporal stabilization pipeline
        
        Args:
            use_gpu: Use GPU acceleration if available
        """
        self.optical_flow = OpticalFlowEstimator(use_gpu=use_gpu)
        self.landmark_stabilizer = TemporalLandmarkStabilization(alpha=0.6)
        self.warping_stabilizer = WarpingTemporalStabilization(alpha=0.5)
        self.adaptive_ema = AdaptiveEMA(window_size=5)
        self.temporal_filter = TemporalFilter(buffer_size=5)
        
        self.prev_frame = None
    
    def stabilize_landmarks(self, landmarks: np.ndarray, 
                           current_frame: np.ndarray) -> np.ndarray:
        """Stabilize landmarks"""
        return self.landmark_stabilizer.stabilize(landmarks, current_frame)
    
    def stabilize_warp_grid(self, warp_grid: np.ndarray, 
                           frame: np.ndarray, 
                           landmarks: np.ndarray) -> np.ndarray:
        """Stabilize warp grid"""
        return self.warping_stabilizer.stabilize(warp_grid, frame, landmarks)
    
    def get_estimated_flow(self, prev_frame: np.ndarray, 
                          curr_frame: np.ndarray) -> np.ndarray:
        """Get optical flow"""
        return self.optical_flow.estimate_flow(prev_frame, curr_frame)


# Metrics computation
def calculate_jitter(position_sequence: np.ndarray) -> float:
    """
    Calculate jitter (frame-to-frame position variation)
    
    Args:
        position_sequence: (T, N, 2) positions over time
    
    Returns:
        jitter: Standard deviation of frame-to-frame differences
    """
    if len(position_sequence) < 2:
        return 0.0
    
    differences = np.diff(position_sequence, axis=0)
    jitter = np.mean(np.std(differences, axis=0))
    
    return jitter


def calculate_temporal_coherence(sequence: np.ndarray) -> float:
    """
    Calculate temporal coherence (consistency over time)
    
    Args:
        sequence: (T, ...) values over time
    
    Returns:
        coherence: 1.0 - normalized variance (higher = more coherent)
    """
    if len(sequence) == 0:
        return 1.0
    
    variance = np.var(sequence)
    # Normalize by range
    range_val = np.max(sequence) - np.min(sequence)
    
    if range_val == 0:
        return 1.0
    
    normalized_var = variance / (range_val ** 2)
    coherence = 1.0 - normalized_var
    
    return np.clip(coherence, 0.0, 1.0)


if __name__ == "__main__":
    print("Phase 3 Temporal Stabilization Module")
    print("=" * 50)
    print("\nClasses:")
    print("  - OpticalFlowEstimator: Compute optical flow")
    print("  - TemporalLandmarkStabilization: Stabilize landmarks")
    print("  - WarpingTemporalStabilization: Stabilize warping grid")
    print("  - AdaptiveEMA: Adaptive exponential moving average")
    print("  - TemporalFilter: Multi-scale temporal filtering")
    print("  - TemporalStabilizationPipeline: Complete pipeline")
    print("\nMetrics:")
    print("  - calculate_jitter: Frame-to-frame variation")
    print("  - calculate_temporal_coherence: Consistency over time")

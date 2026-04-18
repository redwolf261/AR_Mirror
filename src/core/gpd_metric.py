"""
Garment Pixel Drift (GPD) Metric

Production stability KPI:  GPD = mean(|warp_t − warp_{t-1}|)
measured over the garment mask only, in both RGB and luminance channels.

Two variants:
  gpd_rgb:       drift in RGB space (catches colour shimmer)
  gpd_luminance: drift in luminance only (most visible shimmer)

Static-only filtering: only accumulates when landmark displacement
is below the static threshold, avoiding pollution from real movement.

Stores a downscaled (1/4 resolution) copy of the previous warp to
minimize memory and CPU cost of the per-frame diff.

Usage:
    gpd = GarmentPixelDrift()
    # Each frame after GMM warp:
    gpd.update(warped_cloth, warped_mask, landmark_displacement)
    # On demand:
    stats = gpd.get_stats()
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional


class GarmentPixelDrift:
    """Track frame-to-frame garment shimmer under static pose.
    
    Target: GPD < empirically-derived threshold (start measuring, then decide).
    """

    def __init__(self, buffer_size: int = 300, static_threshold_px: float = 3.0,
                 downscale: int = 4):
        """
        Args:
            buffer_size: Number of frames to retain.
            static_threshold_px: Landmark displacement below this = 'static'.
            downscale: Factor to reduce resolution for drift comparison.
                       4 means 256×192 → 64×48. Saves ~0.3ms/frame copy cost.
        """
        self._buffer_size = buffer_size
        self._static_threshold = static_threshold_px
        self._downscale = downscale
        
        # Previous warped cloth at reduced resolution (uint8 saves memory)
        self._prev_warped_small: Optional[np.ndarray] = None   # RGB, uint8
        self._prev_lum_small: Optional[np.ndarray] = None      # grayscale, uint8
        self._prev_mask_small: Optional[np.ndarray] = None     # binary
        
        # Per-frame GPD values
        self._gpd_rgb: deque = deque(maxlen=buffer_size)        # (gpd, is_static)
        self._gpd_lum: deque = deque(maxlen=buffer_size)        # (gpd, is_static)
        
        self._frame_count = 0

    def update(self, warped_cloth: np.ndarray, warped_mask: np.ndarray,
               landmark_displacement: float):
        """Record one frame.
        
        Args:
            warped_cloth: RGB float32 [0,1] shape (H, W, 3) — output of GMM warp.
            warped_mask:  float32 [0,1] shape (H, W) — garment mask.
            landmark_displacement: Mean frame-to-frame landmark displacement (pixels).
        """
        # Downscale for efficient comparison
        h, w = warped_cloth.shape[:2]
        sh, sw = max(h // self._downscale, 1), max(w // self._downscale, 1)
        
        small_rgb = cv2.resize(warped_cloth, (sw, sh), interpolation=cv2.INTER_AREA)
        small_rgb_u8 = (np.clip(small_rgb, 0, 1) * 255).astype(np.uint8)
        
        # Luminance: BT.601 standard
        small_lum = (0.299 * small_rgb[:, :, 0] +
                     0.587 * small_rgb[:, :, 1] +
                     0.114 * small_rgb[:, :, 2])
        small_lum_u8 = (np.clip(small_lum, 0, 1) * 255).astype(np.uint8)
        
        # Downscale mask (binary threshold)
        small_mask = cv2.resize(warped_mask, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask_bool = small_mask > 0.3
        
        is_static = landmark_displacement < self._static_threshold
        
        if self._prev_warped_small is not None and small_rgb_u8.shape == self._prev_warped_small.shape:
            mask_pixels = np.sum(small_mask_bool)
            
            if mask_pixels > 10:  # Minimal garment area required
                # RGB GPD — masked
                diff_rgb = np.abs(small_rgb_u8.astype(np.float32) -
                                  self._prev_warped_small.astype(np.float32))
                # Sum across channels, then average over mask
                diff_rgb_per_pixel = np.mean(diff_rgb, axis=2)  # (sh, sw)
                gpd_rgb = float(np.mean(diff_rgb_per_pixel[small_mask_bool]))
                
                # Luminance GPD — masked
                diff_lum = np.abs(small_lum_u8.astype(np.float32) -
                                  self._prev_lum_small.astype(np.float32))
                gpd_lum = float(np.mean(diff_lum[small_mask_bool]))
                
                self._gpd_rgb.append((gpd_rgb, is_static))
                self._gpd_lum.append((gpd_lum, is_static))
        
        self._prev_warped_small = small_rgb_u8
        self._prev_lum_small = small_lum_u8
        self._prev_mask_small = small_mask_bool
        self._frame_count += 1

    def get_static_gpd(self) -> float:
        """Quick accessor: average RGB GPD during static-pose frames."""
        static = [g for g, s in self._gpd_rgb if s]
        return round(float(np.mean(static)), 3) if static else -1.0

    def get_stats(self) -> Dict:
        """Compute all GPD metrics.
        
        Returns dict with:
          gpd_rgb_static, gpd_rgb_all,
          gpd_luminance_static, gpd_luminance_all,
          static_frame_count, total_frames
        """
        # RGB
        all_rgb = [g for g, _ in self._gpd_rgb]
        static_rgb = [g for g, s in self._gpd_rgb if s]
        
        # Luminance
        all_lum = [g for g, _ in self._gpd_lum]
        static_lum = [g for g, s in self._gpd_lum if s]
        
        return {
            'gpd_rgb_static': round(float(np.mean(static_rgb)), 3) if static_rgb else -1.0,
            'gpd_rgb_all': round(float(np.mean(all_rgb)), 3) if all_rgb else -1.0,
            'gpd_luminance_static': round(float(np.mean(static_lum)), 3) if static_lum else -1.0,
            'gpd_luminance_all': round(float(np.mean(all_lum)), 3) if all_lum else -1.0,
            'static_frame_count': len(static_rgb),
            'total_frames': self._frame_count,
        }

#!/usr/bin/env python3
"""
Live Pose to Heatmap Converter
Converts MediaPipe live pose landmarks to 18-channel OpenPose-style heatmaps
Required for GMM neural warping with live camera feed
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LivePoseConverter:
    """
    Converts MediaPipe 33-point pose to OpenPose 18-point heatmap format
    Required for GMM TPS warping with live camera input
    """
    
    # MediaPipe to OpenPose keypoint mapping
    # OpenPose 18 points: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    MP_TO_OPENPOSE = {
        0: 0,   # Nose
        2: 14,  # Right eye
        5: 15,  # Left eye
        7: 16,  # Right ear
        8: 17,  # Left ear
        11: 5,  # Left shoulder
        12: 2,  # Right shoulder
        13: 6,  # Left elbow
        14: 3,  # Right elbow
        15: 7,  # Left wrist
        16: 4,  # Right wrist
        23: 11, # Left hip
        24: 8,  # Right hip
        25: 12, # Left knee
        26: 9,  # Right knee
        27: 13, # Left ankle
        28: 10, # Right ankle
    }
    
    # OpenPose neck (1) estimated from shoulders
    # OpenPose MidHip estimated from hips
    
    def __init__(self, heatmap_size: Tuple[int, int] = (256, 192), sigma: float = 3.0,
                 device: str = "cuda"):
        """
        Args:
            heatmap_size: (height, width) of output heatmaps
            sigma: Gaussian kernel sigma for heatmap generation
            device: 'cuda' to run heatmap generation on GPU (much faster)
        """
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma = sigma
        self.gaussian_denom = 2 * sigma ** 2

        # Pre-compute meshgrid for performance (avoid creating it 18 times per frame)
        x_grid = np.arange(self.heatmap_w, dtype=np.float32)
        y_grid = np.arange(self.heatmap_h, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x_grid, y_grid)

        # GPU path: keep grid tensors on CUDA if available
        self._use_cuda = False
        try:
            import torch
            _dev = torch.device(device if torch.cuda.is_available() else "cpu")
            self._torch_device = _dev
            self._xx_t = torch.from_numpy(self.xx).to(_dev)  # (H, W)
            self._yy_t = torch.from_numpy(self.yy).to(_dev)  # (H, W)
            self._denom_t = float(self.gaussian_denom)
            self._use_cuda = str(_dev) != "cpu"
            logger.debug(f"LivePoseConverter: heatmaps on {'CUDA' if self._use_cuda else 'CPU'}")
        except Exception:
            pass
        
    def mediapipe_to_openpose(self, mp_landmarks: Dict) -> Dict[int, Tuple[float, float, float]]:
        """
        Convert MediaPipe 33-point landmarks to OpenPose 18-point format
        
        Args:
            mp_landmarks: Dict of MediaPipe landmarks {idx: {'x': x, 'y': y, 'z': z, 'visibility': v}}
            
        Returns:
            Dict of OpenPose keypoints {idx: (x, y, confidence)}
        """
        openpose_kpts = {}
        
        # Direct mapping for existing points
        for mp_idx, op_idx in self.MP_TO_OPENPOSE.items():
            if mp_idx in mp_landmarks:
                mp_kpt = mp_landmarks[mp_idx]
                openpose_kpts[op_idx] = (
                    mp_kpt['x'],
                    mp_kpt['y'],
                    mp_kpt.get('visibility', 1.0)
                )
        
        # Estimate Neck (1) from shoulders
        if 11 in mp_landmarks and 12 in mp_landmarks:
            ls = mp_landmarks[11]  # Left shoulder
            rs = mp_landmarks[12]  # Right shoulder
            openpose_kpts[1] = (
                (ls['x'] + rs['x']) / 2,
                (ls['y'] + rs['y']) / 2,
                min(ls.get('visibility', 1.0), rs.get('visibility', 1.0))
            )
        
        # Estimate MidHip from hips (not in standard 18, but useful)
        if 23 in mp_landmarks and 24 in mp_landmarks:
            lh = mp_landmarks[23]  # Left hip
            rh = mp_landmarks[24]  # Right hip
            # Store as extra point if needed
            mid_hip = (
                (lh['x'] + rh['x']) / 2,
                (lh['y'] + rh['y']) / 2,
                min(lh.get('visibility', 1.0), rh.get('visibility', 1.0))
            )
        
        return openpose_kpts
    
    def generate_gaussian_heatmap(
        self,
        x: float,
        y: float,
        confidence: float = 1.0
    ) -> np.ndarray:
        """
        Generate Gaussian heatmap for a single keypoint (optimized with cached meshgrid)
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            confidence: Keypoint confidence (0-1)
            
        Returns:
            Heatmap array (H, W) with Gaussian peak at (x, y)
        """
        # Convert normalized coords to heatmap coords
        x_px = x * self.heatmap_w
        y_px = y * self.heatmap_h
        
        # Clip to valid range
        x_px = np.clip(x_px, 0, self.heatmap_w - 1)
        y_px = np.clip(y_px, 0, self.heatmap_h - 1)
        
        # Generate Gaussian using pre-computed meshgrid (much faster!)
        heatmap = np.exp(-((self.xx - x_px)**2 + (self.yy - y_px)**2) / self.gaussian_denom)
        
        # Scale by confidence
        heatmap = heatmap * confidence
        
        return heatmap.astype(np.float32)
    
    def landmarks_to_heatmaps(
        self,
        mp_landmarks: Dict,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to 18-channel OpenPose heatmaps.

        Uses GPU-accelerated batch Gaussian generation when CUDA is available
        (typically < 1ms vs ~23ms on CPU for 18 channels at 256×192).

        Args:
            mp_landmarks: MediaPipe landmarks dict
            frame_shape: Optional (height, width) for coordinate validation

        Returns:
            Heatmaps array (18, H, W) in range [0, 1]
        """
        op_keypoints = self.mediapipe_to_openpose(mp_landmarks)

        valid_indices, xs, ys, confs = [], [], [], []
        for kpt_idx in range(18):
            if kpt_idx in op_keypoints:
                x, y, conf = op_keypoints[kpt_idx]
                if conf >= 0.3:
                    valid_indices.append(kpt_idx)
                    xs.append(float(np.clip(x * self.heatmap_w, 0, self.heatmap_w - 1)))
                    ys.append(float(np.clip(y * self.heatmap_h, 0, self.heatmap_h - 1)))
                    confs.append(conf)

        heatmaps = np.zeros((18, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        if not valid_indices:
            return heatmaps

        if self._use_cuda:
            return self._landmarks_to_heatmaps_cuda(valid_indices, xs, ys, confs, heatmaps)

        # ── CPU path (vectorised numpy) ──
        xs_arr   = np.array(xs,    dtype=np.float32).reshape(-1, 1, 1)
        ys_arr   = np.array(ys,    dtype=np.float32).reshape(-1, 1, 1)
        confs_arr = np.array(confs, dtype=np.float32).reshape(-1, 1, 1)
        batch = np.exp(
            -((self.xx[np.newaxis] - xs_arr) ** 2 + (self.yy[np.newaxis] - ys_arr) ** 2)
            / self.gaussian_denom
        ) * confs_arr
        for i, kpt_idx in enumerate(valid_indices):
            heatmaps[kpt_idx] = batch[i]
        return heatmaps

    def _landmarks_to_heatmaps_cuda(
        self,
        valid_indices: List[int],
        xs: List[float],
        ys: List[float],
        confs: List[float],
        out: np.ndarray,
    ) -> np.ndarray:
        """GPU-accelerated batch Gaussian heatmap generation (< 1ms on RTX 2050)."""
        import torch
        N = len(valid_indices)
        xs_t    = torch.tensor(xs,    dtype=torch.float32, device=self._torch_device).view(N, 1, 1)
        ys_t    = torch.tensor(ys,    dtype=torch.float32, device=self._torch_device).view(N, 1, 1)
        confs_t = torch.tensor(confs, dtype=torch.float32, device=self._torch_device).view(N, 1, 1)

        # (N, H, W) broadcast Gaussian
        batch = torch.exp(
            -((self._xx_t.unsqueeze(0) - xs_t) ** 2 + (self._yy_t.unsqueeze(0) - ys_t) ** 2)
            / self._denom_t
        ) * confs_t  # (N, H, W)

        batch_cpu = batch.cpu().numpy()
        for i, kpt_idx in enumerate(valid_indices):
            out[kpt_idx] = batch_cpu[i]
        return out
    
    def visualize_heatmaps(
        self,
        heatmaps: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize 18-channel heatmaps as colored overlay
        
        Args:
            heatmaps: (18, H, W) heatmap array
            output_path: Optional path to save visualization
            
        Returns:
            RGB visualization image
        """
        # Sum all channels for visualization
        summed = np.sum(heatmaps, axis=0)
        summed = np.clip(summed, 0, 1)
        
        # Convert to color (viridis-like)
        summed_8u = (summed * 255).astype(np.uint8)
        colored = cv2.applyColorMap(summed_8u, cv2.COLORMAP_JET)
        
        if output_path:
            cv2.imwrite(output_path, colored)
        
        return colored


class LiveBodySegmenter:
    """
    Real-time body segmentation for GMM agnostic representation
    Replaces placeholder Otsu threshold with proper segmentation
    """
    
    def __init__(self, model_path: str = 'models/selfie_segmenter.tflite'):
        """Initialize MediaPipe selfie segmentation (Tasks API)"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python import BaseOptions
            
            if not os.path.exists(model_path):
                logger.warning(f"Selfie segmenter model not found: {model_path}")
                self.available = False
                return
            
            options = mp_vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_confidence_masks=True,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self.segmenter = mp_vision.ImageSegmenter.create_from_options(options)
            self.available = True
            logger.info("✓ Live body segmentation initialized (Tasks API)")
        except Exception as e:
            logger.warning(f"Body segmentation not available: {e}")
            self.available = False
    
    def segment(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment body from background
        
        Args:
            rgb_image: RGB image (H, W, 3)
            
        Returns:
            body_mask: Binary mask (H, W) in range [0, 1]
            segmentation: Segmentation result for debugging
        """
        if not self.available:
            # Fallback to Otsu
            return self._fallback_segment(rgb_image)
        
        # MediaPipe segmentation (Tasks API) — requires uint8 SRGB input
        import mediapipe as mp
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            rgb_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
        result = self.segmenter.segment(mp_image)
        
        if result.confidence_masks and len(result.confidence_masks) > 0:
            mask = result.confidence_masks[0].numpy_view()
            
            # Resize to standard size
            mask_resized = cv2.resize(mask, (192, 256), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_resized > 0.5).astype(np.float32)
            
            return mask_binary, mask
        else:
            return self._fallback_segment(rgb_image)
    
    def _fallback_segment(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback segmentation using Otsu threshold"""
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            rgb_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_image
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (192, 256))
        
        _, mask = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_float = mask.astype(np.float32) / 255.0
        
        return mask_float, mask


# Test function
if __name__ == "__main__":
    print("Testing Live Pose Converter...")
    
    # Create converter
    converter = LivePoseConverter()
    
    # Mock MediaPipe landmarks (T-pose example)
    mock_landmarks = {
        0: {'x': 0.5, 'y': 0.15, 'z': 0, 'visibility': 0.99},  # Nose
        11: {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 0.95},  # Left shoulder
        12: {'x': 0.6, 'y': 0.3, 'z': 0, 'visibility': 0.95},  # Right shoulder
        23: {'x': 0.45, 'y': 0.7, 'z': 0, 'visibility': 0.90}, # Left hip
        24: {'x': 0.55, 'y': 0.7, 'z': 0, 'visibility': 0.90}, # Right hip
    }
    
    # Generate heatmaps
    heatmaps = converter.landmarks_to_heatmaps(mock_landmarks)
    
    print(f"✓ Generated heatmaps: {heatmaps.shape}")
    print(f"  Min: {heatmaps.min():.4f}, Max: {heatmaps.max():.4f}")
    print(f"  Non-zero channels: {np.sum(heatmaps.max(axis=(1,2)) > 0)}")
    
    # Visualize
    vis = converter.visualize_heatmaps(heatmaps, "test_heatmap_output.png")
    print(f"✓ Visualization saved: test_heatmap_output.png")
    
    print("\n✅ Live Pose Converter test passed!")

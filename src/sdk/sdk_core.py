"""
AR Mirror SDK Core (Phase C4)

Clean SDK entry point: process_frame(image) → composited_image.
No UI, no OpenCV windows, no keyboard handling.

Usage:
    from src.sdk.sdk_core import ARMirrorSDK
    
    sdk = ARMirrorSDK()
    sdk.load_garment("path/to/cloth.jpg", "path/to/mask.jpg")
    
    composited = sdk.process_frame(bgr_frame)
    diagnostics = sdk.get_diagnostics()
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ARMirrorSDK:
    """Minimal SDK for virtual garment try-on.
    
    Wraps the full pipeline (pose → smooth → warp → composite)
    behind a single process_frame() call.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SDK.
        
        Args:
            config: Optional configuration dict. Keys:
                - device: 'auto', 'cuda', 'cpu' (default: 'auto')
                - enable_smoothing: bool (default: True)
                - enable_tom: bool (default: False)
                - use_densepose: bool (default: False) - Use DensePose for 3D body mapping
                - render_mode: str (default: 'neural_warp') - Rendering mode:
                    * 'neural_warp': GMM+TOM (21 FPS)
                    * 'neural_densepose': GMM+DensePose (15 FPS, better fit)
                    * 'hybrid_cached': Cached diffusion + real-time (30 FPS display)
                    * 'cloud_api': Cloud diffusion API (1-5s, best quality)
                - cloud_api_key: str (optional) - API key for cloud rendering
                - resolution: tuple (H, W) for input frames
        """
        self._config = config or {}
        device = self._config.get('device', 'auto')
        enable_tom = self._config.get('enable_tom', False)
        use_densepose = self._config.get('use_densepose', False)
        render_mode_str = self._config.get('render_mode', 'neural_warp')
        cloud_api_key = self._config.get('cloud_api_key', None)
        
        # State
        self._cloth_rgb: Optional[np.ndarray] = None
        self._cloth_mask: Optional[np.ndarray] = None
        self._frame_count = 0
        self._total_time = 0.0
        self._use_densepose = use_densepose
        
        # Parse render mode
        from src.pipelines.diffusion_renderer import RenderMode, MultiModalRenderer
        try:
            self._render_mode = RenderMode(render_mode_str)
        except ValueError:
            logger.warning(f"Unknown render mode '{render_mode_str}', using 'neural_warp'")
            self._render_mode = RenderMode.NEURAL_WARP
        
        # --- Initialize components ---
        logger.info("ARMirrorSDK: Initializing...")
        
        # Body-aware fitter (includes landmark logger + smoother + static lock)
        from src.core.body_aware_fitter import BodyAwareGarmentFitter
        self._body_fitter = BodyAwareGarmentFitter()
        
        # Phase 2 neural pipeline (includes transform logger + GPD)
        self._pipeline = None
        self._renderer = None
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            self._pipeline = Phase2NeuralPipeline(
                device=device,
                enable_tom=enable_tom,
                batch_size=1,
                enable_optimizations=True,
            )
            
            # Initialize multi-modal renderer
            self._renderer = MultiModalRenderer(
                neural_pipeline=self._pipeline,
                cache_dir="data/diffusion_cache",
                cloud_api_key=cloud_api_key
            )
            logger.info(f"Render mode: {self._render_mode.value}")
            
        except Exception as e:
            logger.warning(f"Neural pipeline not available: {e}")
            logger.warning("Falling back to geometric fitting only.")
        
        logger.info("ARMirrorSDK: Ready.")
    
    def load_garment(self, cloth_path: str, mask_path: str,
                     target_size: Optional[tuple] = None) -> bool:
        """Load a garment image and mask.
        
        Args:
            cloth_path: Path to garment RGB image.
            mask_path: Path to garment binary mask.
            target_size: Optional (width, height) to resize to.
            
        Returns:
            True if loaded successfully.
        """
        cloth_bgr = cv2.imread(cloth_path, cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if cloth_bgr is None or mask_gray is None:
            logger.error(f"Failed to load garment: {cloth_path}")
            return False
        
        cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
        
        if target_size is not None:
            cloth_rgb = cv2.resize(cloth_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            mask_gray = cv2.resize(mask_gray, target_size, interpolation=cv2.INTER_NEAREST)
        
        self._cloth_rgb = cloth_rgb.astype(np.float32) / 255.0
        self._cloth_mask = (mask_gray > 127).astype(np.float32)
        self._cloth_mask = np.expand_dims(self._cloth_mask, axis=-1)
        
        logger.info(f"Garment loaded: {Path(cloth_path).name} ({cloth_rgb.shape[1]}×{cloth_rgb.shape[0]})")
        return True
    
    def load_garment_from_dataset(self, dataset_root: str, cloth_filename: str) -> bool:
        """Load a garment from a VITON-format dataset.
        
        Args:
            dataset_root: Path to dataset train/ directory.
            cloth_filename: Filename of the cloth image (e.g., '03195_00.jpg').
            
        Returns:
            True if loaded successfully.
        """
        from src.app.rendering import load_viton_cloth
        cloth_rgb, cloth_mask = load_viton_cloth(dataset_root, cloth_filename)
        if cloth_rgb is None:
            return False
        self._cloth_rgb = cloth_rgb
        self._cloth_mask = cloth_mask
        return True
    
    def process_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Process a single camera frame and return the composited result.
        
        Args:
            bgr_frame: Input BGR frame (H, W, 3), uint8.
            
        Returns:
            Composited BGR frame (H, W, 3), uint8. If no garment loaded
            or no person detected, returns the input frame unchanged.
        """
        t0 = time.time()
        self._frame_count += 1
        
        if self._cloth_rgb is None:
            return bgr_frame
        
        frame = cv2.flip(bgr_frame, 1)
        
        # 1. Detect body (includes smoothing + static lock)
        measurements = self._body_fitter.extract_body_measurements(frame)
        if measurements is None:
            self._total_time += time.time() - t0
            return frame
        
        # 2. Try multi-modal renderer
        if self._renderer is not None:
            try:
                return self._render_with_mode(frame, measurements)
            except Exception as e:
                if self._frame_count <= 1:
                    logger.warning(f"Renderer failed, using geometric: {e}")
        
        # 3. Fallback: geometric fitting
        result = self._body_fitter.fit_garment_to_body(
            frame, self._cloth_rgb, self._cloth_mask, measurements
        )
        
        self._total_time += time.time() - t0
        return result
    
    def set_render_mode(self, mode: str) -> bool:
        """
        Switch rendering mode at runtime.
        
        Args:
            mode: One of 'neural_warp', 'neural_densepose', 'hybrid_cached', 'cloud_api'
            
        Returns:
            True if mode changed successfully, False otherwise
        """
        from src.pipelines.diffusion_renderer import RenderMode
        
        try:
            new_mode = RenderMode(mode)
            self._render_mode = new_mode
            logger.info(f"Render mode changed to: {mode}")
            return True
        except ValueError:
            logger.error(f"Invalid render mode: {mode}")
            return False
    
    def _render_with_mode(self, frame: np.ndarray,
                          measurements: dict) -> np.ndarray:
        """Run multi-modal rendering pipeline."""
        h, w = frame.shape[:2]
        landmarks = measurements['landmarks']
        
        # Build MediaPipe dict for renderer
        mp_dict = {}
        for idx in range(len(landmarks)):
            lm = landmarks[idx]
            mp_dict[idx] = {'x': lm.x, 'y': lm.y, 'visibility': lm.visibility}
        
        # Render with selected mode
        render_result = self._renderer.render(
            frame, 
            self._cloth_rgb, 
            self._cloth_mask,
            mp_dict,
            mode=self._render_mode,
            body_mask=measurements.get('body_mask')
        )
        
        return render_result.output_image
    
    def _neural_composite(self, frame: np.ndarray,
                          measurements: dict) -> np.ndarray:
        """Run neural warping pipeline and composite onto frame."""
        h, w = frame.shape[:2]
        landmarks = measurements['landmarks']
        
        # Build MediaPipe dict for pipeline
        mp_dict = {}
        for idx in range(len(landmarks)):
            lm = landmarks[idx]
            mp_dict[idx] = {'x': lm.x, 'y': lm.y, 'visibility': lm.visibility}
        
        person_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        result = self._pipeline.warp_garment(
            person_rgb, self._cloth_rgb, self._cloth_mask,
            mp_dict, body_mask=measurements.get('body_mask'),
            use_densepose=self._use_densepose
        )
        
        # Resize warped cloth to frame resolution
        warped_cloth = cv2.resize(result.warped_cloth, (w, h), interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.resize(result.warped_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if warped_mask.ndim == 2:
            warped_mask = np.expand_dims(warped_mask, axis=-1)
        
        # Apply body mask
        body_mask = measurements.get('body_mask')
        if body_mask is not None:
            body_mask_f = body_mask.astype(np.float32)
            if body_mask_f.ndim == 2:
                body_mask_f = np.expand_dims(body_mask_f, axis=-1)
            warped_mask = warped_mask * body_mask_f
        
        # Alpha composite in RGB space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        composite = warped_cloth * warped_mask + frame_rgb * (1.0 - warped_mask)
        composite_bgr = cv2.cvtColor((composite * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        self._total_time += time.time() - time.time()  # tracked in process_frame
        return composite_bgr
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get all diagnostic metrics.
        
        Returns dict with keys: fps, landmark_stability, gmm_transform, gpd, pose.
        """
        fps = self._frame_count / self._total_time if self._total_time > 0 else 0
        
        result = {
            'fps': round(fps, 1),
            'frames_processed': self._frame_count,
            'pose': self._body_fitter.get_diagnostics(),
            'landmark_stability': self._body_fitter.landmark_logger.get_stats(),
        }
        
        if self._pipeline is not None:
            result['gmm_transform'] = self._pipeline.transform_logger.get_stats()
            result['gpd'] = self._pipeline.gpd_metric.get_stats()
        
        return result
    
    def reset(self):
        """Reset all state (smoother, loggers, pose lock)."""
        self._body_fitter.landmark_smoother.reset()
        self._body_fitter._consecutive_static = 0
        self._body_fitter._locked_measurements = None
        self._frame_count = 0
        self._total_time = 0.0

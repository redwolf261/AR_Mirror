"""
AR Mirror — Garment Rendering Logic

Extracted from app.py to reduce monolith size.
Contains all garment rendering methods (neural, geometric, alpha blending).
"""

import os
import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Import semantic parsing (optional)
SEMANTIC_PARSING_AVAILABLE = False
try:
    from src.core.semantic_parser import SemanticParser, create_occlusion_aware_composite
    SEMANTIC_PARSING_AVAILABLE = True
except Exception:
    pass

# Import CatVTON pre-warper (optional — degrades gracefully)
CATVTON_AVAILABLE = False
try:
    from src.core.catvton_prewarper import get_prewarper as _get_catvton_singleton
    CATVTON_AVAILABLE = True
except Exception:
    pass


def load_viton_cloth(dataset_root, cloth_filename, target_size=None):
    """
    KNOWN-GOOD VITON garment loader (CP-VTON / VITON-HD pattern)
    
    Returns:
        cloth_rgb : float32, shape (H, W, 3), range [0,1]
        cloth_mask: float32, shape (H, W, 1), range [0,1]
    """
    cloth_path = os.path.join(dataset_root, "cloth", cloth_filename)
    mask_path = os.path.join(dataset_root, "cloth-mask", cloth_filename)
    
    if not os.path.exists(cloth_path):
        return None, None
    if not os.path.exists(mask_path):
        return None, None
    
    # Load images
    cloth_bgr = cv2.imread(cloth_path, cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if cloth_bgr is None or mask_gray is None:
        return None, None
    
    # Convert BGR to RGB
    cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize if required
    if target_size is not None:
        cloth_rgb = cv2.resize(cloth_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        mask_gray = cv2.resize(mask_gray, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Normalize to float32 [0,1]
    cloth_rgb = cloth_rgb.astype(np.float32) / 255.0
    cloth_mask = (mask_gray > 127).astype(np.float32)
    cloth_mask = np.expand_dims(cloth_mask, axis=-1)
    
    # Validate
    assert cloth_rgb.ndim == 3 and cloth_rgb.shape[2] == 3
    assert cloth_mask.ndim == 3 and cloth_mask.shape[2] == 1
    assert cloth_rgb.dtype == np.float32 and cloth_mask.dtype == np.float32
    assert 0.0 <= cloth_rgb.min() and cloth_rgb.max() <= 1.0
    
    return cloth_rgb, cloth_mask


class GarmentRenderer:
    """
    Mixin providing garment rendering methods for ARMirrorApp.
    
    Expects the host class to provide:
      - self.phase, self.phase2_pipeline, self.gmm_warper
      - self.body_fitter, self.semantic_parser
      - self.show_debug, self.dataset_pairs, self.current_garment_idx
      - self.garments, self._garment_cache
      - self._cached_body_parts, self._semantic_frame_counter, self._semantic_skip_interval
      - self._cached_body_measurements, self._pose_frame_counter, self._pose_skip_interval
      - self._last_body_measurements
      - self._stage_times, self._warp_sub_times
    """

    # ----------------------------------------------------------------
    # CatVTON helpers
    # ----------------------------------------------------------------

    def _catvton_prewarper(self):
        """Lazily return (or create) the CatVTON pre-warper singleton."""
        if not CATVTON_AVAILABLE:
            return None
        if not hasattr(self, '_catvton_pw'):
            try:
                import torch
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._catvton_pw = _get_catvton_singleton(device=dev)
            except Exception as exc:
                logger.warning(f"CatVTON init failed: {exc}")
                self._catvton_pw = None
        return getattr(self, '_catvton_pw', None)

    def _render_garment_catvton(
        self,
        frame: np.ndarray,
        garment_bgr: np.ndarray,
        body_measurements: dict,
    ) -> Optional[np.ndarray]:
        """
        Composite the CatVTON pre-warped garment onto *frame* using the same
        body-conforming silhouette alpha applied in _render_garment_neural.
        Returns None when the pre-warper is unavailable or triggered an error.
        """
        pw = self._catvton_prewarper()
        if pw is None or not pw.is_available:
            return None

        h, w = frame.shape[:2]

        # ---- trigger or reuse pre-warp ----
        if pw.needs_rewarp(frame, garment_bgr):
            # Run in a background thread so we don't stall the render loop.
            # On this frame we fall through and let GMM handle it; the next
            # frame will pick up the fresh CatVTON result.
            import threading
            t = threading.Thread(
                target=pw.prewarp,
                args=(frame.copy(), garment_bgr.copy()),
                daemon=True,
            )
            t.start()
            return None  # fall through to GMM this frame

        warped_rgb  = pw.cached_result   # float32 [0,1] RGB, output_size
        warped_mask = pw.cached_mask     # float32 [0,1]
        if warped_rgb is None or warped_mask is None:
            return None

        # ---- body-conforming placement (same logic as _render_garment_neural) ----
        torso_x1, torso_y1, torso_x2, torso_y2 = body_measurements['torso_box']
        shoulder_width = body_measurements['shoulder_width']
        torso_height   = body_measurements['torso_height']
        body_mask      = body_measurements.get('body_mask')

        if body_mask is not None and np.any(body_mask[torso_y1:torso_y2, torso_x1:torso_x2]):
            torso_bm = body_mask[torso_y1:torso_y2, torso_x1:torso_x2]
            rows_any = np.any(torso_bm, axis=1)
            cols_any = np.any(torso_bm, axis=0)
            rmin, rmax = int(np.where(rows_any)[0][0]),  int(np.where(rows_any)[0][-1])
            cmin, cmax = int(np.where(cols_any)[0][0]),  int(np.where(cols_any)[0][-1])
            h_margin = max(10, int((cmax - cmin) * 0.08))
            v_margin = max(5,  int((rmax - rmin) * 0.05))
            gx1 = max(0, torso_x1 + cmin - h_margin)
            gy1 = max(0, torso_y1 + rmin - v_margin)
            gx2 = min(w, torso_x1 + cmax + h_margin)
            gy2 = min(h, torso_y1 + rmax + v_margin)
        else:
            tw = int(shoulder_width * 1.1)
            th = int(torso_height   * 1.2)
            cx = (torso_x1 + torso_x2) // 2
            gx1 = max(0, cx - tw // 2)
            gy1 = max(0, torso_y1)
            gx2 = min(w, gx1 + tw)
            gy2 = min(h, gy1 + th)

        aw, ah = gx2 - gx1, gy2 - gy1
        if aw < 2 or ah < 2:
            return None

        # Resize CatVTON output to placement region
        warped_cloth_rs = cv2.resize(
            (warped_rgb * 255).astype(np.uint8), (aw, ah),
            interpolation=cv2.INTER_LINEAR
        )
        warped_cloth_f32 = warped_cloth_rs.astype(np.float32) / 255.0
        warped_mask_rs = cv2.resize(warped_mask, (aw, ah), interpolation=cv2.INTER_LINEAR)

        # Silhouette-feathered alpha
        if body_mask is not None:
            bm_roi = body_mask[gy1:gy2, gx1:gx2].astype(np.float32)
            _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bm_roi = cv2.dilate(bm_roi, _k, iterations=1)
            bm_roi = cv2.GaussianBlur(bm_roi, (13, 13), sigmaX=5, sigmaY=5)
            bm_roi = np.clip(bm_roi, 0.0, 1.0)
            alpha_3d = np.expand_dims(warped_mask_rs * bm_roi, axis=-1)
        else:
            alpha_3d = np.expand_dims(warped_mask_rs, axis=-1)

        frame_roi     = frame[gy1:gy2, gx1:gx2]
        frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Semantic occlusion compositing if available
        if self.semantic_parser is not None and SEMANTIC_PARSING_AVAILABLE:
            try:
                body_parts = self.semantic_parser.parse(
                    frame_roi, person_mask=None, target_resolution=(256, 192)
                )
                composite_bgr = create_occlusion_aware_composite(
                    frame_roi, warped_cloth_f32,
                    alpha_3d.squeeze(), body_parts, collar_constraint=True
                )
            except Exception:
                comp = warped_cloth_f32 * alpha_3d + frame_roi_rgb * (1 - alpha_3d)
                composite_bgr = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            comp = warped_cloth_f32 * alpha_3d + frame_roi_rgb * (1 - alpha_3d)
            composite_bgr = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        output = frame.copy()
        output[gy1:gy2, gx1:gx2] = composite_bgr
        return output

    def _convert_landmarks_to_dict(self, landmarks) -> dict:
        """Convert MediaPipe NormalizedLandmark list to dict format for Phase 2 pipeline."""
        mp_dict = {}
        for idx, lm in enumerate(landmarks):
            mp_dict[idx] = {
                'x': lm.x,
                'y': lm.y,
                'visibility': lm.visibility
            }
        return mp_dict

    def _load_garment_image(self, garment_filename):
        """Load VITON garment + mask with caching (avoids per-frame disk I/O)."""
        if garment_filename in self._garment_cache:
            return self._garment_cache[garment_filename]
        
        cloth_rgb, cloth_mask = load_viton_cloth(
            "dataset/train",  
            garment_filename, 
            target_size=(512, 384)
        )
        
        self._garment_cache[garment_filename] = (cloth_rgb, cloth_mask)
        return cloth_rgb, cloth_mask

    def _render_garment_neural(self, frame, cloth_rgb, cloth_mask, body_measurements):
        """Render garment using Phase 2 neural warping (GMM + optional TOM).
        
        Converts landmarks, runs neural pipeline, composites with occlusion handling.
        Returns None on failure (caller should fall back to geometric fitting).
        """
        _neural_t = {}
        h, w = frame.shape[:2]
        
        try:
            # Convert MediaPipe landmarks to dict format
            landmarks = body_measurements['landmarks']
            mp_landmarks_dict = self._convert_landmarks_to_dict(landmarks)
            
            # Convert frame to RGB float32 for pipeline (downscaled)
            t0 = time.perf_counter()
            small_frame = cv2.resize(frame, (192, 256), interpolation=cv2.INTER_LINEAR)
            person_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            _neural_t['rgb_convert'] = time.perf_counter() - t0
            
            # Ensure cloth_mask is 2D for pipeline
            cloth_mask_2d = cloth_mask.squeeze() if cloth_mask.ndim == 3 else cloth_mask
            
            # Pass pre-computed body mask to avoid double segmentation
            body_mask_precomputed = body_measurements.get('body_mask')
            
            # Run neural warping pipeline
            t_warp = time.perf_counter()
            result = self.phase2_pipeline.warp_garment(
                person_rgb, cloth_rgb, cloth_mask_2d, mp_landmarks_dict,
                body_mask=body_mask_precomputed
            )
            _neural_t['warp_garment'] = time.perf_counter() - t_warp
            # Cache depth_proxy for session logger (app.py reads self._last_depth_proxy)
            self._last_depth_proxy: float = getattr(result, 'depth_proxy', 0.0)
            # Log sub-timings from pipeline for first 30 frames
            if result.timings:
                for k, v in result.timings.items():
                    self._warp_sub_times.setdefault(k, []).append(v)
            
            if result.warped_cloth is None:
                return None
            
            # Get torso region from body measurements for placement
            torso_x1, torso_y1, torso_x2, torso_y2 = body_measurements['torso_box']
            shoulder_width = body_measurements['shoulder_width']
            torso_height = body_measurements['torso_height']
            body_mask = body_measurements.get('body_mask')
            
            # ----------------------------------------------------------------
            # BODY-CONFORMING PLACEMENT
            # Use body_mask silhouette bounds instead of a fixed
            # shoulder_width × torso_height rectangle.  This lets the garment
            # follow the actual body contour — including the sides — rather
            # than being clipped to a flat centred rectangle.
            # ----------------------------------------------------------------
            if body_mask is not None and np.any(body_mask[torso_y1:torso_y2, torso_x1:torso_x2]):
                torso_bm = body_mask[torso_y1:torso_y2, torso_x1:torso_x2]
                rows_any = np.any(torso_bm, axis=1)
                cols_any = np.any(torso_bm, axis=0)
                rmin, rmax = np.where(rows_any)[0][[0, -1]]
                cmin, cmax = np.where(cols_any)[0][[0, -1]]
                # Add horizontal margin so sides aren't clipped
                h_margin = max(10, int((cmax - cmin) * 0.08))
                v_margin = max(5, int((rmax - rmin) * 0.05))
                garment_x1 = max(0, torso_x1 + cmin - h_margin)
                garment_y1 = max(0, torso_y1 + rmin - v_margin)
                garment_x2 = min(w, torso_x1 + cmax + h_margin)
                garment_y2 = min(h, torso_y1 + rmax + v_margin)
            else:
                # Fallback: centred rectangle (original behaviour)
                target_width  = int(shoulder_width * 1.1)
                target_height = int(torso_height   * 1.2)
                torso_center_x = (torso_x1 + torso_x2) // 2
                garment_x1 = max(0, torso_center_x - target_width  // 2)
                garment_y1 = max(0, torso_y1)
                garment_x2 = min(w, garment_x1 + target_width)
                garment_y2 = min(h, garment_y1 + target_height)

            actual_width  = garment_x2 - garment_x1
            actual_height = garment_y2 - garment_y1
            if actual_width < 2 or actual_height < 2:
                return None

            warped_cloth_final = cv2.resize(
                result.warped_cloth, (actual_width, actual_height),
                interpolation=cv2.INTER_LINEAR
            )
            warped_mask_raw = cv2.resize(
                result.warped_mask.squeeze() if result.warped_mask.ndim == 3 else result.warped_mask,
                (actual_width, actual_height),
                interpolation=cv2.INTER_LINEAR
            )

            # ----------------------------------------------------------------
            # SILHOUETTE-CONFORMING ALPHA
            # Dilate the body mask slightly to recover side-torso pixels that
            # MediaPipe may have missed, then feather the edge so the garment
            # blends smoothly into the background at the body silhouette.
            # ----------------------------------------------------------------
            if body_mask is not None:
                body_mask_roi = body_mask[garment_y1:garment_y2,
                                          garment_x1:garment_x2].astype(np.float32)
                # Dilate ~6 px to capture side pixels
                _dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                body_mask_roi = cv2.dilate(body_mask_roi, _dil_k, iterations=1)
                # Feather edges — smooth alpha at silhouette boundary
                body_mask_roi = cv2.GaussianBlur(
                    body_mask_roi, (13, 13), sigmaX=5, sigmaY=5
                )
                body_mask_roi = np.clip(body_mask_roi, 0.0, 1.0)
                composite_alpha = (warped_mask_raw * body_mask_roi)
            else:
                composite_alpha = warped_mask_raw.copy()

            # Ensure 3-channel alpha for broadcasting
            warped_mask_3d = np.expand_dims(composite_alpha, axis=-1)

            # Extract frame region
            frame_roi = frame[garment_y1:garment_y2, garment_x1:garment_x2]
            if frame_roi.size == 0:
                return None
            
            # Semantic occlusion compositing (hair/face on top of garment)
            # Skip-frame: run SCHP-LIP every N frames, reuse cached masks
            if self.semantic_parser is not None and SEMANTIC_PARSING_AVAILABLE:
                try:
                    t_sem = time.perf_counter()
                    self._semantic_frame_counter += 1
                    if (self._cached_body_parts is None or 
                        self._semantic_frame_counter >= self._semantic_skip_interval):
                        body_parts = self.semantic_parser.parse(
                            frame_roi, person_mask=None, target_resolution=(256, 192)
                        )
                        self._cached_body_parts = body_parts
                        self._semantic_frame_counter = 0
                    else:
                        # Reuse cached semantic masks (resize if ROI changed)
                        body_parts = {}
                        for k, v in self._cached_body_parts.items():
                            if v.shape[:2] != (frame_roi.shape[0], frame_roi.shape[1]):
                                body_parts[k] = cv2.resize(v, (frame_roi.shape[1], frame_roi.shape[0]),
                                                           interpolation=cv2.INTER_NEAREST)
                            else:
                                body_parts[k] = v
                    self._stage_times['semantic_parse'].append(time.perf_counter() - t_sem)
                    t_comp = time.perf_counter()
                    composite_bgr = create_occlusion_aware_composite(
                        frame_roi, warped_cloth_final,
                        warped_mask_3d.squeeze() if warped_mask_3d.ndim == 3 else warped_mask_3d,
                        body_parts, collar_constraint=True
                    )
                    self._stage_times['composite'].append(time.perf_counter() - t_comp)
                except Exception as e:
                    logger.debug(f"Semantic compositing failed: {e}")
                    frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    composite = warped_cloth_final * warped_mask_3d + frame_roi_rgb * (1 - warped_mask_3d)
                    composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                composite = warped_cloth_final * warped_mask_3d + frame_roi_rgb * (1 - warped_mask_3d)
                composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            output = frame.copy()
            output[garment_y1:garment_y2, garment_x1:garment_x2] = composite_bgr

            # ----------------------------------------------------------------
            # TEMPORAL SMOOTHING
            # Blend with previous composite at garment region to suppress
            # frame-to-frame jitter without adding optical flow overhead.
            # ----------------------------------------------------------------
            _prev = getattr(self, '_prev_garment_composite', None)
            _prev_roi_key = getattr(self, '_prev_garment_roi', None)
            _curr_roi_key = (garment_y1, garment_y2, garment_x1, garment_x2)
            if (_prev is not None and _prev_roi_key == _curr_roi_key
                    and _prev.shape == composite_bgr.shape):
                # 80 % new + 20 % previous — reduces jitter while staying responsive
                smoothed = cv2.addWeighted(composite_bgr, 0.80, _prev, 0.20, 0)
                output[garment_y1:garment_y2, garment_x1:garment_x2] = smoothed
            self._prev_garment_composite = composite_bgr.copy()
            self._prev_garment_roi = _curr_roi_key

            # Accumulate neural sub-timings
            for k, v in _neural_t.items():
                self._warp_sub_times.setdefault(f'n_{k}', []).append(v)

            return output
            
        except Exception as e:
            logger.error(f"Neural warping failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _blend_garment_image_gmm(self, frame, cloth_rgb, cloth_mask, pose_json_path):
        """
        GMM TPS warping version - replaces resize-paste with geometric deformation.
        """
        if cloth_rgb is None or cloth_mask is None or self.gmm_warper is None:
            return self._blend_garment_image(frame, cloth_rgb, cloth_mask)
        
        h, w = frame.shape[:2]
        
        try:
            from convert_pose_map import load_openpose_json, get_pose_map
            from gmm_warper import build_agnostic_representation
            
            if os.path.exists(pose_json_path):
                keypoints = load_openpose_json(pose_json_path)
            else:
                first_pose_path = "dataset/train/openpose_json/000001_00_keypoints.json"
                if os.path.exists(first_pose_path):
                    keypoints = load_openpose_json(first_pose_path)
                else:
                    logger.warning("No pose data available - falling back to alpha blend")
                    return self._blend_garment_image(frame, cloth_rgb, cloth_mask)
            
            pose_map = get_pose_map(keypoints, height=256, width=192)
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_roi = frame_gray[int(h*0.1):int(h*0.75), int(w*0.15):int(w*0.85)]
            
            _, body_mask = cv2.threshold(frame_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            body_mask_resized = cv2.resize(body_mask, (192, 256), interpolation=cv2.INTER_NEAREST)
            person_shape = body_mask_resized.astype(np.float32) / 255.0
            
            person_head = np.zeros((256, 192), dtype=np.float32)
            person_head[:77] = person_shape[:77]
            
            agnostic = build_agnostic_representation(pose_map, person_shape, person_head)
            
            warped_cloth, warped_mask = self.gmm_warper.warp_cloth(
                cloth_rgb, cloth_mask, pose_map, agnostic
            )
            
            roi_x1 = int(w * 0.15)
            roi_y1 = int(h * 0.1)
            roi_x2 = int(w * 0.85)
            roi_y2 = int(h * 0.75)
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            
            warped_cloth_resized = cv2.resize(warped_cloth, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            warped_mask_resized = cv2.resize(warped_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            
            if warped_mask_resized.ndim == 2:
                warped_mask_resized = np.expand_dims(warped_mask_resized, axis=-1)
            
            frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            use_semantic_parsing = True
            if use_semantic_parsing and self.semantic_parser is not None and SEMANTIC_PARSING_AVAILABLE:
                try:
                    body_parts = self.semantic_parser.parse(
                        frame_roi, person_mask=None, target_resolution=(256, 192)
                    )
                    composite_bgr = create_occlusion_aware_composite(
                        frame_roi, warped_cloth_resized,
                        warped_mask_resized.squeeze() if warped_mask_resized.ndim == 3 else warped_mask_resized,
                        body_parts, collar_constraint=True
                    )
                except Exception as e:
                    logger.warning(f"Semantic parsing failed, using simple blend: {e}")
                    composite = warped_cloth_resized * warped_mask_resized + frame_roi_rgb * (1 - warped_mask_resized)
                    composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                composite = warped_cloth_resized * warped_mask_resized + frame_roi_rgb * (1 - warped_mask_resized)
                composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = composite_bgr
            return frame
        
        except Exception as e:
            logger.error(f"GMM warping failed: {e}")
            return self._blend_garment_image(frame, cloth_rgb, cloth_mask)

    def _blend_garment_image(self, frame, cloth_rgb, cloth_mask):
        """VITON alpha composite with intelligent region-of-interest optimization."""
        if cloth_rgb is None or cloth_mask is None:
            return self._render_colored_shirt(frame, {})
        
        h, w = frame.shape[:2]
        
        try:
            roi_x1 = int(w * 0.15)
            roi_y1 = int(h * 0.1)
            roi_x2 = int(w * 0.85)
            roi_y2 = int(h * 0.75)
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            
            cloth_resized = cv2.resize(cloth_rgb, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(cloth_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            
            if mask_resized.ndim == 2:
                mask_resized = np.expand_dims(mask_resized, axis=-1)
            
            frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            use_semantic_parsing = True
            if use_semantic_parsing and self.semantic_parser is not None and SEMANTIC_PARSING_AVAILABLE:
                try:
                    body_parts = self.semantic_parser.parse(
                        frame_roi, person_mask=None, target_resolution=(256, 192)
                    )
                    composite_bgr = create_occlusion_aware_composite(
                        frame_roi, cloth_resized,
                        mask_resized.squeeze() if mask_resized.ndim == 3 else mask_resized,
                        body_parts, collar_constraint=True
                    )
                except Exception as e:
                    logger.warning(f"Semantic parsing failed, using simple blend: {e}")
                    composite = cloth_resized * mask_resized + frame_roi_rgb * (1 - mask_resized)
                    composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                composite = cloth_resized * mask_resized + frame_roi_rgb * (1 - mask_resized)
                composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = composite_bgr
            return frame
        
        except Exception as e:
            logger.error(f"VITON blending failed: {e}")
            return self._render_colored_shirt(frame, {})

    def _render_garment(self, frame, garment):
        """Render real dataset garment image on the body using VITON loader."""
        t_render_start = time.perf_counter()
        
        if self.dataset_pairs:
            idx = self.current_garment_idx % len(self.dataset_pairs)
            pair = self.dataset_pairs[idx]
            cloth_rgb, cloth_mask = self._load_garment_image(pair['garment'])
            
            if cloth_rgb is not None and cloth_mask is not None:
                # Body-aware fitting (all phases use body detection)
                if self.body_fitter:
                    try:
                        self._pose_frame_counter += 1
                        t_pose = time.perf_counter()
                        if (self._cached_body_measurements is None or 
                            self._pose_frame_counter >= self._pose_skip_interval):
                            body_measurements = self.body_fitter.extract_body_measurements(frame)
                            self._cached_body_measurements = body_measurements
                            self._pose_frame_counter = 0
                        else:
                            body_measurements = self._cached_body_measurements
                        self._stage_times['pose_detect'].append(time.perf_counter() - t_pose)
                        self._last_body_measurements = body_measurements
                        
                        if body_measurements:
                            # CatVTON path (best quality, runs offline in background)
                            # Uses cached result when available; triggers rewarp thread otherwise.
                            _cloth_bgr = (
                                cv2.cvtColor((cloth_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                                if cloth_rgb is not None
                                else np.zeros((10, 10, 3), dtype=np.uint8)
                            )
                            catvton_result = self._render_garment_catvton(
                                frame, _cloth_bgr, body_measurements,
                            )
                            if catvton_result is not None:
                                self._stage_times['total_render'].append(
                                    time.perf_counter() - t_render_start)
                                return catvton_result

                            # Phase 2: Neural warping with GMM + TOM
                            if self.phase == 2 and self.phase2_pipeline:
                                t_neural = time.perf_counter()
                                result = self._render_garment_neural(
                                    frame, cloth_rgb, cloth_mask, body_measurements
                                )
                                self._stage_times['neural_warp'].append(time.perf_counter() - t_neural)
                                if result is not None:
                                    self._stage_times['total_render'].append(time.perf_counter() - t_render_start)
                                    if self.show_debug:
                                        result = self.body_fitter.draw_debug_overlay(
                                            result, body_measurements,
                                            show_box=True, show_measurements=True, show_skeleton=True
                                        )
                                    return result
                            
                            # Geometric fitting (Phase 0/1 or Phase 2 fallback)
                            result = self.body_fitter.fit_garment_to_body(
                                frame, cloth_rgb, cloth_mask, body_measurements
                            )
                            if self.show_debug:
                                result = self.body_fitter.draw_debug_overlay(
                                    result, body_measurements,
                                    show_box=True, show_measurements=True, show_skeleton=True
                                )
                            return result
                        else:
                            result = self._blend_garment_image(frame, cloth_rgb, cloth_mask)
                            if self.show_debug:
                                result = self.body_fitter.draw_debug_overlay(result)
                            return result
                    except Exception as e:
                        logger.error(f"Body-aware fitting failed: {e}")
                        return self._blend_garment_image(frame, cloth_rgb, cloth_mask)
                
                # No body fitter: Phase 1 GMM or simple blending
                if self.phase == 1 and self.gmm_warper:
                    pose_json_path = os.path.join(
                        "dataset/train/openpose_json",
                        pair['person'].replace('.jpg', '_keypoints.json')
                    )
                    return self._blend_garment_image_gmm(frame, cloth_rgb, cloth_mask, pose_json_path)
                else:
                    return self._blend_garment_image(frame, cloth_rgb, cloth_mask)
        
        # Fallback to colored shirt if dataset image not available
        return self._render_colored_shirt(frame, garment)

    def _render_colored_shirt(self, frame, garment):
        """Fallback: Render colored shirt overlay."""
        h, w = frame.shape[:2]
        color = garment.get('color', (200, 200, 200))
        
        center_x = w // 2
        shoulder_top = int(h * 0.2)
        chest_top = int(h * 0.3)
        waist = int(h * 0.65)
        hem = int(h * 0.8)
        
        shoulder_width = int(w * 0.5)
        chest_width = int(w * 0.45)
        waist_width = int(w * 0.4)
        
        neck_radius = int(w * 0.08)
        cv2.ellipse(frame, (center_x, chest_top), (neck_radius, int(neck_radius * 1.2)), 
                   0, 0, 360, color, -1)
        
        shirt_body = np.array([
            [center_x - shoulder_width // 2, shoulder_top],
            [center_x + shoulder_width // 2, shoulder_top],
            [center_x + chest_width // 2, chest_top],
            [center_x + waist_width // 2, waist],
            [center_x + waist_width // 2 - 10, hem],
            [center_x - waist_width // 2 + 10, hem],
            [center_x - waist_width // 2, waist],
            [center_x - chest_width // 2, chest_top]
        ], np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [shirt_body], color)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.polylines(frame, [shirt_body], True, color, 2)
        
        sleeve_length = int(h * 0.25)
        left_sleeve = np.array([
            [center_x - shoulder_width // 2, shoulder_top],
            [center_x - shoulder_width // 2 - 40, shoulder_top],
            [center_x - shoulder_width // 2 - 50, shoulder_top + sleeve_length],
            [center_x - shoulder_width // 2 - 20, shoulder_top + sleeve_length],
            [center_x - chest_width // 2, chest_top]
        ], np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [left_sleeve], color)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.polylines(frame, [left_sleeve], True, color, 2)
        
        right_sleeve = np.array([
            [center_x + shoulder_width // 2, shoulder_top],
            [center_x + shoulder_width // 2 + 40, shoulder_top],
            [center_x + shoulder_width // 2 + 50, shoulder_top + sleeve_length],
            [center_x + shoulder_width // 2 + 20, shoulder_top + sleeve_length],
            [center_x + chest_width // 2, chest_top]
        ], np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [right_sleeve], color)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.polylines(frame, [right_sleeve], True, color, 2)
        
        cv2.line(frame, (center_x, chest_top + neck_radius), (center_x, hem), 
                (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)), 1)
        
        return frame

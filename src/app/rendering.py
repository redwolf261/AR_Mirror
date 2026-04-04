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
from pathlib import Path

logger = logging.getLogger(__name__)

# CatVTON and SemanticParser removed — prewarper disabled
CATVTON_AVAILABLE = False


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

    # ── ESRGAN texture upscaling (Phase A) ────────────────────────────────
    # Up-sample the garment texture 4× before any resize/normalise so the
    # GPU renderer receives fine fabric detail instead of interpolated pixels.
    try:
        from src.core.esrgan_upscaler import get_upscaler as _get_upscaler
        _upscaler = _get_upscaler()
        if _upscaler.available:
            _cloth_f32 = cloth_rgb.astype(np.float32) / 255.0
            _cloth_f32 = _upscaler.upscale(_cloth_f32)          # (H*4, W*4, 3) float32
            cloth_rgb = (_cloth_f32 * 255.0).clip(0, 255).astype(np.uint8)
    except Exception:
        pass  # Non-fatal: keep original-resolution texture
    # ──────────────────────────────────────────────────────────────────

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

        # Direct alpha composite
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

    def _load_garment_image(self, garment_ref):
        """Load real garment assets first, then fallback to VITON template cloth."""
        cache_key = garment_ref.get("sku") if isinstance(garment_ref, dict) else str(garment_ref)
        if cache_key in self._garment_cache:
            return self._garment_cache[cache_key]

        def _derive_mask(image_bgr: np.ndarray) -> np.ndarray:
            if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
                alpha = image_bgr[:, :, 3]
                return (alpha > 10).astype(np.float32)
            rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
            luminance = rgb.mean(axis=2)
            mask = (luminance > 12).astype(np.float32)
            mask = cv2.medianBlur((mask * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
            return mask

        def _pick_real_asset_path(ref: dict) -> Optional[Path]:
            image_root = ref.get("image_path")
            if not image_root:
                return None
            root = Path(image_root)
            if not root.is_absolute():
                root = Path(image_root)
            if root.is_file():
                return root
            if root.is_dir():
                for candidate in [
                    root / "image.png",
                    root / "image.jpg",
                    root / "image.jpeg",
                    root / "14274_00.jpg",
                    root / "model.glb",
                    root / "model.gltf",
                    root / "front_medium.png",
                    root / "front_large.png",
                    root / "front_small.png",
                    root / "front_xlarge.png",
                ]:
                    if candidate.exists():
                        return candidate
                pngs = sorted(root.glob("*.png"))
                jpegs = sorted(root.glob("*.jpg"))
                glbs = sorted(root.glob("*.glb"))
                gltfs = sorted(root.glob("*.gltf"))
                if pngs:
                    return pngs[0]
                if jpegs:
                    return jpegs[0]
                if glbs:
                    return glbs[0]
                if gltfs:
                    return gltfs[0]
            return None

        def _load_real_asset(path: Path):
            def _normalize_rgb_mask(bgr: np.ndarray, mask: np.ndarray):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mask = np.expand_dims(mask.astype(np.float32), axis=-1)
                target_size = (512, 384)
                rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, axis=-1)
                return rgb, mask

            def _load_mesh_asset(mesh_path: Path):
                try:
                    import trimesh
                except Exception:
                    logger.warning("GLB/GLTF asset found but trimesh is not installed: %s", mesh_path)
                    return None, None

                scene = None

                def _from_material_texture(scene):
                    """Fallback path: extract embedded material texture without OpenGL snapshot."""
                    try:
                        for geom in scene.geometry.values():
                            visual = getattr(geom, "visual", None)
                            material = getattr(visual, "material", None)
                            if material is None:
                                continue
                            image = getattr(material, "image", None)
                            if image is None:
                                continue

                            if hasattr(image, "convert"):
                                rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
                            else:
                                arr = np.asarray(image)
                                if arr.ndim == 2:
                                    rgba = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGBA)
                                elif arr.ndim == 3 and arr.shape[2] == 3:
                                    alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
                                    rgba = np.concatenate([arr.astype(np.uint8), alpha], axis=2)
                                elif arr.ndim == 3 and arr.shape[2] >= 4:
                                    rgba = arr[:, :, :4].astype(np.uint8)
                                else:
                                    continue

                            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                            mask = (rgba[:, :, 3] > 10).astype(np.float32)
                            if float(mask.mean()) < 0.01:
                                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                                mask = (gray > 8).astype(np.float32)
                            mask = cv2.GaussianBlur(mask, (5, 5), 0)
                            return _normalize_rgb_mask(bgr, mask)
                    except Exception as exc:
                        logger.warning("Failed material-texture extraction for %s: %s", mesh_path, exc)
                    return None, None

                try:
                    scene_or_mesh = trimesh.load(str(mesh_path), force='scene')
                    scene = scene_or_mesh if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh.scene()

                    # Front-view snapshot from the 3D model. Returned as PNG bytes.
                    png_bytes = scene.save_image(resolution=(1024, 1024), visible=True)
                    if png_bytes:
                        raw = np.frombuffer(png_bytes, dtype=np.uint8)
                        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
                        if decoded is not None:
                            if decoded.ndim == 3 and decoded.shape[2] == 4:
                                alpha = decoded[:, :, 3]
                                bgr = decoded[:, :, :3]
                                mask = (alpha > 10).astype(np.float32)
                            else:
                                bgr = decoded[:, :, :3]
                                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                                mask = (gray > 8).astype(np.float32)
                            mask = cv2.GaussianBlur(mask, (5, 5), 0)
                            return _normalize_rgb_mask(bgr, mask)
                except Exception as exc:
                    logger.warning("Could not rasterize GLB/GLTF asset %s: %s", mesh_path, exc)

                if scene is not None:
                    return _from_material_texture(scene)
                return None, None

            if path.suffix.lower() in {".glb", ".gltf"}:
                return _load_mesh_asset(path)

            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                bgr = img[:, :, :3]
                mask = (alpha > 10).astype(np.float32)
            else:
                bgr = img[:, :, :3]
                # Estimate matte by removing the border/background color.
                # This works better for product shots on white/black studio backdrops.
                h, w = bgr.shape[:2]
                border = np.concatenate([
                    bgr[: max(1, h // 20), :, :].reshape(-1, 3),
                    bgr[-max(1, h // 20):, :, :].reshape(-1, 3),
                    bgr[:, : max(1, w // 20), :].reshape(-1, 3),
                    bgr[:, -max(1, w // 20):, :].reshape(-1, 3),
                ], axis=0).astype(np.float32)
                bg_color = np.median(border, axis=0)
                dist = np.linalg.norm(bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
                # Adaptive threshold: keep pixels that differ meaningfully from the border.
                thr = max(18.0, float(np.percentile(dist, 70) * 0.45))
                mask = (dist > thr).astype(np.float32)
                mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
            return _normalize_rgb_mask(bgr, mask)

        garment_dict = garment_ref if isinstance(garment_ref, dict) else None
        real_asset_path = _pick_real_asset_path(garment_dict) if garment_dict else None
        if real_asset_path is not None:
            cloth_rgb, cloth_mask = _load_real_asset(real_asset_path)
            if cloth_rgb is not None and cloth_mask is not None:
                self._garment_cache[cache_key] = (cloth_rgb, cloth_mask)
                return cloth_rgb, cloth_mask

        garment_filename = None
        if isinstance(garment_ref, dict):
            garment_filename = garment_ref.get("file") or garment_ref.get("name") or garment_ref.get("sku")
        else:
            garment_filename = str(garment_ref)

        cloth_rgb, cloth_mask = load_viton_cloth(
            "dataset/train",
            garment_filename,
            target_size=(512, 384)
        )

        self._garment_cache[cache_key] = (cloth_rgb, cloth_mask)
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
            
            # ----------------------------------------------------------------
            # ENVIRONMENT-MATCHED AMBIENT
            # Sample avg background color from four corner patches in the frame
            # so the garment's ambient term matches the scene illumination.
            # ----------------------------------------------------------------
            if hasattr(self, 'phase2_pipeline') and hasattr(self.phase2_pipeline, 'gpu_renderer'):
                try:
                    ph, pw = frame.shape[:2]
                    patch = 32
                    corners = [
                        frame[:patch,  :patch],
                        frame[:patch,  pw - patch:],
                        frame[ph - patch:, :patch],
                        frame[ph - patch:, pw - patch:],
                    ]
                    bg_mean = np.concatenate(
                        [c.reshape(-1, 3) for c in corners], axis=0
                    ).mean(axis=0)           # BGR uint8
                    # Convert to [0,1] float, swap to RGB
                    amb = np.clip(bg_mean[::-1] / 510.0, 0.04, 0.20).astype(np.float32)
                    self.phase2_pipeline.gpu_renderer.set_lighting(ambient=amb)
                except Exception:
                    pass  # Non-fatal — keep default ambient

            # Run neural warping pipeline
            t_warp = time.perf_counter()
            result = self.phase2_pipeline.warp_garment(
                person_rgb, cloth_rgb, cloth_mask_2d, mp_landmarks_dict,
                body_mask=body_mask_precomputed,
                hand_lm_left=body_measurements.get('hand_lm_left'),
                hand_lm_right=body_measurements.get('hand_lm_right'),
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
            body_mask = body_measurements.get('body_mask')

            # Phase B: merge MediaPipe body mask with SMPL 24-part segmenter mask.
            # Union (np.maximum) so neither segmenter's missed pixels clip the garment.
            _smpl_mask = getattr(result, 'smpl_body_mask', None)
            if _smpl_mask is not None:
                if body_mask is None:
                    body_mask = _smpl_mask
                else:
                    _bm = body_mask.astype(np.float32)
                    if _bm.shape[:2] != _smpl_mask.shape[:2]:
                        _bm = cv2.resize(_bm, (_smpl_mask.shape[1], _smpl_mask.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
                    body_mask = np.maximum(_bm, _smpl_mask.astype(np.float32))

            # ----------------------------------------------------------------
            # ROBUST FULL-TORSO PLACEMENT
            # The torso_box from MediaPipe landmarks (shoulder-to-hip) is the
            # ground-truth bounding region.  Pad it generously so sleeves and
            # collar are never clipped.
            #
            # WHY NOT USE warped_mask as alpha:
            #   GMM is trained on white-background VITON full-body portraits.
            #   For live camera input the grid produces near-zero alpha values,
            #   making the shirt appear as a tiny patch.  We instead use the
            #   ORIGINAL cloth_mask (correct garment silhouette) clipped by
            #   the body silhouette from MediaPipe.
            # ----------------------------------------------------------------
            raw_w     = max(1, torso_x2 - torso_x1)   # shoulder span (px)
            raw_h     = max(1, torso_y2 - torso_y1)   # shoulder-to-hip span (px)
            pad_x     = max(16, int(raw_w * 0.18))    # 18 % side extension
            pad_y_top = max(8,  int(raw_h * 0.08))    # collar clearance above shoulder
            pad_y_bot = max(4,  int(raw_h * 0.04))    # hem clearance below hip
            garment_x1 = max(0, torso_x1 - pad_x)
            garment_y1 = max(0, torso_y1 - pad_y_top)
            garment_x2 = min(w, torso_x2 + pad_x)
            garment_y2 = min(h, torso_y2 + pad_y_bot)

            actual_width  = garment_x2 - garment_x1
            actual_height = garment_y2 - garment_y1
            if actual_width < 10 or actual_height < 10:
                return None

            # Cloth texture — TPS returns full-frame image; GMM returns 192×256
            _tps_frame_space = (result.warped_cloth.shape[:2] == (h, w))
            if _tps_frame_space:
                warped_cloth_final = result.warped_cloth[
                    garment_y1:garment_y2, garment_x1:garment_x2
                ].copy()
            else:
                warped_cloth_final = cv2.resize(
                    result.warped_cloth, (actual_width, actual_height),
                    interpolation=cv2.INTER_LINEAR
                )

            # ── Alpha layer 1: garment silhouette mask ──────────────────
            # TPS: use the TPS-warped mask already in frame space (shows exact
            #       body-shaped silhouette from control-point warp)
            # GMM: fall back to original cloth mask resized to ROI
            wm = getattr(result, 'warped_mask', None)
            if _tps_frame_space and wm is not None and wm.shape[:2] == (h, w):
                garment_alpha = wm[garment_y1:garment_y2, garment_x1:garment_x2].copy()
                garment_alpha = garment_alpha.astype(np.float32)
            else:
                orig_mask = cloth_mask.squeeze() if cloth_mask.ndim == 3 else cloth_mask
                garment_alpha = cv2.resize(
                    orig_mask.astype(np.float32), (actual_width, actual_height),
                    interpolation=cv2.INTER_LINEAR
                )
            garment_alpha = np.clip(garment_alpha, 0.0, 1.0)

            # Enhanced edge anti-aliasing for smoother garment edges
            # Apply a multi-pass blur for natural looking edges
            garment_alpha_smooth = garment_alpha.copy()

            # Step 1: Light blur for immediate edge softening
            garment_alpha_smooth = cv2.GaussianBlur(garment_alpha_smooth, (3, 3), sigmaX=1, sigmaY=1)

            # Step 2: Selective blur on edge regions only to preserve interior details
            edge_mask = cv2.Canny((garment_alpha * 255).astype(np.uint8), 50, 150)
            edge_mask_dilated = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)
            edge_region = (edge_mask_dilated > 0).astype(np.float32)

            # Apply stronger blur only to edge regions
            alpha_heavy_blur = cv2.GaussianBlur(garment_alpha, (7, 7), sigmaX=3, sigmaY=3)
            garment_alpha = garment_alpha_smooth * (1 - edge_region) + alpha_heavy_blur * edge_region

            # ── Alpha layer 2: body silhouette (clip garment to body) ────────
            if body_mask is not None:
                bm = body_mask
                # Ensure mask spatial dims match frame
                if bm.shape[:2] != (h, w):
                    bm = cv2.resize(bm.astype(np.float32), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                bm_roi = bm[garment_y1:garment_y2, garment_x1:garment_x2].astype(np.float32)
                # Dilate to recover shoulder-side pixels MediaPipe may undercount
                _dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                bm_roi = cv2.dilate(bm_roi, _dil_k, iterations=2)
                bm_roi = cv2.GaussianBlur(bm_roi, (17, 17), sigmaX=7, sigmaY=7)
                bm_roi = np.clip(bm_roi, 0.0, 1.0)
                composite_alpha = garment_alpha * bm_roi
            else:
                composite_alpha = garment_alpha

            # Ensure 3-channel alpha for broadcasting
            warped_mask_3d = np.expand_dims(composite_alpha, axis=-1)

            # Extract frame region
            frame_roi = frame[garment_y1:garment_y2, garment_x1:garment_x2]
            if frame_roi.size == 0:
                return None
            
            # Enhanced alpha composite with subtle shading
            frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Add subtle shading effect based on body contours
            if body_mask is not None:
                # Create depth-like shading from body mask gradients
                bm_roi = body_mask[garment_y1:garment_y2, garment_x1:garment_x2].astype(np.float32)
                if bm_roi.shape == composite_alpha.shape:
                    # Calculate gradients to simulate depth/curvature
                    grad_x = cv2.Sobel(bm_roi, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(bm_roi, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    gradient_magnitude = np.clip(gradient_magnitude / gradient_magnitude.max(), 0, 1)

                    # Create subtle shadow/highlight effect
                    shadow_strength = 0.15  # Adjust for subtlety
                    shading = 1.0 - gradient_magnitude * shadow_strength
                    shading = np.expand_dims(shading, axis=-1)

                    # Apply shading to garment
                    warped_cloth_shaded = warped_cloth_final * shading
                else:
                    warped_cloth_shaded = warped_cloth_final
            else:
                warped_cloth_shaded = warped_cloth_final

            # Composite with improved blending
            composite = warped_cloth_shaded * warped_mask_3d + frame_roi_rgb * (1 - warped_mask_3d)

            # ── Hand occlusion ────────────────────────────────────
            # Where hand_mask = 1 (hand / forearm region), restore original
            # frame pixels so hands appear in FRONT of the garment.
            _hm = getattr(result, 'hand_mask', None)
            if _hm is not None and _hm.shape[:2] == (h, w):
                _hm_roi = _hm[garment_y1:garment_y2, garment_x1:garment_x2]
                _hm_f   = np.expand_dims(_hm_roi.astype(np.float32) / 255.0, -1)
                composite = composite * (1.0 - _hm_f) + frame_roi_rgb * _hm_f

            composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            output = frame.copy()
            output[garment_y1:garment_y2, garment_x1:garment_x2] = composite_bgr

            # ----------------------------------------------------------------
            # LAB COLOR GRADING
            # Match the garment's luminance and color to the scene's ambient
            # tone.  Sample a peripheral background ring outside the body
            # silhouette in the torso ROI, compute per-channel mean/std in
            # LAB space, then apply a linear transfer to the garment pixels.
            # ----------------------------------------------------------------
            try:
                roi = output[garment_y1:garment_y2, garment_x1:garment_x2]
                roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                # Background = inverse of the composite_alpha mask
                bg_mask = (composite_alpha < 0.15).astype(np.float32)
                bg_pix_count = bg_mask.sum()
                if bg_pix_count > 200:
                    fg_mask = (composite_alpha > 0.6)
                    for ch in range(3):
                        bg_vals = roi_lab[:, :, ch][bg_mask > 0]
                        fg_vals = roi_lab[:, :, ch][fg_mask]
                        if fg_vals.size < 50:
                            continue
                        bg_mean, bg_std = bg_vals.mean(), bg_vals.std() + 1e-6
                        fg_mean, fg_std = fg_vals.mean(), fg_vals.std() + 1e-6
                        # Linear histogram transfer: map fg stats toward bg stats
                        # Blend 40% toward background tone (avoid over-correction)
                        target_mean = fg_mean * 0.60 + bg_mean * 0.40
                        target_std  = fg_std  * 0.60 + bg_std  * 0.40
                        transfer_scale = target_std / fg_std
                        shifted = roi_lab[:, :, ch].copy()
                        shifted[fg_mask] = (
                            (shifted[fg_mask] - fg_mean) * transfer_scale + target_mean
                        )
                        roi_lab[:, :, ch] = shifted
                    roi_graded = cv2.cvtColor(
                        np.clip(roi_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
                    )
                    # Update both the output region AND composite_bgr so temporal
                    # smoothing operates on the graded frames
                    composite_bgr = roi_graded
                    output[garment_y1:garment_y2, garment_x1:garment_x2] = roi_graded
            except Exception:
                pass  # Non-fatal — keep ungraded composite

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
                # ── AdaptiveEMA temporal compositing (Phase D) ───────────────
                # Blend weight adapts to motion magnitude: fast garment
                # movement uses high alpha (responsive); still frames use
                # low alpha (smooth).  Range: 0.70 – 0.97 new-frame weight.
                _diff = np.abs(
                    composite_bgr.astype(np.float32) - _prev.astype(np.float32)
                )
                _motion = float(_diff.mean()) / 255.0     # normalised [0, 1]
                _alpha_new = float(0.70 + 0.27 * np.tanh(_motion * 20.0))
                smoothed = cv2.addWeighted(
                    composite_bgr, _alpha_new, _prev, 1.0 - _alpha_new, 0
                )
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
            
            composite = cloth_resized * mask_resized + frame_roi_rgb * (1 - mask_resized)
            composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = composite_bgr
            return frame
        
        except Exception as e:
            logger.error(f"VITON blending failed: {e}")
            return self._render_colored_shirt(frame, {})

    # ── HolisticTracker lazy-init helper ─────────────────────────────────────
    def _get_holistic_tracker(self):
        """Return the shared HolisticTracker, creating it on first call."""
        if not hasattr(self, '_holistic_tracker'):
            try:
                from src.core.holistic_tracker import HolisticTracker
                self._holistic_tracker = HolisticTracker(frame_skip=3, model_complexity=0)
                self._holistic_tracker.start()
                logger.info("[Renderer] HolisticTracker started")
            except Exception as exc:
                logger.warning("[Renderer] HolisticTracker unavailable: %s", exc)
                self._holistic_tracker = None
        return self._holistic_tracker

    def _render_garment(self, frame, garment):
        """Render real dataset garment image on the body using VITON loader."""
        t_render_start = time.perf_counter()

        # Feed frame into holistic tracker (non-blocking, daemon thread)
        _ht = self._get_holistic_tracker()
        if _ht is not None:
            _ht.enqueue(frame)

        garment_ref = None
        # Always prefer the actively selected garment from the runtime garment list.
        # dataset_pairs may contain entries that are not present in local sparse datasets.
        if self.garments:
            idx = self.current_garment_idx % len(self.garments)
            garment_ref = self.garments[idx]
        elif self.dataset_pairs:
            idx = self.current_garment_idx % len(self.dataset_pairs)
            pair = self.dataset_pairs[idx]
            garment_ref = pair.get('garment')

        cloth_rgb, cloth_mask = (None, None)
        if garment_ref:
            cloth_rgb, cloth_mask = self._load_garment_image(garment_ref)

        # Fallback: if chosen garment file is unavailable, try dataset pair entry.
        if cloth_rgb is None and cloth_mask is None and self.dataset_pairs:
            idx = self.current_garment_idx % len(self.dataset_pairs)
            pair = self.dataset_pairs[idx]
            pair_filename = pair.get('garment')
            if pair_filename:
                cloth_rgb, cloth_mask = self._load_garment_image(pair_filename)

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

                    # Inject holistic hand landmarks into body_measurements
                    # so _render_garment_neural can pass them to TPSPipeline.
                    if body_measurements is not None and _ht is not None:
                        _hr = _ht.get_latest()
                        if _hr is not None:
                            body_measurements = dict(body_measurements)  # shallow copy
                            body_measurements['hand_lm_left']  = _hr.left_hand_landmarks
                            body_measurements['hand_lm_right'] = _hr.right_hand_landmarks
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
                        
                        # Geometric fitting (Phase 0 or Phase 2 fallback)
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
            
            # No body fitter: simple blending
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

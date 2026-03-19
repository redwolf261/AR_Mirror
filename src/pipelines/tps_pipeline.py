"""
TPS Garment Warp Pipeline
=========================
Replaces the CP-VTON GMM path with a landmark-driven Thin Plate Spline warp
that works on live webcam input without any human-parsing pre-processing.

Pipeline summary per frame
--------------------------
1.  GarmentControlPoints  (computed once per garment, cached)
    → 12 canonical points in garment-image space

2.  BodyControlPoints     (computed per detection, interpolated between)
    → 12 corresponding points in live camera-frame space

3.  TPSWarp.fit(src, dst) → solve (N+3)² linear system (~0.3 ms)
    TPSWarp.warp_image    → cv2.remap garment texture    (~2 ms)
    TPSWarp.warp_mask     → cv2.remap garment mask       (~1 ms)

4.  RVMMatting            → body alpha matte (ONNX CUDA) (~10 ms)

5.  HandOccluder          → hand/forearm region mask     (~0.5 ms)

6.  Return TPSWarpResult (compatible with NeuralWarpResult interface)

Usage
-----
    from src.pipelines.tps_pipeline import TPSPipeline

    pipeline = TPSPipeline()
    result   = pipeline.warp(
        frame_bgr, cloth_rgb, cloth_mask, landmarks,
        hand_lm_left=left_hand, hand_lm_right=right_hand,
        garment_type='tshirt',
    )
    # result.warped_cloth  (H, W, 3) float32 [0,1] RGB
    # result.warped_mask   (H, W)    float32 [0,1]
    # result.rvm_alpha     (H, W)    float32 [0,1]
    # result.hand_mask     (H, W)    uint8
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.core.tps_warp               import TPSWarp
from src.core.garment_control_points import GarmentControlPoints
from src.core.body_control_points    import BodyControlPoints
from src.core.hand_occluder          import HandOccluder

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Result dataclass (drop-in compatible with NeuralWarpResult)
# ------------------------------------------------------------------

@dataclass
class TPSWarpResult:
    """
    Result from the TPS pipeline.

    Attributes
    ----------
    warped_cloth : (H, W, 3) float32 [0,1] RGB
        Garment texture warped to frame space.
    warped_mask : (H, W) float32 [0,1]
        Garment silhouette mask warped to frame space.
    rvm_alpha : (H, W) float32 [0,1]
        Body alpha matte from RVM (or binary body mask fallback).
    hand_mask : (H, W) uint8
        Hand/forearm region mask for occlusion compositing.
    quality_score : float
    timings : dict
    used_neural : bool
    synthesized : None  (TPS does not produce a synthesized full-body image)
    depth_proxy : float
    """
    warped_cloth:  np.ndarray
    warped_mask:   np.ndarray
    rvm_alpha:     np.ndarray
    hand_mask:     np.ndarray
    quality_score: float = 1.0
    timings:       Dict[str, float] = field(default_factory=dict)
    used_neural:   bool = True
    synthesized:   None = None
    depth_proxy:   float = 0.0


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class TPSPipeline:
    """
    Landmark-driven TPS garment warp pipeline.

    Thread-safety: not thread-safe. Call from the render thread only.
    The RVM model maintains recurrent state; concurrent calls would
    corrupt it.
    """

    def __init__(self) -> None:
        self._tps      = TPSWarp(regularization=1e-3)
        self._occluder = HandOccluder()
        self._rvm: Optional[Any] = None
        self._rvm_available = False
        self._init_rvm()

        # Per-garment GCP cache (keyed by id(cloth_mask))
        self._gcp_cache: dict[int, GarmentControlPoints] = {}

        # Last valid body control points (for interpolation)
        self._last_dst_pts: Optional[np.ndarray] = None

        # Pre-allocated canvas buffers (avoids 2 MB numpy zeroing every frame)
        self._canvas_cloth: Optional[np.ndarray] = None
        self._canvas_mask:  Optional[np.ndarray] = None
        self._canvas_shape: tuple[int, int] = (0, 0)

        # Control point change threshold for TPS refit (pixels).
        # If both src AND dst change less than this, reuse the previous warp grid.
        self._CP_REFIT_THRESH: float = 2.0
        self._prev_src_canvas: Optional[np.ndarray] = None
        self._prev_dst_pts:    Optional[np.ndarray] = None

        logger.info("[TPS] Pipeline initialised")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def warp(
        self,
        frame_bgr:     np.ndarray,
        cloth_rgb:     np.ndarray,
        cloth_mask:    np.ndarray,
        landmarks:     dict,
        hand_lm_left:  Optional[list[Any]] = None,
        hand_lm_right: Optional[list[Any]] = None,
        garment_type:  str = 'tshirt',
        cloth_mask_id: Optional[int] = None,
    ) -> Optional[TPSWarpResult]:
        """
        Run the full TPS warp pipeline for one frame.

        Parameters
        ----------
        frame_bgr : (H, W, 3) uint8 BGR
        cloth_rgb : (H, W, 3) float32 [0,1] RGB
        cloth_mask : (H, W) or (H, W, 1) float32 [0,1]
        landmarks : {idx: {'x','y','visibility'}} MediaPipe normalized
        hand_lm_left, hand_lm_right : MediaPipe hand landmark lists or None
        garment_type : 'tshirt' | 'shirt' | 'hoodie' | 'dress' | 'tank' …
        cloth_mask_id : optional integer key for GCP cache (use id(cloth_mask))

        Returns
        -------
        TPSWarpResult or None if control points cannot be computed
        """
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}
        h, w = frame_bgr.shape[:2]

        # ── 1. Garment control points (cached per garment) ──────────────
        t0 = time.perf_counter()
        cache_key = cloth_mask_id if cloth_mask_id is not None else id(cloth_mask)
        if cache_key not in self._gcp_cache:
            gcp_obj = GarmentControlPoints(cloth_mask)
            # Normalise to cloth_rgb spatial dimensions
            ch, cw = cloth_rgb.shape[:2]
            gcp_obj._mask = cv2.resize(
                gcp_obj._mask, (cw, ch), interpolation=cv2.INTER_NEAREST
            )
            self._gcp_cache[cache_key] = gcp_obj
        gcp = self._gcp_cache[cache_key]
        src_pts = gcp.compute()                     # (12, 2) garment px
        timings['gcp'] = time.perf_counter() - t0

        ch, cw = cloth_rgb.shape[:2]

        # ── 2. Body control points ───────────────────────────────────────
        t0 = time.perf_counter()
        dst_pts = BodyControlPoints.compute(landmarks, w, h, garment_type)
        timings['bcp'] = time.perf_counter() - t0

        if dst_pts is None:
            if self._last_dst_pts is not None:
                dst_pts = self._last_dst_pts
                logger.debug("[TPS] Using cached body control points")
            else:
                logger.debug("[TPS] No body control points — skipping frame")
                return None
        else:
            self._last_dst_pts = dst_pts

        # ── 3. Body-measurement coarse alignment ─────────────────────────
        # Pipeline:
        #   a) Measure body: shoulder span (sx) + collar-to-hem height (sy)
        #   b) Measure garment: same two distances from GCPs
        #   c) Scale garment so its torso exactly matches body torso dimensions
        #   d) Anchor: garment C-collar (pt 10) → body C-collar (pt 10)
        #      The body collar point is ABOVE the glenohumeral shoulder joint,
        #      sitting at collarbone height — exactly where the shirt collar sits.
        #   e) TPS corrects only small residual per-landmark misalignments.
        t0 = time.perf_counter()

        # ── Body measurements (frame pixel space) ────────────────────────
        b_l_shoulder = dst_pts[0].astype(np.float64)   # expanded left shoulder
        b_r_shoulder = dst_pts[1].astype(np.float64)   # expanded right shoulder
        b_collar_mid = dst_pts[10].astype(np.float64)  # C-collar (collarbone level)
        b_hem_mid    = dst_pts[11].astype(np.float64)  # C-hem

        body_span_x       = float(b_r_shoulder[0] - b_l_shoulder[0])
        body_collar_to_hem = max(float(b_hem_mid[1] - b_collar_mid[1]), 1.0)

        # ── Garment measurements (garment pixel space) ───────────────────
        g_l_shoulder = src_pts[0].astype(np.float64)
        g_r_shoulder = src_pts[1].astype(np.float64)
        g_collar_mid = src_pts[10].astype(np.float64)  # C-collar in garment px
        g_hem_mid    = src_pts[11].astype(np.float64)  # C-hem in garment px

        garment_span_x        = max(float(g_r_shoulder[0] - g_l_shoulder[0]), 1.0)
        garment_collar_to_hem = max(float(g_hem_mid[1]    - g_collar_mid[1]),  1.0)

        # ── Scale factors ────────────────────────────────────────────────
        sx = body_span_x       / garment_span_x
        sy = body_collar_to_hem / garment_collar_to_hem
        sx = float(np.clip(sx, 0.3, 5.0))
        sy = float(np.clip(sy, 0.3, 5.0))

        target_w = max(int(cw * sx), 64)
        target_h = max(int(ch * sy), 64)

        # Scale cloth and mask to body-fitted size
        cloth_sized = cv2.resize(cloth_rgb, (target_w, target_h),
                                 interpolation=cv2.INTER_LINEAR)
        cloth_mask_2d = (cloth_mask.squeeze() if cloth_mask.ndim == 3
                         else cloth_mask).astype(np.float32)
        mask_sized = cv2.resize(cloth_mask_2d, (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR)

        # ── Anchor: garment C-collar → body C-collar ────────────────────
        # This aligns the shirt neckhole with the collarbone, placing the
        # shoulder seams and hem at the correct body positions.
        g_anchor_scaled_x = int(round(g_collar_mid[0] * sx))
        g_anchor_scaled_y = int(round(g_collar_mid[1] * sy))

        paste_x = int(round(b_collar_mid[0])) - g_anchor_scaled_x
        paste_y = int(round(b_collar_mid[1])) - g_anchor_scaled_y

        # Build frame-sized canvas and paste the body-fitted garment
        # Re-use pre-allocated buffers; only re-allocate when frame size changes.
        if self._canvas_shape != (h, w):
            self._canvas_cloth = np.zeros((h, w, 3), dtype=np.float32)
            self._canvas_mask  = np.zeros((h, w),    dtype=np.float32)
            self._canvas_shape = (h, w)
        else:
            self._canvas_cloth[:] = 0.0
            self._canvas_mask[:]  = 0.0
        canvas_cloth = self._canvas_cloth
        canvas_mask  = self._canvas_mask

        x0 = max(0, paste_x);  x1 = min(w, paste_x + target_w)
        y0 = max(0, paste_y);  y1 = min(h, paste_y + target_h)
        gx0 = x0 - paste_x;   gx1 = gx0 + (x1 - x0)
        gy0 = y0 - paste_y;   gy1 = gy0 + (y1 - y0)

        if x1 > x0 and y1 > y0:
            canvas_cloth[y0:y1, x0:x1] = cloth_sized[gy0:gy1, gx0:gx1]
            canvas_mask [y0:y1, x0:x1] = mask_sized [gy0:gy1, gx0:gx1]

        # Express garment control points in canvas (frame) coordinates
        src_pts_canvas = np.zeros_like(src_pts, dtype=np.float32)
        src_pts_canvas[:, 0] = src_pts[:, 0] * sx + paste_x
        src_pts_canvas[:, 1] = src_pts[:, 1] * sy + paste_y

        # Clamp to frame
        src_pts_canvas[:, 0] = np.clip(src_pts_canvas[:, 0], 0, w - 1)
        src_pts_canvas[:, 1] = np.clip(src_pts_canvas[:, 1], 0, h - 1)

        timings['body_fit'] = time.perf_counter() - t0
        logger.debug("[TPS] body-fit: sx=%.2f sy=%.2f paste=(%d,%d) target=%dx%d",
                     sx, sy, paste_x, paste_y, target_w, target_h)

        # ── 4. TPS fine-correction solve + remap ────────────────────────
        # Skip refit when control points haven't changed enough — reuse
        # the cached warp grid for smooth motion continuity.
        need_refit = True
        if self._prev_src_canvas is not None and self._prev_dst_pts is not None:
            src_delta = float(np.abs(src_pts_canvas - self._prev_src_canvas).max())
            dst_delta = float(np.abs(dst_pts         - self._prev_dst_pts        ).max())
            if src_delta < self._CP_REFIT_THRESH and dst_delta < self._CP_REFIT_THRESH:
                need_refit = False
        self._prev_src_canvas = src_pts_canvas.copy()
        self._prev_dst_pts    = dst_pts.copy()
        t0 = time.perf_counter()
        if need_refit:
            try:
                self._tps.fit(src_pts_canvas, dst_pts)
            except np.linalg.LinAlgError as exc:
                logger.warning("[TPS] TPS solve failed: %s", exc)
                return None
        timings['tps_solve'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        warped_cloth = self._tps.warp_image(canvas_cloth)
        warped_mask  = self._tps.warp_mask(canvas_mask)

        # Soften mask edges for natural alpha blending
        warped_mask = cv2.GaussianBlur(warped_mask, (21, 21), 7)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)
        timings['tps_remap'] = time.perf_counter() - t0

        # ── 4. RVM body alpha ────────────────────────────────────────────
        t0 = time.perf_counter()
        rvm_alpha = self._get_rvm_alpha(frame_bgr, h, w)
        timings['rvm'] = time.perf_counter() - t0

        # ── 5. Hand occlusion mask ───────────────────────────────────────
        t0 = time.perf_counter()
        hand_mask = self._occluder.make_mask(
            frame_bgr.shape, hand_lm_left, hand_lm_right, landmarks
        )
        timings['hand_mask'] = time.perf_counter() - t0

        timings['total'] = time.perf_counter() - t_total
        logger.debug("[TPS] %.1f ms total  (solve=%.1f remap=%.1f rvm=%.1f)",
                     timings['total'] * 1e3,
                     timings.get('tps_solve', 0) * 1e3,
                     timings.get('tps_remap', 0) * 1e3,
                     timings.get('rvm', 0) * 1e3)

        return TPSWarpResult(
            warped_cloth  = warped_cloth,
            warped_mask   = warped_mask,
            rvm_alpha     = rvm_alpha,
            hand_mask     = hand_mask,
            quality_score = float(warped_mask.mean() * 4.0),  # rough coverage score
            timings       = timings,
            used_neural   = True,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_rvm(self) -> None:
        try:
            from src.core.rvm_matting import RVMMatting
            self._rvm = RVMMatting()
            self._rvm_available = self._rvm.available
            if self._rvm_available:
                logger.info("[TPS] RVM body matting ready")
            else:
                logger.info("[TPS] RVM unavailable — will use body_mask fallback")
        except Exception as exc:
            self._rvm = None
            self._rvm_available = False
            logger.warning("[TPS] RVM import failed: %s", exc)

    def _get_rvm_alpha(
        self,
        frame_bgr: np.ndarray,
        h: int, w: int,
    ) -> np.ndarray:
        """
        Return (H, W) float32 [0,1] body alpha. Falls back to uniform 1.0
        when RVM is unavailable (garment alpha is then controlled solely by
        the warped cloth mask * body_mask from rendering.py).
        """
        if self._rvm_available and self._rvm is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                alpha = self._rvm.matte(frame_rgb)
                if alpha is not None:
                    if alpha.shape != (h, w):
                        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
                    return alpha.astype(np.float32)
            except Exception as exc:
                logger.debug("[TPS] RVM matte error: %s", exc)
        # Fallback: all body pixels visible (rely on body_mask from MediaPipe)
        return np.ones((h, w), dtype=np.float32)

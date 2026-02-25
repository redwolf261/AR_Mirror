#!/usr/bin/env python3
"""
AR Try-On Selector
==================
Left panel  – Live webcam with selected garment composited onto your body.
Right panel – Scrollable grid of garments loaded from dataset/train/cloth/.

Controls
--------
  Mouse click on right panel  →  select & wear that garment
  Scroll wheel (on panel)     →  browse garment list
  W / S / A / D  or arrows   →  keyboard navigation
  Q  or  ESC                 →  quit
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from collections import deque
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CAM_W, CAM_H = 960, 720        # live-camera panel
PANEL_W       = 400             # garment picker width
WINDOW_W      = CAM_W + PANEL_W
WINDOW_H      = CAM_H

THUMB_COLS    = 2
THUMB_PAD     = 8
THUMB_W       = (PANEL_W - THUMB_PAD * (THUMB_COLS + 1)) // THUMB_COLS  # ≈ 188
THUMB_H       = int(THUMB_W * 256 / 192)                               # keep CP-VTON aspect

PANEL_HEADER_H = 56
GARMENT_AREA_H = WINDOW_H - PANEL_HEADER_H
VISIBLE_ROWS   = GARMENT_AREA_H // (THUMB_H + THUMB_PAD)

# Dataset paths (relative to script)
DATASET_TRAIN = Path("dataset/train")
CLOTH_DIR     = DATASET_TRAIN / "cloth"
MASK_DIR      = DATASET_TRAIN / "cloth-mask"
MAX_GARMENTS  = 200            # limit to keep start-up fast

# Fixed torso fractions (used when pose detection unavailable)
FIX_X1, FIX_X2 = 0.18, 0.82
FIX_Y1, FIX_Y2 = 0.18, 0.80


# ---------------------------------------------------------------------------
# Mouse state  (shared between callback and main loop)
# ---------------------------------------------------------------------------
_mouse: dict = {"click_x": -1, "click_y": -1, "clicked": False, "scroll": 0}

def _on_mouse(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse["click_x"] = x
        _mouse["click_y"] = y
        _mouse["clicked"] = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        # flags > 0 → scroll up (negative delta), flags < 0 → scroll down
        _mouse["scroll"] += -1 if flags > 0 else 1


# ---------------------------------------------------------------------------
# Garment panel
# ---------------------------------------------------------------------------
class GarmentPanel:
    """Scrollable grid of garment thumbnails + selection state."""

    def __init__(self):
        self.filenames: list = []
        self.thumbs:    list = []
        self.selected   = 0
        self.scroll_row = 0
        self._load()

    # ---- loading ----
    def _load(self):
        paths = sorted(CLOTH_DIR.glob("*.jpg"))[:MAX_GARMENTS]
        if not paths:
            paths = sorted(CLOTH_DIR.glob("*.png"))[:MAX_GARMENTS]
        log.info(f"Loading {len(paths)} garments …")
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            thumb = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            self.filenames.append(p.name)
            self.thumbs.append(thumb)
        log.info(f"  → {len(self.thumbs)} garments loaded")

    # ---- rendering ----
    def render(self) -> np.ndarray:
        panel = np.full((WINDOW_H, PANEL_W, 3), 28, dtype=np.uint8)

        # ── header ──
        cv2.rectangle(panel, (0, 0), (PANEL_W, PANEL_HEADER_H), (45, 45, 45), -1)
        cv2.putText(panel, "Garments  — click to wear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
        sel_name = self.filenames[self.selected] if self.filenames else "—"
        cv2.putText(panel, f"Wearing: {sel_name[:24]}", (10, PANEL_HEADER_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 110), 1, cv2.LINE_AA)

        # ── thumbnails ──
        for i, thumb in enumerate(self.thumbs):
            row = i // THUMB_COLS
            col = i %  THUMB_COLS
            if row < self.scroll_row:
                continue
            vis_row = row - self.scroll_row
            y0 = PANEL_HEADER_H + THUMB_PAD + vis_row * (THUMB_H + THUMB_PAD)
            if y0 + THUMB_H > WINDOW_H:
                break
            x0 = THUMB_PAD + col * (THUMB_W + THUMB_PAD)

            panel[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb

            is_sel = (i == self.selected)
            color  = (40, 220, 110) if is_sel else (75, 75, 75)
            thick  = 3              if is_sel else 1
            cv2.rectangle(panel, (x0 - 1, y0 - 1),
                          (x0 + THUMB_W, y0 + THUMB_H), color, thick)

            # index badge
            cv2.putText(panel, str(i + 1), (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # ── scroll bar ──
        total_rows = max(1, (len(self.thumbs) + THUMB_COLS - 1) // THUMB_COLS)
        if total_rows > VISIBLE_ROWS:
            bar_h = max(20, int(GARMENT_AREA_H * VISIBLE_ROWS / total_rows))
            bar_y = PANEL_HEADER_H + int(
                GARMENT_AREA_H * self.scroll_row / max(1, total_rows - VISIBLE_ROWS)
            )
            bar_y = min(bar_y, WINDOW_H - bar_h)
            cv2.rectangle(panel, (PANEL_W - 7, bar_y),
                          (PANEL_W - 3, bar_y + bar_h), (110, 110, 110), -1)

        # ── tip ──
        cv2.putText(panel, "Scroll or W/S/A/D to navigate",
                    (8, WINDOW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)
        return panel

    # ---- interaction ----
    def handle_click(self, panel_x: int, panel_y: int) -> bool:
        """Return True if selection changed."""
        if panel_y < PANEL_HEADER_H or panel_x < THUMB_PAD:
            return False
        rel_y = panel_y - PANEL_HEADER_H - THUMB_PAD
        if rel_y < 0:
            return False
        row = rel_y // (THUMB_H + THUMB_PAD) + self.scroll_row
        col = (panel_x - THUMB_PAD) // (THUMB_W + THUMB_PAD)
        if col < 0 or col >= THUMB_COLS:
            return False
        idx = row * THUMB_COLS + col
        if 0 <= idx < len(self.filenames):
            self.selected = idx
            self._clamp_scroll()
            return True
        return False

    def navigate(self, delta: int):
        self.selected = max(0, min(len(self.filenames) - 1, self.selected + delta))
        self._clamp_scroll()

    def scroll(self, delta: int):
        total_rows = (len(self.thumbs) + THUMB_COLS - 1) // THUMB_COLS
        self.scroll_row = max(0, min(total_rows - VISIBLE_ROWS, self.scroll_row + delta))

    def _clamp_scroll(self):
        sel_row = self.selected // THUMB_COLS
        if sel_row < self.scroll_row:
            self.scroll_row = sel_row
        elif sel_row >= self.scroll_row + VISIBLE_ROWS:
            self.scroll_row = sel_row - VISIBLE_ROWS + 1
        self.scroll_row = max(0, self.scroll_row)

    @property
    def selected_filename(self):
        return self.filenames[self.selected] if self.filenames else None


# ---------------------------------------------------------------------------
# Garment renderer  — 3-tier: neural GMM → geometric body-fit → fixed blend
# ---------------------------------------------------------------------------
class GarmentRenderer:
    """
    Tier 1 – Phase2NeuralPipeline (GMM TPS warping via ONNX): garment
             actually deforms to your shoulder/torso shape with wrinkles.
    Tier 2 – BodyAwareGarmentFitter (pose landmark + body-mask scaling):
             geometrically scales garment to measured torso.
    Tier 3 – Fixed-fraction alpha-blend fallback.
    """

    def __init__(self):
        self._cache: dict  = {}
        self._fitter       = None   # BodyAwareGarmentFitter
        self._pipeline     = None   # Phase2NeuralPipeline
        self._last_meas    = None   # cached body measurements
        self._meas_age     = 0
        self._MEAS_TTL     = 5      # re-detect every 5 frames
        self._dbg_count    = 0      # controls diagnostic print rate
        self._init_fitter()
        self._init_pipeline()

    # ---- tier-2: body-aware geometric fitter ----
    def _init_fitter(self):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from src.core.body_aware_fitter import BodyAwareGarmentFitter
            self._fitter = BodyAwareGarmentFitter(
                model_path=str(Path("pose_landmarker_lite.task"))
            )
            log.info("Tier 2 ready: BodyAwareGarmentFitter (geometric body-fit)")
        except Exception as e:
            log.warning(f"BodyAwareGarmentFitter unavailable → fixed torso fallback  ({e})")

    # ---- tier-1: neural GMM warping ----
    def _init_pipeline(self):
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            self._pipeline = Phase2NeuralPipeline(
                device='auto',
                enable_tom=False,        # TOM synthesis is slow; GMM alone is enough
                batch_size=1,
                enable_optimizations=True,
            )
            log.info("Tier 1 ready: Phase2NeuralPipeline (GMM neural warping)")
        except Exception as e:
            log.warning(f"Phase2NeuralPipeline unavailable → geometric fallback  ({e})")

    # ---- garment loading ----
    def _load(self, filename: str):
        """Returns (cloth_rgb_float32_[0,1], mask_HW1_float32)."""
        if filename in self._cache:
            return self._cache[filename]
        cloth_path = CLOTH_DIR / filename
        mask_path  = MASK_DIR  / filename
        if not cloth_path.exists() or not mask_path.exists():
            self._cache[filename] = (None, None)
            return None, None
        cloth_bgr  = cv2.imread(str(cloth_path))
        if cloth_bgr is None:
            log.warning(f"Failed to read cloth image: {cloth_path}")
            self._cache[filename] = (None, None)
            return None, None
        cloth_rgb  = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_gray  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            log.warning(f"Failed to read mask image: {mask_path}")
            self._cache[filename] = (None, None)
            return None, None
        mask       = (mask_gray > 127).astype(np.float32)[:, :, np.newaxis]   # H×W×1
        self._cache[filename] = (cloth_rgb, mask)
        return cloth_rgb, mask

    # ---- landmark format converter ----
    @staticmethod
    def _lm_to_dict(landmarks) -> dict:
        """Convert MediaPipe Tasks API landmark list → {idx: {'x','y','visibility'}}."""
        return {
            i: {'x': lm.x, 'y': lm.y, 'visibility': getattr(lm, 'visibility', 1.0)}
            for i, lm in enumerate(landmarks)
        }

    # ---- tier-3 fallback ----
    def _fixed_blend(self, frame, cloth_rgb, mask):
        h, w  = frame.shape[:2]
        x1, x2 = int(w * FIX_X1), int(w * FIX_X2)
        y1, y2 = int(h * FIX_Y1), int(h * FIX_Y2)
        bw, bh = x2 - x1, y2 - y1
        c = cv2.resize(cloth_rgb, (bw, bh), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask.squeeze(), (bw, bh), interpolation=cv2.INTER_LINEAR)
        m = m[:, :, np.newaxis]
        out = frame.copy()
        roi = out[y1:y2, x1:x2].astype(np.float32) / 255.0
        roi_bgr = cv2.cvtColor(
            np.clip(c * m + roi * (1 - m), 0, 1).astype(np.float32),
            cv2.COLOR_RGB2BGR
        )
        out[y1:y2, x1:x2] = (roi_bgr * 255).astype(np.uint8)
        return out

    # ---- main render ----
    def render(self, frame: np.ndarray, filename: str) -> np.ndarray:
        cloth_rgb, mask = self._load(filename)
        if cloth_rgb is None:
            return frame

        # ── Get body measurements (shared by tier-1 and tier-2) ─────────
        measurements = None
        if self._fitter is not None:
            try:
                self._meas_age += 1
                if self._last_meas is None or self._meas_age >= self._MEAS_TTL:
                    m = self._fitter.extract_body_measurements(frame)
                    if m is not None:
                        self._last_meas = m
                        self._dbg_count += 1
                    self._meas_age = 0
                measurements = self._last_meas
            except Exception as e:
                log.debug(f"Fitter measurement error: {e}")

        # ── Validate & improve torso box using segmentation mask ─────────
        if measurements is not None:
            body_mask = measurements.get('body_mask')
            tb = measurements.get('torso_box')

            # If segmentation mask is available, derive a reliable torso box from it
            if body_mask is not None and body_mask.any():
                h_fr, w_fr = frame.shape[:2]
                # Resize mask to match current frame if dimensions differ
                if body_mask.shape[:2] != (h_fr, w_fr):
                    body_mask = cv2.resize(
                        body_mask.astype(np.float32), (w_fr, h_fr),
                        interpolation=cv2.INTER_NEAREST
                    )
                    body_mask = (body_mask > 0.5).astype(np.uint8)
                bm2d = body_mask.squeeze()            # (H,W,1) → (H,W)
                ys, xs = np.where(bm2d > 0)
                if len(ys) > 100:  # enough pixels to be a real body
                    bx1, bx2 = int(xs.min()), int(xs.max())
                    by1, by2 = int(ys.min()), int(ys.max())
                    bh_px    = by2 - by1

                    # MediaPipe segmentation starts BELOW the shoulder line.
                    # Shirt top (collar) is ~40% of the mask height ABOVE by1.
                    # Shirt covers down to about by2 (waist).
                    shirt_top    = max(0, by1 - int(bh_px * 0.45))
                    shirt_bottom = by2
                    measurements = dict(measurements)   # don't mutate cached copy
                    measurements['torso_box'] = (bx1, shirt_top, bx2, shirt_bottom)
                    if self._dbg_count % 30 == 1:
                        log.info(f"[SEG] box ({bx1},{shirt_top},{bx2},{shirt_bottom})  mask_y=[{by1},{by2}]  frame={frame.shape[:2]}")

            elif tb is not None:
                # Fallback: validate landmark-based box
                tx1, ty1, tx2, ty2 = tb
                if tx2 < tx1: tx1, tx2 = tx2, tx1
                if ty2 < ty1: ty1, ty2 = ty2, ty1
                box_w = tx2 - tx1
                h_fr  = frame.shape[0]
                w_fr  = frame.shape[1]
                if box_w < w_fr * 0.10 or ty1 > h_fr * 0.70:
                    log.debug(f"Torso box rejected ({box_w}px wide, ty1={ty1}) — fixed fallback")
                    measurements = None

        # ── Tier 1: Neural GMM warping ───────────────────────────────────
        if self._pipeline is not None and measurements is not None:
            try:
                landmarks   = measurements['landmarks']
                mp_lm_dict  = self._lm_to_dict(landmarks)
                body_mask   = measurements.get('body_mask')

                # Person image: RGB float32 [0,1] at 256×192
                if frame is None:
                    raise ValueError("Null camera frame")
                frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # Cloth mask as 2-D for pipeline
                cloth_mask_2d = mask.squeeze()

                # Body mask as float32 if available
                body_mask_f = body_mask.astype(np.float32) if body_mask is not None else None

                result = self._pipeline.warp_garment(
                    person_image=frame_rgb,
                    cloth_rgb=cloth_rgb,
                    cloth_mask=cloth_mask_2d,
                    mp_landmarks=mp_lm_dict,
                    body_mask=body_mask_f,
                )

                if result.warped_cloth is not None:
                    return self._place_warped(frame, result, measurements)
            except Exception as e:
                log.debug(f"Neural pipeline error: {e}")

        # ── Tier 2: Geometric body-fit ────────────────────────────────────
        if self._fitter is not None and measurements is not None:
            try:
                return self._fitter.fit_garment_to_body(
                    frame, cloth_rgb, mask, measurements
                )
            except Exception as e:
                log.debug(f"Geometric fitter error: {e}")

        # ── Tier 3: Fixed-region fallback ────────────────────────────────
        return self._fixed_blend(frame, cloth_rgb, mask)

    def _place_warped(self, frame, result, measurements) -> np.ndarray:
        """Composite neural-warped garment onto the frame at the torso box."""
        h, w = frame.shape[:2]
        torso_x1, torso_y1, torso_x2, torso_y2 = measurements['torso_box']
        body_mask = measurements.get('body_mask')

        # Fully normalise the torso box before any arithmetic
        if torso_x2 < torso_x1: torso_x1, torso_x2 = torso_x2, torso_x1
        if torso_y2 < torso_y1: torso_y1, torso_y2 = torso_y2, torso_y1
        torso_y1 = max(0, min(torso_y1, int(h * 0.65)))   # shoulder no lower than 65 %
        torso_y2 = max(torso_y1 + 40, min(torso_y2, h))   # clamp hip to frame edge
        torso_x1 = max(0, torso_x1)
        torso_x2 = min(w, torso_x2)

        raw_box_w = torso_x2 - torso_x1
        raw_box_h = torso_y2 - torso_y1

        # Garment must be at LEAST 35 % of frame width so it covers the torso
        # even when the landmark detector returns a too-narrow box
        target_w = max(int(w * 0.35), int(raw_box_w * 1.15))
        target_h = max(2, int(raw_box_h * 1.20))

        wc = cv2.resize(result.warped_cloth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        wm = cv2.resize(result.warped_mask,  (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if wm.ndim == 2:
            wm = wm[:, :, np.newaxis]

        # Centre on torso
        cx   = (torso_x1 + torso_x2) // 2
        gx1  = max(0, cx - target_w // 2)
        gy1  = max(0, torso_y1)
        gx2  = min(w, gx1 + target_w)
        gy2  = min(h, gy1 + target_h)

        aw, ah = gx2 - gx1, gy2 - gy1
        if aw < 2 or ah < 2:
            return frame

        wc = wc[:ah, :aw]
        wm = wm[:ah, :aw]

        # Optionally restrict to body silhouette
        if body_mask is not None:
            # Ensure mask matches frame dimensions before slicing
            if body_mask.shape[:2] != (h, w):
                body_mask = cv2.resize(
                    body_mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            bm2d  = body_mask.squeeze()               # (H,W,1) → (H,W)
            bm_roi = bm2d[gy1:gy2, gx1:gx2].astype(np.float32)
            if bm_roi.ndim == 2:
                bm_roi = bm_roi[:, :, np.newaxis]
            wm = wm * bm_roi

        out = frame.copy()
        roi = out[gy1:gy2, gx1:gx2].astype(np.float32) / 255.0
        # warped_cloth is RGB float32 [0,1] — convert to BGR for frame
        wc_bgr = cv2.cvtColor(wc, cv2.COLOR_RGB2BGR)
        comp   = wc_bgr * wm + roi * (1.0 - wm)
        out[gy1:gy2, gx1:gx2] = np.clip(comp * 255, 0, 255).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("  AR Try-On Selector")
    log.info("=" * 60)

    if not CLOTH_DIR.exists():
        log.error(f"Dataset not found at: {CLOTH_DIR.resolve()}")
        log.error("Ensure CP-VTON dataset is at  dataset/train/cloth/")
        sys.exit(1)

    panel    = GarmentPanel()
    renderer = GarmentRenderer()

    if not panel.filenames:
        log.error("No garment images found in dataset/train/cloth/")
        sys.exit(1)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam (device 0).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("AR Try-On | Click a garment to wear it", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AR Try-On | Click a garment to wear it", WINDOW_W, WINDOW_H)
    cv2.setMouseCallback("AR Try-On | Click a garment to wear it", _on_mouse)

    log.info(f"\n  {len(panel.filenames)} garments ready.")
    log.info("  Click a thumbnail on the right to wear it.")
    log.info("  Use W/S/A/D or arrow keys to navigate, Q to quit.\n")

    fps_buf: deque = deque(maxlen=30)

    while True:
        t0     = time.perf_counter()
        ret, f = cap.read()
        if not ret:
            continue

        # Mirror + resize to panel width
        f = cv2.flip(f, 1)
        f = cv2.resize(f, (CAM_W, CAM_H))

        # ── Process mouse events ──────────────────────────────
        if _mouse["clicked"]:
            mx, my = _mouse["click_x"], _mouse["click_y"]
            _mouse["clicked"] = False
            if mx >= CAM_W:                # click inside right panel
                panel.handle_click(mx - CAM_W, my)

        if _mouse["scroll"] != 0:
            panel.scroll(_mouse["scroll"])
            _mouse["scroll"] = 0

        # ── Render garment on body ────────────────────────────
        sel   = panel.selected_filename
        frame = renderer.render(f, sel) if sel else f

        # ── DEBUG: draw body segmentation bounding box ──────────
        if renderer._last_meas is not None:
            bm = renderer._last_meas.get('body_mask')
            if bm is not None and bm.any():
                h_fr, w_fr = frame.shape[:2]
                if bm.shape[:2] != (h_fr, w_fr):
                    bm = cv2.resize(bm.astype(np.float32), (w_fr, h_fr),
                                    interpolation=cv2.INTER_NEAREST)
                    bm = (bm > 0.5).astype(np.uint8)
                bm2d = bm.squeeze()                   # (H,W,1) → (H,W)
                ys, xs = np.where(bm2d > 0)
                if len(ys) > 100:
                    bx1, bx2 = int(xs.min()), int(xs.max())
                    by1, by2 = int(ys.min()), int(ys.max())
                    shirt_top = max(0, by1 - int((by2 - by1) * 0.45))
                    cv2.rectangle(frame, (bx1, shirt_top), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(frame, f"SEG y={shirt_top}-{by2}",
                                (bx1, max(0, shirt_top - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # ── FPS + garment label on left panel ────────────────
        dt = time.perf_counter() - t0
        fps_buf.append(dt)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

        cv2.rectangle(frame, (0, 0), (400, 70), (0, 0, 0), -1)   # status bar bg
        cv2.putText(frame, f"FPS: {fps:.0f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Wearing: {sel or 'none'}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    "Click garment on right  |  Scroll to browse  |  Q = quit",
                    (10, CAM_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Compose full window ───────────────────────────────
        right = panel.render()
        window = np.hstack([frame, right])
        cv2.imshow("AR Try-On | Click a garment to wear it", window)

        # ── Keyboard ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):            # Q / ESC
            break
        elif key in (82, ord('w')):          # UP
            panel.navigate(-THUMB_COLS)
        elif key in (84, ord('s')):          # DOWN
            panel.navigate(+THUMB_COLS)
        elif key in (81, ord('a')):          # LEFT
            panel.navigate(-1)
        elif key in (83, ord('d')):          # RIGHT
            panel.navigate(+1)

    cap.release()
    cv2.destroyAllWindows()
    log.info("Bye!")


if __name__ == "__main__":
    main()

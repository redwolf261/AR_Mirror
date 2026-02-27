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
from web_server import WebServer, get_param as _wp
from auto_calibrator import AutoCalibrator
from pathlib import Path
import sys
import logging
import json
import threading
import urllib.request
import urllib.error
from collections import deque
import time

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
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
# Skeleton / landmark overlay  (toggle with K key)
# ---------------------------------------------------------------------------
# MediaPipe pose landmark indices used for the garment anchor skeleton
_SKEL_JOINTS = {
    0:  ("nose",     (255, 255, 255)),
    11: ("L.shldr",  (0, 220, 255)),
    12: ("R.shldr",  (0, 220, 255)),
    13: ("L.elbow",  (80, 180, 255)),
    14: ("R.elbow",  (80, 180, 255)),
    15: ("L.wrist",  (140, 140, 255)),
    16: ("R.wrist",  (140, 140, 255)),
    23: ("L.hip",    (0, 255, 120)),
    24: ("R.hip",    (0, 255, 120)),
}
_SKEL_BONES = [
    (11, 12, (0, 220, 255)),    # shoulder bar
    (23, 24, (0, 255, 120)),    # hip bar
    (11, 23, (0, 255, 60)),     # left side torso
    (12, 24, (0, 255, 60)),     # right side torso
    (11, 13, (80, 200, 255)),   # left upper arm
    (12, 14, (80, 200, 255)),   # right upper arm
    (13, 15, (140, 160, 255)),  # left forearm
    (14, 16, (140, 160, 255)),  # right forearm
    (0,  11, (200, 200, 200)),  # neck left
    (0,  12, (200, 200, 200)),  # neck right
]


def draw_skeleton_overlay(frame: np.ndarray, meas: dict) -> None:
    """
    Draw pose skeleton + garment anchor points directly on *frame* (in-place).
    Shows exactly where the torso box and shoulder/hip joints sit.
    """
    if meas is None:
        return
    landmarks = meas.get('landmarks')
    torso_box = meas.get('torso_box')
    h, w = frame.shape[:2]

    # --- Draw torso anchor box (where garment is placed) ---
    if torso_box is not None:
        tx1, ty1, tx2, ty2 = torso_box
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 180, 255), 2)
        cv2.putText(frame, "GARMENT BOX", (tx1 + 4, ty1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)

    if landmarks is None:
        return

    # Convert normalised coords → pixel coords
    pts = {}
    for idx, (label, col) in _SKEL_JOINTS.items():
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        vis = getattr(lm, 'visibility', 1.0)
        if vis < 0.25:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        pts[idx] = (px, py, col, label)

    # --- Draw bones ---
    for i, j, col in _SKEL_BONES:
        if i in pts and j in pts:
            cv2.line(frame, pts[i][:2], pts[j][:2], col, 2, cv2.LINE_AA)

    # --- Draw joint dots + labels ---
    for idx, (px, py, col, label) in pts.items():
        cv2.circle(frame, (px, py), 7, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)   # black border
        # Label only the anchor joints to avoid clutter
        if idx in (11, 12, 23, 24):
            cv2.putText(frame, label, (px + 9, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # --- Midpoint markers: collar centre & waist centre ---
    if 11 in pts and 12 in pts:
        mx = (pts[11][0] + pts[12][0]) // 2
        my = (pts[11][1] + pts[12][1]) // 2
        cv2.drawMarker(frame, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(frame, "collar", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    if 23 in pts and 24 in pts:
        mx = (pts[23][0] + pts[24][0]) // 2
        my = (pts[23][1] + pts[24][1]) // 2
        cv2.drawMarker(frame, (mx, my), (0, 255, 120), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(frame, "waist", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# SKU session logger  (Step 5B — tracks garment view time to disk)
# ---------------------------------------------------------------------------
class SKUSessionLogger:
    """Appends per-garment dwell-time records to data/logs/session_data.jsonl."""

    LOG_PATH = Path("data/logs/session_data.jsonl")

    def __init__(self):
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._sku:   str   = ""
        self._shape: str   = "UNKNOWN"
        self._start: float = 0.0

    def on_change(self, new_sku: str, body_shape: str = "UNKNOWN"):
        """Call when the selected garment changes."""
        now = time.time()
        if self._sku:
            record = {
                "sku":        self._sku,
                "duration_s": round(now - self._start, 2),
                "body_shape": self._shape,
                "ts":         now,
            }
            with open(self.LOG_PATH, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        self._sku   = new_sku or ""
        self._shape = body_shape
        self._start = now

    def flush(self):
        """Flush any pending record on exit."""
        if self._sku:
            self.on_change("", self._shape)


# --------------------------------------------------------------------------
# Cloud uploader  (Step 6 — background POST to NestJS /measurements)
# --------------------------------------------------------------------------
_BACKEND_URL = "http://localhost:3000"   # override via BACKEND_URL env var
import os as _os
_BACKEND_URL = _os.environ.get("BACKEND_URL", _BACKEND_URL)


def _post_measurement_bg(payload: dict) -> None:
    """POST *payload* to the backend /measurements endpoint (best-effort)."""
    url  = f"{_BACKEND_URL}/measurements"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            log.debug(f"[cloud] POST /measurements → {resp.status}")
    except urllib.error.URLError as e:
        log.debug(f"[cloud] POST skipped (backend unreachable): {e.reason}")
    except Exception as e:
        log.debug(f"[cloud] POST error: {e}")


def try_upload_measurement(meas: dict, garment_sku: str, body_shape: str = "UNKNOWN") -> None:
    """
    Fire-and-forget upload of body measurements to the cloud backend.
    Runs in a daemon thread so it never blocks the AR render loop.
    """
    if not meas:
        return
    shoulder_px = meas.get('shoulder_width', 0)
    torso_px    = meas.get('torso_height', 0)
    if shoulder_px < 10:
        return
    ppc          = shoulder_px / _REAL_SHOULDER_CM
    shoulder_cm  = _REAL_SHOULDER_CM
    chest_cm     = shoulder_cm * 1.25
    torso_cm     = torso_px / ppc

    payload = {
        "shoulderWidthCm": round(shoulder_cm, 2),
        "chestWidthCm":    round(chest_cm, 2),
        "torsoLengthCm":   round(torso_cm, 2),
        "confidence":      0.9,
        "detectedRegions": ["upper_body"],
        "garmentSku":      garment_sku,
        "bodyShape":       body_shape,
    }
    t = threading.Thread(target=_post_measurement_bg, args=(payload,), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Measurements HUD helper
# ---------------------------------------------------------------------------
_REAL_SHOULDER_CM = 42.0   # same reference as BodyAwareGarmentFitter

def _infer_size(shoulder_cm: float) -> tuple:
    """Return (size_label, color_bgr) based on shoulder width in cm."""
    if shoulder_cm < 38:
        return "XS", (255, 200, 60)
    elif shoulder_cm < 42:
        return "S",  (80, 220, 110)
    elif shoulder_cm < 46:
        return "M",  (80, 220, 110)
    elif shoulder_cm < 50:
        return "L",  (255, 200, 60)
    else:
        return "XL", (60, 100, 255)

def draw_measurements_hud(frame: np.ndarray, meas: dict) -> None:
    """
    Draw a measurements HUD in the bottom-right of the camera view.
    Modifies *frame* in-place.
    """
    if meas is None:
        return

    h, w = frame.shape[:2]
    shoulder_px = meas.get('shoulder_width', 0)
    torso_px    = meas.get('torso_height',   0)
    if shoulder_px < 10:
        return   # no reliable measurement yet

    ppc = shoulder_px / _REAL_SHOULDER_CM       # pixels-per-cm
    shoulder_cm = _REAL_SHOULDER_CM
    torso_cm    = torso_px / ppc
    chest_cm    = shoulder_cm * 1.25            # rough anatomical ratio
    waist_cm    = shoulder_cm * 0.92

    size_lbl, size_col = _infer_size(shoulder_cm)

    # HUD box: bottom-right area, width=240 height=140
    bx, by = w - 250, h - 158
    bw, bh = 240, 148
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)

    # Title
    cv2.putText(frame, "BODY MEASUREMENTS", (bx + 8, by + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    rows = [
        (f"Shoulder : {shoulder_cm:.1f} cm", (0, 210, 255)),
        (f"Est. Chest: {chest_cm:.1f} cm",   (120, 220, 120)),
        (f"Est. Waist: {waist_cm:.1f} cm",   (120, 220, 120)),
        (f"Torso Ht : {torso_cm:.1f} cm",    (160, 160, 160)),
    ]
    for i, (txt, col) in enumerate(rows):
        cv2.putText(frame, txt, (bx + 8, by + 38 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    # Size badge
    cv2.rectangle(frame, (bx + 8, by + bh - 32), (bx + bw - 8, by + bh - 8),
                  size_col, -1)
    cv2.putText(frame, f"Est. Size: {size_lbl}", (bx + 16, by + bh - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Style advice HUD helper  (Step 5A)
# ---------------------------------------------------------------------------
_PALETTE_COLOR = {
    "bright":      (0, 255, 200),
    "jewel_tones": (200, 100, 255),
    "pastels":     (180, 220, 255),
    "neutrals":    (200, 200, 200),
    "bold":        (0, 120, 255),
    "any":         (160, 220, 120),
}


def draw_style_advice_hud(frame: np.ndarray, advice_list: list) -> None:
    """
    Draw top-3 style recommendations above the measurements HUD.
    Modifies *frame* in-place.
    """
    if not advice_list:
        return
    h, w = frame.shape[:2]
    rows   = min(3, len(advice_list))
    row_h  = 22
    bx, bh = w - 250, rows * row_h + 28
    by     = h - 158 - bh - 8    # sit directly above measurements HUD

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + 240, by + bh), (20, 20, 40), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (bx, by), (bx + 240, by + bh), (60, 60, 100), 1)

    cv2.putText(frame, "STYLE TIPS", (bx + 8, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 160, 255), 1, cv2.LINE_AA)

    for i, adv in enumerate(advice_list[:rows]):
        pal_name = adv.color_suggestion.value if hasattr(adv.color_suggestion, 'value') \
                   else str(adv.color_suggestion)
        col = _PALETTE_COLOR.get(pal_name, (160, 200, 160))
        label = f"{adv.category[:18]:18s}  {pal_name[:8]}"
        cv2.putText(frame, label, (bx + 8, by + 26 + (i + 1) * row_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


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
        self._last_meas    = None   # cached raw body measurements
        self._last_used_meas = None  # improved measurements (corrected torso_box)
        self._meas_age     = 0
        self._MEAS_TTL     = 5      # re-detect every 5 frames
        self._dbg_count    = 0      # controls diagnostic print rate
        self._last_loaded_garment = None  # track garment changes for TOM pre-warming
        self._init_fitter()
        self._init_pipeline()
        # Phase 3: per-joint landmark smoother (velocity-adaptive EMA)
        try:
            from src.core.landmark_smoother import LandmarkSmoother
            self._landmark_smoother = LandmarkSmoother()
            log.info("Phase 3: LandmarkSmoother active")
        except Exception as _ls_err:
            self._landmark_smoother = None
            log.debug(f"LandmarkSmoother unavailable: {_ls_err}")
        # Phase 4A: multi-garment layer manager
        try:
            sys.path.insert(0, str(Path(__file__).parent / "python-ml" / "src"))
            from multi_garment_system import (LayerManager, LayerType,
                                              ProductCategory, GarmentLayer)
            self._layer_manager = LayerManager()
            self._LayerType = LayerType
            self._ProductCategory = ProductCategory
            self._GarmentLayer = GarmentLayer
            self._multi_garment_available = True
            log.info("Phase 4A: LayerManager active  (L = add layer, U = clear)")
        except Exception as _mg_err:
            self._layer_manager = None
            self._multi_garment_available = False
            log.debug(f"LayerManager unavailable: {_mg_err}")
        self._extra_layers: list = []   # filenames stacked as outer garments
        # Phase 5A: style recommender
        try:
            sys.path.insert(0, str(Path(__file__).parent / "python-ml" / "src"))
            from style_recommender import StyleRecommender, BodyMeasurementProfile
            self._style_recommender = StyleRecommender()
            self._StyleBodyProfile  = BodyMeasurementProfile
            self._style_advice: list = []
            log.info("Phase 5A: StyleRecommender active")
        except Exception as _sr_err:
            self._style_recommender = None
            self._StyleBodyProfile  = None
            self._style_advice: list = []
            log.debug(f"StyleRecommender unavailable: {_sr_err}")

    # ---- Phase 5A: recompute style advice from current measurements ----
    def refresh_style_advice(self, meas: dict) -> list:
        """Classify body shape & return top-3 style recommendation objects."""
        if self._style_recommender is None or meas is None:
            return []
        shoulder_px = meas.get('shoulder_width', 0)
        torso_px    = meas.get('torso_height',   0)
        if shoulder_px < 10:
            return []
        ppc         = shoulder_px / _REAL_SHOULDER_CM
        shoulder_cm = _REAL_SHOULDER_CM
        chest_cm    = shoulder_cm * 1.25
        torso_cm    = torso_px / ppc
        try:
            profile = self._StyleBodyProfile(
                shoulder_width_cm=shoulder_cm,
                chest_width_cm=chest_cm,
                torso_length_cm=torso_cm,
            )
            self._style_advice = self._style_recommender.recommend(profile)[:3]
        except Exception as _e:
            log.debug(f"Style advice error: {_e}")
            self._style_advice = []
        return self._style_advice

    # ---- Phase 4A: composite extra layers on top of primary garment ----
    def apply_extra_layers(self, frame: np.ndarray) -> np.ndarray:
        """Composite any extra-layered garments (e.g. jacket over shirt) onto frame."""
        if not self._extra_layers or not self._multi_garment_available:
            return frame

        # Build landmark dict from last cached measurements
        lm_dict = None
        if self._last_meas is not None:
            lms = self._last_meas.get('landmarks')
            if lms is not None:
                lm_dict = self._lm_to_dict(lms)

        for fn in self._extra_layers:
            cloth_rgb, mask = self._load(fn)
            if cloth_rgb is None:
                continue
            cloth_bgr = cv2.cvtColor(
                (cloth_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            try:
                layer = self._GarmentLayer(
                    garment_id=fn,
                    category=self._ProductCategory.JACKET,
                    layer_type=self._LayerType.OUTER,
                    z_index=3,
                    overlay_image=cloth_bgr,
                    scale_factor=1.04,   # slightly bigger to sit over primary
                    opacity=0.90,
                )
                if lm_dict is not None:
                    frame = self._layer_manager._composite_layer(frame, layer, lm_dict)
                else:
                    frame = self._layer_manager._composite_layer_simple(frame, layer)
            except Exception as _e:
                log.debug(f"Extra layer '{fn}' composite error: {_e}")
        return frame

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
                enable_tom=True,         # CP-VTON TOM: full person reconstruction
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
    def _fixed_blend(self, frame, cloth_rgb, mask, measurements=None):
        h, w = frame.shape[:2]
        # Use landmark-based torso box if available, else fixed fractions
        if measurements is not None:
            tb  = measurements.get('torso_box')
            lms = measurements.get('landmarks')
            if tb is not None:
                x1, y1, x2, y2 = tb
            elif lms is not None and len(lms) > 24:
                x1 = int(min(lms[11].x, lms[23].x) * w)
                x2 = int(max(lms[12].x, lms[24].x) * w)
                y1 = int(min(lms[11].y, lms[12].y) * h)
                y2 = int((lms[23].y + lms[24].y) / 2 * h)
            else:
                x1, x2 = int(w * FIX_X1), int(w * FIX_X2)
                y1, y2 = int(h * FIX_Y1), int(h * FIX_Y2)
        else:
            x1, x2 = int(w * FIX_X1), int(w * FIX_X2)
            y1, y2 = int(h * FIX_Y1), int(h * FIX_Y2)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
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

        # ── TOM Pre-warming: eliminate cold-start distortion ────────────
        # When garment changes, pre-warm TOM cache so first frame shows TOM quality
        # instead of distorted GMM fallback
        if filename != self._last_loaded_garment and self._pipeline is not None:
            self._pipeline.prewarm_tom_cache(cloth_rgb, mask.squeeze())
            self._last_loaded_garment = filename

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

                    # ── Anchor shirt_top to shoulder landmarks, NOT head top ──
                    # by1 is the top of the segmentation mask which includes the
                    # head, placing the garment over the face.  Use the actual
                    # shoulder landmark Y so the collar sits at shoulder level.
                    lms = measurements.get('landmarks')
                    if lms is not None and len(lms) > 12:
                        sh_y_left  = lms[11].y * h_fr
                        sh_y_right = lms[12].y * h_fr
                        sh_y = min(sh_y_left, sh_y_right)
                        offset_px = int(_wp('shoulder_y_offset_px') or 8)
                        shirt_top = max(0, int(sh_y) - offset_px)
                        # Also derive x-bounds from shoulder + hip landmarks for accuracy
                        if len(lms) > 24:
                            lx = int(min(lms[11].x, lms[23].x) * w_fr)
                            rx = int(max(lms[12].x, lms[24].x) * w_fr)
                            if rx > lx + 20:  # sanity check
                                bx1, bx2 = lx, rx
                    else:
                        shirt_top = max(0, by1 - int(bh_px * 0.08))

                    shirt_bottom = by2

                    # Widen x-extent — fraction read live from web_server
                    pad_pct = float(_wp('torso_x_pad_pct') or 0.25)
                    pad_x = int((bx2 - bx1) * pad_pct)
                    bx1   = max(0, bx1 - pad_x)
                    bx2   = min(w_fr, bx2 + pad_x)
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

        # Store improved measurements for auto-calibrator (corrected torso_box)
        self._last_used_meas = measurements

        # ── Tier 1: Neural GMM warping ───────────────────────────────────
        if self._pipeline is not None and measurements is not None:
            try:
                landmarks   = measurements['landmarks']
                mp_lm_dict  = self._lm_to_dict(landmarks)
                # Phase 3: smooth landmark positions before GMM
                if self._landmark_smoother is not None:
                    h_fr, w_fr = frame.shape[:2]
                    mp_lm_dict = self._landmark_smoother.smooth_dict(
                        mp_lm_dict, frame_shape=(h_fr, w_fr)
                    )
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
        return self._fixed_blend(frame, cloth_rgb, mask, measurements)

    def _place_warped(self, frame, result, measurements) -> np.ndarray:
        """Composite neural-warped garment onto the frame at the torso box."""
        h, w = frame.shape[:2]
        body_mask = measurements.get('body_mask')

        # ── TOM full-synthesis branch ─────────────────────────────────────
        # When the async TOM thread has produced a synthesized frame:
        #   result.synthesized is (256, 192, 3) float32 [0,1] RGB
        #   It is the *full person* reconstructed with the garment applied.
        # Composite: TOM inside body silhouette, camera frame as background.
        if result.synthesized is not None:
            log.debug(f"[Render] Using TOM synthesis output (shape: {result.synthesized.shape})")
            # Scale TOM output back to camera resolution
            synth_bgr = cv2.cvtColor(result.synthesized, cv2.COLOR_RGB2BGR)
            synth_u8  = np.clip(synth_bgr * 255, 0, 255).astype(np.uint8)
            synth_full = cv2.resize(synth_u8, (w, h), interpolation=cv2.INTER_LINEAR)

            if body_mask is not None:
                # Resize body mask to frame size
                bm = body_mask.astype(np.float32)
                if bm.shape[:2] != (h, w):
                    bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_LINEAR)
                bm = bm.squeeze()
                # Dilate: let clothing slightly overflow body boundary
                dil_px = int(_wp('mask_dilation_px') or 31)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
                bm = cv2.dilate(bm, kernel, iterations=1)
                # Soft feathered edge (sigma ≈ dil_px / 2)
                sigma = max(3, dil_px // 2) | 1   # must be odd
                bm_soft = cv2.GaussianBlur(bm, (sigma * 4 + 1, sigma * 4 + 1), sigma)
                bm_soft = np.clip(bm_soft, 0.0, 1.0)[:, :, np.newaxis]   # (H,W,1)
                # Blend
                out_f = (synth_full.astype(np.float32) * bm_soft +
                         frame.astype(np.float32) * (1.0 - bm_soft))
                return np.clip(out_f, 0, 255).astype(np.uint8)
            else:
                # No body mask: return TOM output directly (full frame)
                return synth_full

        # ── Fallback: manual GMM overlay (while TOM warms up) ────────────
        log.warning("[Render] Using GMM fallback overlay (TOM cache not ready yet)")
        torso_x1, torso_y1, torso_x2, torso_y2 = measurements['torso_box']
        # Fully normalise the torso box before any arithmetic
        if torso_x2 < torso_x1: torso_x1, torso_x2 = torso_x2, torso_x1
        if torso_y2 < torso_y1: torso_y1, torso_y2 = torso_y2, torso_y1
        torso_y1 = max(0, min(torso_y1, int(h * 0.65)))   # shoulder no lower than 65 %
        torso_y2 = max(torso_y1 + 40, min(torso_y2, h))   # clamp hip to frame edge
        torso_x1 = max(0, torso_x1)
        torso_x2 = min(w, torso_x2)

        raw_box_w = torso_x2 - torso_x1
        raw_box_h = torso_y2 - torso_y1

        # Garment must be at LEAST target_w_min_pct of frame width
        tw_min_pct = float(_wp('target_w_min_pct') or 0.55)
        tw_scale   = float(_wp('target_w_scale')   or 1.30)
        th_scale   = float(_wp('target_h_scale')   or 1.10)
        dil_px     = int(_wp('mask_dilation_px')   or 31)
        blend_body = float(_wp('mask_blend_body')  or 0.70)

        target_w = max(int(w * tw_min_pct), int(raw_box_w * tw_scale))
        target_h = max(2, int(raw_box_h * th_scale))

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

        # Softly restrict to body silhouette: dilate the mask so garment edges
        # don't get hard-clipped when segmentation underestimates the body width.
        if body_mask is not None:
            if body_mask.shape[:2] != (h, w):
                body_mask = cv2.resize(
                    body_mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            bm2d  = body_mask.squeeze().astype(np.float32)
            # Dilate body mask — amount read live from web_server
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
            bm2d   = cv2.dilate(bm2d, kernel, iterations=1)
            bm_roi = bm2d[gy1:gy2, gx1:gx2]
            if bm_roi.ndim == 2:
                bm_roi = bm_roi[:, :, np.newaxis]
            # Blend body-masked + free — amount read live from web_server
            wm = wm * (bm_roi * blend_body + (1.0 - blend_body))

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

    panel         = GarmentPanel()
    renderer      = GarmentRenderer()
    sku_logger    = SKUSessionLogger()
    calibrator    = AutoCalibrator(enabled=True)
    prev_sel      = None
    show_skeleton = False   # toggle with K

    # ── Web debug server (http://localhost:5050) ──────────────────
    web = WebServer(port=5050)
    web.register_garment_list(lambda: panel.filenames)
    def _web_garment_cb(name: str):
        idx = panel.filenames.index(name) if name in panel.filenames else -1
        if idx >= 0:
            panel._idx = idx
    web.register_garment_callback(_web_garment_cb)
    def _web_param_cb(updates):
        nonlocal show_skeleton
        if "show_skeleton" in updates:
            show_skeleton = bool(updates["show_skeleton"])
    web.register_param_callback(_web_param_cb)
    web.start()

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

        # Phase 4A: composite extra layers (multi-garment stacking)
        if renderer._extra_layers:
            frame = renderer.apply_extra_layers(frame)

        # Phase 5: update style advice + SKU logger when garment changes
        if sel != prev_sel:
            calibrator.reset(f"garment → {sel}")
            body_shape = "UNKNOWN"
            if renderer._style_recommender is not None and renderer._last_meas is not None:
                renderer.refresh_style_advice(renderer._last_meas)
                if renderer._style_advice:
                    body_shape = renderer._style_advice[0].body_shape.value
            sku_logger.on_change(sel or "", body_shape)
            # Phase 6: upload to cloud backend (best-effort, non-blocking)
            try_upload_measurement(renderer._last_meas, sel or "", body_shape)
            prev_sel = sel

        # ── Skeleton / landmark overlay (press K to toggle) ──────
        if show_skeleton and renderer._last_meas is not None:
            draw_skeleton_overlay(frame, renderer._last_meas)

        # ── FPS + garment label on left panel ────────────────
        dt = time.perf_counter() - t0
        fps_buf.append(dt)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

        cv2.rectangle(frame, (0, 0), (400, 70), (0, 0, 0), -1)   # status bar bg
        cv2.putText(frame, f"FPS: {fps:.0f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
        layer_badge = f"  +{len(renderer._extra_layers)} layer(s)" if renderer._extra_layers else ""
        cv2.putText(frame, f"Wearing: {sel or 'none'}{layer_badge}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        skel_hint = "[K=skeleton ON] " if show_skeleton else ""
        cv2.putText(frame,
                    f"{skel_hint}W-S-A-D=select  L=layer  U=clear  K=skeleton  Q=quit",
                    (10, CAM_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

        # Phase 5A: style advice HUD (above measurements)
        draw_style_advice_hud(frame, renderer._style_advice)

        # Phase 4B: Measurements HUD (bottom-right corner)
        draw_measurements_hud(frame, renderer._last_meas)

        # ── Auto-calibrator tick (uses improved measurements with corrected torso_box) ──
        calibrator.tick(renderer._last_used_meas, frame)
        q = calibrator.quality
        if q:
            score  = q.get("total", 0)
            smooth = calibrator.smooth_total
            locked = calibrator._locked
            # Animated progress bar at bottom of frame
            bar_w  = int(CAM_W * smooth)
            bar_col = (0, 230, 100) if locked else (0, 180, 255)
            cv2.rectangle(frame, (0, CAM_H - 6), (bar_w, CAM_H), bar_col, -1)
            status = "PERFECT FIT" if locked else f"Perfecting fit… {int(smooth*100)}%"
            col_txt = (0, 230, 100) if locked else (0, 180, 255)
            cv2.putText(frame, status, (12, CAM_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_txt, 1, cv2.LINE_AA)

        # ── Push to web debug server ──────────────────────────
        web.push_frame(frame)
        web.push_state(fps, sel or "", renderer._last_meas)

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
        elif key == ord('l'):               # L  — add current garment as extra layer
            if sel and sel not in renderer._extra_layers:
                renderer._extra_layers.append(sel)
                log.info(f"Layer added: {sel}  (total layers: {len(renderer._extra_layers)})")
            elif sel:
                log.info(f"'{sel}' is already a layer")
        elif key == ord('u'):               # U  — clear all extra layers
            renderer._extra_layers.clear()
            log.info("All extra layers cleared")
        elif key == ord('k'):               # K  — toggle skeleton overlay
            show_skeleton = not show_skeleton
            log.info(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")

    sku_logger.flush()
    cap.release()
    cv2.destroyAllWindows()
    log.info("Bye!")


if __name__ == "__main__":
    main()

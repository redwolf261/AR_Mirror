"""
auto_calibrator.py — CMA-ES powered self-perfecting parameter optimizer for AR Mirror.

Every render frame it:
  1. Scores the rendered frame (HSV pixel-vision + MediaPipe landmark geometry)
  2. Feeds the combined score into a CMA-ES black-box optimizer
  3. CMA-ES proposes _CMA_POPSIZE candidate parameter sets per generation
  4. Each candidate is evaluated for _CMA_EVAL_FRAMES frames, then ranked
  5. Automatically locks when converged; unlocks if person moves / garment changes

All 5 placement params are tuned jointly in normalised [0,1] space,
so CMA-ES handles their correlations and scales automatically.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Import lazily so this module doesn't hard-require web_server at load time
_ws = None
def _get_ws():
    global _ws
    if _ws is None:
        try:
            import web_server as ws
            _ws = ws
        except ImportError:
            pass
    return _ws


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Landmarks (MediaPipe)
_NOSE       = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_HIP      = 23
_R_HIP      = 24

_SCORE_HISTORY    = 60     # frames for smoothing (~2 s at 30 fps)
_LOCK_THRESHOLD   = 0.85   # above this → locked
_UNLOCK_THRESHOLD = 0.68   # below this after locking → resume tuning

# Param bounds — CMA-ES explores within these
_PARAM_BOUNDS = {
    "shoulder_y_offset_px": (-15, 30),
    "torso_x_pad_pct":      (0.04, 0.50),
    "target_w_min_pct":     (0.25, 0.92),
    "target_w_scale":       (0.85, 2.0),
    "target_h_scale":       (0.85, 1.80),
}
_PARAM_NAMES = list(_PARAM_BOUNDS.keys())

# CMA-ES hyper-params
_CMA_POPSIZE    = 6    # candidates per generation
_CMA_EVAL_FRAMES = 8   # frames to evaluate each candidate  (~16 s/gen at 3 fps)
_CMA_SIGMA0     = 0.25  # initial exploration step in normalised [0,1] space


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-vision scorer  — actually looks at the rendered frame
# ─────────────────────────────────────────────────────────────────────────────

def _skin_fraction(region_bgr: np.ndarray) -> float:
    """Returns fraction of pixels that match human skin tone in HSV."""
    import cv2 as _cv2
    if region_bgr is None or region_bgr.size == 0:
        return 0.5
    hsv  = _cv2.cvtColor(region_bgr, _cv2.COLOR_BGR2HSV)
    s1   = _cv2.inRange(hsv, np.array([0,   25, 70]),  np.array([22,  180, 255]))
    s2   = _cv2.inRange(hsv, np.array([158, 25, 70]),  np.array([180, 180, 255]))
    mask = _cv2.bitwise_or(s1, s2)
    return float(mask.sum()) / (255.0 * mask.size + 1e-6)


def _score_frame_pixels(frame: np.ndarray, meas: dict) -> dict:
    """
    Analyse the actual rendered pixels to score placement quality.
    Three zones are sampled:
      1. Face zone (nose Y)     — skin should be visible (no garment on face)
      2. Collar zone (sh Y)     — garment texture should be visible (not skin)
      3. Mid-torso zone         — garment texture should cover the torso
    """
    result = {
        "face_pixel_score":   1.0,   # 1.0 = face fully visible (good)
        "collar_pixel_score": 0.5,   # 1.0 = garment at collar line (good)
        "coverage_score":     0.5,   # 1.0 = garment covers mid-torso (good)
        "px_diag": {},
    }
    if frame is None:
        return result

    lms = meas.get("landmarks") if meas else None
    if lms is None or len(lms) < 25:
        return result

    h, w = frame.shape[:2]

    def _px(idx):
        lm = lms[idx]
        return int(lm.x * w), int(lm.y * h)

    nose_x,  nose_y  = _px(0)
    lsh_x,   lsh_y   = _px(11)
    rsh_x,   rsh_y   = _px(12)
    lhip_y           = int(lms[23].y * h)
    rhip_y           = int(lms[24].y * h)
    hip_y            = (lhip_y + rhip_y) // 2
    sh_y             = min(lsh_y, rsh_y)
    x_left           = max(0, min(lsh_x, rsh_x) - 10)
    x_right          = min(w, max(lsh_x, rsh_x) + 10)

    def _strip(y_center, half_h, xl, xr):
        y0 = max(0, y_center - half_h)
        y1 = min(h, y_center + half_h)
        xl = max(0, xl); xr = min(w, xr)
        if y1 <= y0 or xr <= xl:
            return None
        return frame[y0:y1, xl:xr]

    # ── 1. Face zone: skin SHOULD be visible (no garment on face) ────────
    face_hw  = int(abs(nose_y - sh_y) * 0.25) + 8
    face_xl  = max(0, nose_x - int((x_right - x_left) * 0.35))
    face_xr  = min(w, nose_x + int((x_right - x_left) * 0.35))
    face_strip = _strip(nose_y, face_hw, face_xl, face_xr)
    skin_at_face = _skin_fraction(face_strip) if face_strip is not None else 0.5
    # High skin at face = face is visible = garment NOT covering face = GOOD
    face_pixel_score = skin_at_face

    # ── 2. Collar zone: garment SHOULD be visible (not skin) ─────────────
    collar_strip = _strip(sh_y, 12, x_left, x_right)
    skin_at_collar = _skin_fraction(collar_strip) if collar_strip is not None else 0.5
    # Low skin at collar = garment is there = GOOD
    collar_pixel_score = 1.0 - skin_at_collar

    # ── 3. Mid-torso coverage: garment should cover mid-torso ─────────────
    mid_y = (sh_y + hip_y) // 2
    mid_strip = _strip(mid_y, 15, x_left, x_right)
    skin_at_mid = _skin_fraction(mid_strip) if mid_strip is not None else 0.5
    # Low skin at mid = garment IS there = GOOD
    coverage_score = 1.0 - skin_at_mid

    result.update({
        "face_pixel_score":   round(face_pixel_score,   3),
        "collar_pixel_score": round(collar_pixel_score, 3),
        "coverage_score":     round(coverage_score,     3),
        "px_diag": {
            "skin_at_face":   round(skin_at_face,   3),
            "skin_at_collar": round(skin_at_collar, 3),
            "skin_at_mid":    round(skin_at_mid,    3),
        },
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Geometric scorer  (landmark bounding-box based)
# ─────────────────────────────────────────────────────────────────────────────

def _score_placement(meas: dict, frame_h: int, frame_w: int) -> dict:
    """
    Return a dict of quality sub-scores given the latest measurements.
    Each sub-score is in [0, 1].  `total` is the overall quality.
    """
    result = {
        "collar_score":  0.0,
        "width_score":   0.0,
        "height_score":  0.0,
        "face_clear":    0.0,
        "total":         0.0,
        "diagnostics": {},
    }

    lms = meas.get("landmarks")
    tb  = meas.get("torso_box")
    if lms is None or tb is None or len(lms) < 25:
        return result

    # ── Landmark pixel coords ──────────────────────────────────────
    def _px(idx):
        lm = lms[idx]
        return lm.x * frame_w, lm.y * frame_h

    nose_x,   nose_y   = _px(_NOSE)
    lsh_x,    lsh_y    = _px(_L_SHOULDER)
    rsh_x,    rsh_y    = _px(_R_SHOULDER)
    lhip_x,   lhip_y   = _px(_L_HIP)
    rhip_x,   rhip_y   = _px(_R_HIP)

    sh_y     = min(lsh_y, rsh_y)           # shoulder top (px)
    hip_y    = (lhip_y + rhip_y) / 2       # hip midpoint (px)
    sh_span  = abs(rsh_x - lsh_x)          # shoulder width (px)
    torso_h  = hip_y - sh_y                # torso height (px)

    tx1, ty1, tx2, ty2 = tb
    placed_top   = ty1
    placed_w     = tx2 - tx1
    placed_h     = ty2 - ty1

    # ── Collar score: shirt_top vs shoulder Y ───────────────────────
    # ideal: ty1 == sh_y  (allow ±5 px)
    collar_err_px = placed_top - sh_y       # positive → placed too LOW
    collar_score  = math.exp(-abs(collar_err_px) / max(torso_h * 0.15, 20))

    # ── Width score: placed_w vs ideal shoulder span + padding ─────
    ideal_w_px  = sh_span * 1.50            # garment should be ~1.5× shoulder span
    width_ratio = placed_w / max(ideal_w_px, 1)
    # penalise if too narrow or too wide
    width_score = math.exp(-0.5 * ((width_ratio - 1.0) ** 2) / (0.25 ** 2))

    # ── Height score ────────────────────────────────────────────────
    ideal_h_px   = torso_h * 1.15
    height_ratio = placed_h / max(ideal_h_px, 1)
    height_score = math.exp(-0.5 * ((height_ratio - 1.0) ** 2) / (0.30 ** 2))

    # ── Face-clear score ────────────────────────────────────────────
    # Shirt must not start above nose_y
    overlap_px  = nose_y - placed_top    # positive → garment top is above nose
    face_clear  = 1.0 if overlap_px <= 0 else math.exp(-overlap_px / 30.0)

    # Geometric total (landmarks only)
    geo_total = (collar_score ** 1.5) * (width_score ** 0.8) * \
                (height_score ** 0.5) * (face_clear ** 2.0)
    geo_total = min(1.0, max(0.0, geo_total))

    result.update({
        "collar_score":  round(collar_score, 3),
        "width_score":   round(width_score,  3),
        "height_score":  round(height_score, 3),
        "face_clear":    round(face_clear,   3),
        "geo_total":     round(geo_total,    3),
        "total":         round(geo_total,    3),   # overwritten if pixel scores available
        "diagnostics": {
            "collar_err_px":  round(collar_err_px, 1),
            "width_ratio":    round(width_ratio,   3),
            "height_ratio":   round(height_ratio,  3),
            "overlap_px":     round(max(0, overlap_px), 1),
            "sh_y":           round(sh_y, 1),
            "placed_top":     placed_top,
            "sh_span":        round(sh_span, 1),
            "placed_w":       placed_w,
            "torso_h":        round(torso_h, 1),
            "placed_h":       placed_h,
        },
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES optimizer
# ─────────────────────────────────────────────────────────────────────────────

class _CMAOptimizer:
    """
    Online CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for
    joint optimisation of all 5 placement parameters.

    Protocol (frame-sequential):
      1. ask()  → _CMA_POPSIZE candidate param-sets
      2. Apply each candidate to the renderer for _CMA_EVAL_FRAMES frames
      3. Record the mean quality score for each candidate
      4. tell() → CMA-ES updates its distribution
      5. Apply the best candidate and start the next generation

    All params are worked in normalised [0, 1] space so CMA-ES treats
    them uniformly regardless of scale.
    """

    def __init__(self) -> None:
        self._es:          object        = None   # cma.CMAEvolutionStrategy
        self._solutions:   list          = []     # [(x_vec, params_dict, [scores])]
        self._cand_idx:    int           = 0
        self._eval_frame:  int           = 0
        self._generation:  int           = 0
        self._best_params: Optional[dict] = None
        self._best_score:  float          = 0.0

    # ── normalisation ──────────────────────────────────────────────────────
    def _norm(self, params: dict) -> list:
        """Map each param to [0, 1] within its bounds."""
        out = []
        for name, (lo, hi) in _PARAM_BOUNDS.items():
            v = params.get(name, (lo + hi) / 2)
            out.append(float(_clamp((v - lo) / (hi - lo), 0.0, 1.0)))
        return out

    def _denorm(self, x: list) -> dict:
        """Map normalised vector back to param dict, clamped and rounded."""
        result: dict = {}
        for i, (name, (lo, hi)) in enumerate(_PARAM_BOUNDS.items()):
            v = lo + float(x[i]) * (hi - lo)
            v = _clamp(v, lo, hi)
            result[name] = round(v, 1) if name == "shoulder_y_offset_px" else round(v, 3)
        return result

    # ── CMA initialisation ─────────────────────────────────────────────────
    def _init(self, current_params: dict) -> None:
        import cma  # lazy import — only if CMA optimiser is actually used
        x0   = self._norm(current_params)
        opts = {
            "popsize":   _CMA_POPSIZE,
            "bounds":    [[0.0] * len(_PARAM_NAMES), [1.0] * len(_PARAM_NAMES)],
            "verbose":   -9,          # fully silent
            "maxiter":   500,         # max generations before hard-stopping
            # Do NOT set tolx/tolfun — they trigger on all-zero-score generations
        }
        self._es = cma.CMAEvolutionStrategy(x0, _CMA_SIGMA0, opts)
        self._start_generation()

    def _start_generation(self) -> None:
        xs = self._es.ask()
        self._solutions  = [(x, self._denorm(x), []) for x in xs]
        self._cand_idx   = 0
        self._eval_frame = 0

    # ── per-frame step ─────────────────────────────────────────────────────
    def step(self, current_params: dict, score: float, ws) -> None:
        """Feed one frame's quality score; applies new params to ws when ready."""
        if self._es is None:
            self._init(current_params)
            ws.set_params_from_web(self._solutions[0][1])
            log.info(f"[CMA] Generation 0 — {len(self._solutions)} candidates "
                     f"× {_CMA_EVAL_FRAMES} real-body frames each")
            return

        # Safety guard: if solutions list is empty (post-converge), re-init
        if not self._solutions or self._cand_idx >= len(self._solutions):
            warm = self._best_params or current_params
            log.info("[CMA] Re-initialising from best known params (warm start)")
            self._init(warm)
            ws.set_params_from_web(self._solutions[0][1])
            return

        # Only count frames where body is actually visible (score > 0)
        # Zero-score frames mean no body detected — don’t pollute candidate evaluation
        if score > 0.0:
            self._solutions[self._cand_idx][2].append(score)
            self._eval_frame += 1

        if self._eval_frame < _CMA_EVAL_FRAMES:
            return   # still gathering data for this candidate

        # Advance to next candidate
        self._cand_idx  += 1
        self._eval_frame = 0

        if self._cand_idx < len(self._solutions):
            ws.set_params_from_web(self._solutions[self._cand_idx][1])
            return

        # ── All candidates evaluated → tell CMA-ES ──────────────────────
        xs      = [s[0] for s in self._solutions]
        scores  = [float(np.mean(s[2])) if s[2] else 0.0 for s in self._solutions]
        fitvals = [-s for s in scores]   # CMA-ES minimises; we maximise score

        try:
            self._es.tell(xs, fitvals)
        except Exception as exc:
            log.warning(f"[CMA] tell() error: {exc}")

        best_i = int(np.argmax(scores))
        best_s = scores[best_i]
        log.info(
            f"[CMA] Gen {self._generation} complete — "
            f"best={best_s:.3f}  scores={[round(s, 3) for s in scores]}"
        )

        if best_s > self._best_score:
            self._best_score  = best_s
            self._best_params = dict(self._solutions[best_i][1])

        # Apply best from this generation
        ws.set_params_from_web(self._solutions[best_i][1])
        self._generation += 1

        if self._es.stop():
            log.info(f"[CMA] Converged at generation {self._generation}")
            if self._best_params:
                ws.set_params_from_web(self._best_params)
            # Reset _es so next call warm-restarts from best known position
            self._es = None
            self._solutions = []
            return

        self._start_generation()

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self, current_params: dict) -> None:
        """Re-initialise around the supplied starting point."""
        self._es        = None
        self._solutions = []
        self._cand_idx  = 0
        self._eval_frame = 0
        # Keep generation counter and best knowledge across resets
        if current_params:
            self._init(current_params)


# ─────────────────────────────────────────────────────────────────────────────
# AutoCalibrator  (public class, used by tryon_selector)
# ─────────────────────────────────────────────────────────────────────────────

class AutoCalibrator:
    """
    Stand in front of the camera — this class does the rest.

    Every render frame it:
      1. Samples the actual pixels to see if the garment looks right
      2. Cross-checks with MediaPipe landmark geometry
      3. Nudges ALL relevant params toward the perfect value
      4. Automatically unlocks and re-tunes if you move or the garment changes
    """

    def __init__(self, enabled: bool = True):
        self.enabled       = enabled
        self._history:     deque = deque(maxlen=_SCORE_HISTORY)
        self._scores:      dict  = {}
        self._locked:      bool  = False
        self._n_frames:    int   = 0
        self._peak_smooth: float = 0.0
        self._opt = _CMAOptimizer()   # CMA-ES parameter optimizer

    @property
    def quality(self) -> dict:
        return dict(self._scores)

    @property
    def smooth_total(self) -> float:
        if not self._history:
            return 0.0
        return sum(s["total"] for s in self._history) / len(self._history)

    def tick(self, meas: Optional[dict], frame: np.ndarray) -> None:
        """
        Main self-perfecting loop.  Call once per render frame.
        Looks at actual pixels + landmark geometry to score the fit,
        then automatically adjusts all placement params.
        """
        if not self.enabled or meas is None or frame is None:
            return

        ws = _get_ws()
        if ws is None:
            return

        self._n_frames += 1
        frame_h, frame_w = frame.shape[:2]

        # ── Score: pixel vision + landmark geometry ────────────────────────
        geo_scores = _score_placement(meas, frame_h, frame_w)

        # Guard: no diagnostics means no landmarks — don't adjust blindly
        has_landmarks = bool(geo_scores.get("diagnostics"))

        try:
            px_scores = _score_frame_pixels(frame, meas)
        except Exception as _e:
            log.debug(f"[AutoCal] pixel scoring failed: {_e}")
            px_scores = {"face_pixel_score": 1.0, "collar_pixel_score": 0.5,
                         "coverage_score": 0.5, "px_diag": {}}

        geo_total = geo_scores.get("geo_total", 0.0)
        px_total  = (px_scores["face_pixel_score"] ** 2.5 *
                     px_scores["collar_pixel_score"] ** 1.0 *
                     px_scores["coverage_score"]     ** 0.5)
        combined  = round(math.sqrt(max(0.0, geo_total) * max(0.0, px_total)), 3)

        scores = dict(geo_scores)
        scores.update({
            "face_pixel_score":   px_scores["face_pixel_score"],
            "collar_pixel_score": px_scores["collar_pixel_score"],
            "coverage_score":     px_scores["coverage_score"],
            "px_diag":            px_scores.get("px_diag", {}),
            "total":              combined,
            "cma_gen":            self._opt._generation,
            "cma_cand":           self._opt._cand_idx,
        })

        self._scores = scores
        self._history.append(scores)

        smooth = self.smooth_total
        if smooth > self._peak_smooth:
            self._peak_smooth = smooth

        # ── Lock / auto-unlock ─────────────────────────────────────────────
        if not self._locked and smooth >= _LOCK_THRESHOLD:
            self._locked = True
            log.info(f"[AutoCal] ✓ LOCKED at smooth={smooth:.2f}")
        elif self._locked and smooth < _UNLOCK_THRESHOLD:
            self._locked = False
            self._history.clear()
            self._opt.reset(self._current_params(ws))
            log.info(f"[AutoCal] ↻ UNLOCKED (score dropped to {smooth:.2f}), re-tuning…")

        ws.patch_state({
            "quality":        scores,
            "quality_smooth": round(smooth, 3),
            "auto_locked":    self._locked,
            "cma_gen":        self._opt._generation,
        })

        if self._locked or not has_landmarks:
            return

        # ── Feed score into CMA-ES ─────────────────────────────────────────
        self._opt.step(self._current_params(ws), combined, ws)

    def _current_params(self, ws) -> dict:
        """Read current live params from web_server (thread-safe)."""
        with ws._params_lock:
            return {k: ws._params[k] for k in _PARAM_NAMES if k in ws._params}

    def reset(self, reason: str = ""):
        """Call when garment changes or person leaves frame — restart tuning."""
        self._history.clear()
        self._locked      = False
        self._n_frames    = 0
        self._peak_smooth = 0.0
        ws = _get_ws()
        cp = self._current_params(ws) if ws is not None else {}
        self._opt.reset(cp)
        if reason:
            log.info(f"[AutoCal] reset ({reason})")

"""
Synthetic data generator — Phase 1.5 / Phase 2 bridge.

Generates (ratios, height_cm) → (collar, jacket, trouser_waist) training rows
from ANSUR II-calibrated body distributions WITHOUT requiring a 3D body model.

Pipeline per sample
-------------------
1.  Sample correlated body dims from ANSUR II multivariate normal
    (height, shoulder_cm, chest_cm, waist_cm, neck_cm)
2.  Simulate a 640×480 landscape webcam front-view and a side-view image,
    placing a stick-figure skeleton at realistic body proportions.
3.  Apply RTMPose-calibrated joint noise (σ ≈ 0.015 normalised).
4.  Compute BodyProxyMeasurements ratios via the live code path.
5.  Derive TRUE size labels directly from physical measurements:
      collar      ← neck_cm / 2.54, rounded to nearest 0.5"
      jacket      ← chest_cm / 2.54, rounded to nearest 2"
      trouser_waist ← waist_cm / 2.54, rounded to nearest 2"

Output JSONL schema (one line per sample)
------------------------------------------
{
  "record_type":          "synthetic",
  "shoulder_width_ratio": 0.55,
  "chest_width_ratio":    0.58,
  "hip_width_ratio":      0.42,
  "torso_depth_ratio":    0.12,
  "shoulder_depth_ratio": 0.10,
  "torso_height_px":      0.25,
  "height_cm":            175.0,
  "height_norm":          0.0,
  "collar":               "16.0",
  "jacket":               "42",
  "trouser_waist":        "34",
  "_phys": {              // physical measurements kept for inspection
    "shoulder_cm": ..., "chest_cm": ..., "waist_cm": ..., "neck_cm": ...
  }
}

Usage
-----
    python -m fitengine.data_gen.synthetic --n 50000 \\
        --output data/synthetic_train.jsonl [--seed 42] [--stats]

Requirements: numpy only (no torch / STAR / SMPL).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSUR II statistics — 10-dimensional body model
# Variables: [height, shoulder, chest, waist, neck,
#             arm_length, inseam, head_circ, hip_circ, elbow_breadth]
# ---------------------------------------------------------------------------

#                   h     sh    ch    wa    ne    al    is    hc    hic   eb
_ANSUR_MEAN = np.array(
    [175.0, 43.2, 99.0, 89.0, 39.3, 60.5, 81.5, 57.5, 98.5, 7.10],
    dtype=np.float64,
)
_ANSUR_STD  = np.array(
    [  8.0,  4.0,  9.0, 12.0,  2.5,  3.5,  4.0,  1.8,  8.5,  0.55],
    dtype=np.float64,
)

# 10×10 correlation matrix (ANSUR II + literature estimates)
#            h     sh    ch    wa    ne    al    is    hc    hic   eb
_ANSUR_CORR = np.array([
    [1.00, 0.45, 0.35, 0.15, 0.30, 0.70, 0.65, 0.35, 0.20, 0.40],  # height
    [0.45, 1.00, 0.70, 0.45, 0.65, 0.45, 0.35, 0.30, 0.40, 0.60],  # shoulder
    [0.35, 0.70, 1.00, 0.65, 0.70, 0.38, 0.28, 0.28, 0.65, 0.55],  # chest
    [0.15, 0.45, 0.65, 1.00, 0.55, 0.20, 0.22, 0.15, 0.80, 0.40],  # waist
    [0.30, 0.65, 0.70, 0.55, 1.00, 0.32, 0.25, 0.42, 0.50, 0.50],  # neck
    [0.70, 0.45, 0.38, 0.20, 0.32, 1.00, 0.55, 0.30, 0.18, 0.45],  # arm_length
    [0.65, 0.35, 0.28, 0.22, 0.25, 0.55, 1.00, 0.28, 0.20, 0.35],  # inseam
    [0.35, 0.30, 0.28, 0.15, 0.42, 0.30, 0.28, 1.00, 0.20, 0.30],  # head_circ
    [0.20, 0.40, 0.65, 0.80, 0.50, 0.18, 0.20, 0.20, 1.00, 0.38],  # hip_circ
    [0.40, 0.60, 0.55, 0.40, 0.50, 0.45, 0.35, 0.30, 0.38, 1.00],  # elbow_breadth
], dtype=np.float64)

_CHOL = np.linalg.cholesky(_ANSUR_CORR)

# Hard clip [min, max] per variable
_ANSUR_CLIP = np.array([
    [155.0, 210.0],   # height (cm)
    [ 34.0,  56.0],   # shoulder (cm)
    [ 78.0, 135.0],   # chest circumference (cm)
    [ 62.0, 135.0],   # waist circumference (cm)
    [ 33.0,  48.0],   # neck circumference (cm)
    [ 50.0,  75.0],   # arm length shoulder→wrist (cm)
    [ 68.0,  98.0],   # inseam hip→floor (cm)
    [ 53.0,  62.0],   # head circumference (cm)
    [ 78.0, 130.0],   # hip circumference (cm)
    [  5.5,   9.5],   # elbow breadth (cm)
])

# Female body offset (applied to 20% of samples)
# Women: narrower shoulders, slightly shorter, wider hips, smaller chest
_FEMALE_DELTA_MEAN = np.array(
    [-7.0, -4.5, -8.0, -3.0, -3.5, -4.5, -4.5,  -1.0,  +5.0, -0.5],
    dtype=np.float64,
)
_FEMALE_DELTA_STD = np.array(
    [0.0,  0.5,  1.5,  2.0,  0.3,  0.5,  0.5,   0.2,   1.5,  0.1],
    dtype=np.float64,
)

# Body-proportion fractions from the TOP of the head
_FRAC = {
    "nose":     0.08,   # nose tip
    "eye":      0.07,   # eye level (slightly above nose)
    "ear":      0.08,   # ear level
    "shoulder": 0.20,   # shoulder joint (biacromial)
    "elbow":    0.38,   # elbow
    "wrist":    0.53,   # wrist
    "hip":      0.53,   # hip joint (greater trochanter)
    "knee":     0.73,   # knee
    "ankle":    0.93,   # ankle
    "heel":     0.97,
    "toe":      0.98,
}

# Body-33 joint indices we use in BodyProxyMeasurements
_KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8,
    "l_wr": 9, "r_wr": 10, "l_hip": 11, "r_hip": 12,
    "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16,
    "l_heel": 17, "r_heel": 18, "l_ftoe": 19, "r_ftoe": 20,
    "neck": 27,    # synthesized midpoint of shoulders
    "mid_hip": 32, # synthesized midpoint of hips
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _snap_collar(neck_inches: float) -> str:
    """Round neck circumference in inches to nearest 0.5" collar size."""
    snapped = round(neck_inches * 2) / 2.0
    snapped = max(14.0, min(17.5, snapped))
    return f"{snapped:.1f}"


def _snap_jacket(chest_inches: float) -> str:
    """Round chest in inches to nearest 2" jacket size (round-half-up)."""
    snapped = math.floor(chest_inches / 2.0 + 0.5) * 2
    snapped = max(36, min(50, snapped))
    return str(int(snapped))


def _snap_trouser(waist_inches: float) -> str:
    """Round waist in inches to nearest 2" trouser size (round-half-up)."""
    snapped = math.floor(waist_inches / 2.0 + 0.5) * 2
    snapped = max(28, min(46, snapped))
    return str(int(snapped))


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Phase 1.5 keypoint-ratio synthetic data generator.

    Produces training rows for a lightweight regressor that replaces the
    heuristic estimator.  No 3D body model required.

    Args:
        seed       : random seed for reproducibility.
        h_over_w   : webcam aspect ratio H/W.  None = sample per image
                     (0.75 for landscape, 1.78 for portrait, varying).
        portrait_frac : fraction of samples that use a portrait device.
    """

    def __init__(
        self,
        seed:          Optional[int] = None,
        h_over_w:      Optional[float] = None,
        portrait_frac: float = 0.25,
        full_body:     bool = False,
    ) -> None:
        self._rng      = np.random.default_rng(seed)
        self._h_w      = h_over_w
        self._p_frac   = portrait_frac
        self._full_body = full_body   # if True, skip crop variation (full-body only)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _sample_bodies(self, n: int) -> np.ndarray:
        """Return [n, 10] physical measurements with ANSUR-calibrated covariance."""
        z  = self._rng.standard_normal((n, 10))
        zc = z @ _CHOL.T
        vals = _ANSUR_MEAN + zc * _ANSUR_STD
        for i in range(10):
            vals[:, i] = np.clip(vals[:, i], _ANSUR_CLIP[i, 0], _ANSUR_CLIP[i, 1])
        return vals

    def _place_skeleton(
        self,
        height_cm:     float,
        shoulder_cm:   float,
        hip_width_cm:  float,
        fill:          float,
        h_over_w:      float,
        view:          str   = "front",
        arm_length_cm: float = 60.5,
        inseam_cm:     float = 81.5,
        head_circ_cm:  float = 57.5,
        elbow_breadth_cm: float = 7.1,
    ) -> np.ndarray:
        """
        Return body-33 keypoints [33, 3] (x, y, conf) in normalised [0,1] coords.
        Joint positions derived from actual body dimensions, NOT fixed fractions.
        """
        y_offset = self._rng.uniform(0.0, max(0.01, 1.0 - fill))
        y_top    = y_offset

        def _y(frac: float) -> float:
            return y_top + frac * fill

        px_per_cm_x = fill * h_over_w / height_cm

        kp = np.zeros((33, 3), dtype=np.float32)

        def _conf(base: float = 0.85) -> float:
            return float(np.clip(self._rng.normal(base, 0.06), 0.40, 0.98))

        # Derived fractions from real body dimensions
        sh_frac     = _FRAC["shoulder"]
        hip_frac    = _FRAC["hip"]
        # arm: elbow at shoulder + 0.47 * arm, wrist at shoulder + arm_length
        arm_len_frac    = (arm_length_cm / height_cm)
        elbow_frac      = sh_frac + arm_len_frac * 0.47
        wrist_frac      = sh_frac + arm_len_frac
        # leg: knee at hip + 0.50 * inseam, ankle at hip + 0.945 * inseam
        inseam_frac = inseam_cm / height_cm
        knee_frac   = hip_frac + inseam_frac * 0.50
        ankle_frac  = hip_frac + inseam_frac * 0.945
        heel_frac   = ankle_frac + 0.005
        toe_frac    = heel_frac  + 0.010

        if view == "front":
            sh_half  = (shoulder_cm / 2.0) * px_per_cm_x
            hip_half = hip_width_cm * px_per_cm_x
            cx = 0.50 + self._rng.normal(0.0, 0.025)

            # Head — ear width from head circumference
            head_half = (head_circ_cm / (2 * math.pi)) * px_per_cm_x * 0.85
            for ji, frac in [
                (_KP["nose"],  _FRAC["nose"]),
                (_KP["l_eye"], _FRAC["eye"]), (_KP["r_eye"], _FRAC["eye"]),
            ]:
                kp[ji] = [cx, _y(frac), _conf(0.80)]
            kp[_KP["l_ear"]] = [cx - head_half, _y(_FRAC["ear"]), _conf(0.75)]
            kp[_KP["r_ear"]] = [cx + head_half, _y(_FRAC["ear"]), _conf(0.75)]

            # Shoulders
            y_sh = _y(sh_frac)
            kp[_KP["l_sh"]] = [cx - sh_half, y_sh, _conf()]
            kp[_KP["r_sh"]] = [cx + sh_half, y_sh, _conf()]

            # Hips
            y_hip = _y(hip_frac)
            kp[_KP["l_hip"]] = [cx - hip_half, y_hip, _conf()]
            kp[_KP["r_hip"]] = [cx + hip_half, y_hip, _conf()]

            # Arms — random angle variation gives independent elbow/wrist spread
            # arm_angle: 0 = straight down, + = arms spread outward
            upper_arm_cm = arm_length_cm * 0.47
            lower_arm_cm = arm_length_cm * 0.53

            for side, sign in (("l", -1), ("r", +1)):
                angle_upper = self._rng.uniform(-5.0, 30.0)   # deg from vertical
                angle_lower = angle_upper + self._rng.uniform(-10.0, 20.0)
                a_up  = math.radians(angle_upper)
                a_lo  = math.radians(angle_lower)

                sh_x = cx + sign * sh_half
                el_x = sh_x + sign * (upper_arm_cm * math.sin(a_up) * px_per_cm_x)
                el_y = y_sh  + upper_arm_cm * math.cos(a_up) * px_per_cm_x / h_over_w
                wr_x = el_x  + sign * (lower_arm_cm * math.sin(a_lo) * px_per_cm_x)
                wr_y = el_y  + lower_arm_cm * math.cos(a_lo) * px_per_cm_x / h_over_w

                kp[_KP[f"{side}_el"]] = [float(el_x), float(el_y), _conf(0.80)]
                kp[_KP[f"{side}_wr"]] = [float(wr_x), float(wr_y), _conf(0.78)]

            # Legs — from real inseam
            knee_half  = hip_half * 0.50 + self._rng.normal(0, hip_half * 0.04)
            ankle_half = knee_half * 0.55 + self._rng.normal(0, knee_half * 0.04)
            for (jl, jr), frac, half in [
                ((_KP["l_kn"],   _KP["r_kn"]),   knee_frac,  knee_half),
                ((_KP["l_an"],   _KP["r_an"]),   ankle_frac, ankle_half),
                ((_KP["l_heel"], _KP["r_heel"]), heel_frac,  ankle_half),
                ((_KP["l_ftoe"], _KP["r_ftoe"]), toe_frac,   ankle_half * 1.4),
            ]:
                y_j = _y(frac)
                kp[jl] = [cx - half, y_j, _conf(0.75)]
                kp[jr] = [cx + half, y_j, _conf(0.75)]

            kp[_KP["neck"]]       = (kp[_KP["l_sh"]]  + kp[_KP["r_sh"]]) / 2
            kp[_KP["neck"]][2]    = _conf(0.90)
            kp[_KP["mid_hip"]]    = (kp[_KP["l_hip"]] + kp[_KP["r_hip"]]) / 2
            kp[_KP["mid_hip"]][2] = _conf(0.90)

        else:  # side view
            depth_cm      = shoulder_cm      # chest depth passed as shoulder_cm
            depth_half_px = (depth_cm / 2.0) * px_per_cm_x
            hip_d_px      = hip_width_cm * px_per_cm_x

            cx = 0.50 + self._rng.normal(0.0, 0.03)
            y_sh  = _y(sh_frac)
            y_hip = _y(hip_frac)

            kp[_KP["l_sh"]]  = [cx - depth_half_px, y_sh,  _conf(0.80)]
            kp[_KP["r_sh"]]  = [cx + depth_half_px, y_sh,  _conf(0.70)]
            kp[_KP["l_hip"]] = [cx - hip_d_px,      y_hip, _conf(0.75)]
            kp[_KP["r_hip"]] = [cx + hip_d_px,      y_hip, _conf(0.65)]

            for ji, frac in [
                (_KP["nose"],   _FRAC["nose"]),
                (_KP["l_eye"],  _FRAC["eye"]),   (_KP["r_eye"],   _FRAC["eye"]),
                (_KP["l_ear"],  _FRAC["ear"]),   (_KP["r_ear"],   _FRAC["ear"]),
                (_KP["l_el"],   elbow_frac),      (_KP["r_el"],   elbow_frac),
                (_KP["l_wr"],   wrist_frac),      (_KP["r_wr"],   wrist_frac),
                (_KP["l_kn"],   knee_frac),       (_KP["r_kn"],   knee_frac),
                (_KP["l_an"],   ankle_frac),      (_KP["r_an"],   ankle_frac),
                (_KP["l_heel"], heel_frac),        (_KP["r_heel"], heel_frac),
                (_KP["l_ftoe"], toe_frac),         (_KP["r_ftoe"], toe_frac),
            ]:
                kp[ji] = [cx, _y(frac), _conf(0.70)]

            kp[_KP["neck"]]       = (kp[_KP["l_sh"]]  + kp[_KP["r_sh"]]) / 2
            kp[_KP["neck"]][2]    = _conf(0.85)
            kp[_KP["mid_hip"]]    = (kp[_KP["l_hip"]] + kp[_KP["r_hip"]]) / 2
            kp[_KP["mid_hip"]][2] = _conf(0.85)

        # RTMPose-calibrated joint noise
        noise   = self._rng.normal(0.0, 0.012, (33, 2)).astype(np.float32)
        kp[:, :2] += noise
        kp[:, :2]  = np.clip(kp[:, :2], 0.0, 1.0)
        return kp

    def _sample_h_over_w(self) -> float:
        """Sample the image aspect ratio H/W for this sample."""
        if self._h_w is not None:
            return self._h_w
        if self._rng.random() < self._p_frac:
            # Portrait phone (9:16 ± variation)
            return float(np.clip(self._rng.normal(1.78, 0.12), 1.40, 2.10))
        else:
            # Landscape webcam (4:3 ± variation)
            return float(np.clip(self._rng.normal(0.75, 0.05), 0.60, 0.90))

    # ── Public API ────────────────────────────────────────────────────────

    def generate_batch(self, n: int) -> list[dict]:
        """
        Generate n synthetic samples with maximum body dimension variance.

        Returns:
            List of dicts, each with the JSONL output schema.
        """
        _SRC = Path(__file__).resolve().parents[3]
        if str(_SRC) not in sys.path:
            sys.path.insert(0, str(_SRC))
        from fitengine.measurements import BodyProxyMeasurements

        bodies = self._sample_bodies(n)
        records: list[dict] = []

        for i in range(n):
            h_cm, sh_cm, ch_cm, wa_cm, ne_cm, \
                al_cm, is_cm, hc_cm, hipc_cm, eb_cm = bodies[i]

            # 20% chance of female proportions (separate size tables not needed —
            # label snapping still maps to the same collar/jacket/trouser grid)
            if self._rng.random() < 0.20:
                delta = _FEMALE_DELTA_MEAN + self._rng.standard_normal(10) * _FEMALE_DELTA_STD
                h_cm  = float(np.clip(h_cm  + delta[0], 148, 195))
                sh_cm = float(np.clip(sh_cm + delta[1],  30,  50))
                ch_cm = float(np.clip(ch_cm + delta[2],  72, 120))
                wa_cm = float(np.clip(wa_cm + delta[3],  60, 120))
                ne_cm = float(np.clip(ne_cm + delta[4],  30,  43))
                al_cm = float(np.clip(al_cm + delta[5],  46,  70))
                is_cm = float(np.clip(is_cm + delta[6],  62,  90))
                hc_cm = float(np.clip(hc_cm + delta[7],  51,  60))
                hipc_cm = float(np.clip(hipc_cm + delta[8], 80, 130))

            # Hip geometry
            hip_half = (hipc_cm / np.pi) / 2.0            # from hip_circ
            chest_depth_cm = ch_cm / np.pi                # side-view chest depth

            h_over_w   = self._sample_h_over_w()

            # Crop variation: full / 3/4 / upper-body
            # full_body=True skips partial crops — use for ratio-regressor training
            # so all leg/hip features are always cleanly visible.
            if self._full_body:
                crop_type  = "full"
                fill_front = float(self._rng.uniform(0.55, 0.88))
                fill_side  = float(self._rng.uniform(0.25, 0.75))
            else:
                crop_type  = self._rng.choice(["full", "threequarter", "upper"],
                                              p=[0.55, 0.30, 0.15])
                fill_front = float(self._rng.uniform(0.55, 0.88))
                fill_side  = float(self._rng.uniform(0.25, 0.75))

                # For upper-body crops: reduce fill so legs are off-frame
                if crop_type == "upper":
                    fill_front = float(self._rng.uniform(0.30, 0.55))
                    fill_side  = float(self._rng.uniform(0.20, 0.45))
                elif crop_type == "threequarter":
                    fill_front = float(self._rng.uniform(0.45, 0.72))

            kp_front = self._place_skeleton(
                h_cm, sh_cm, hip_half, fill_front, h_over_w,
                view="front",
                arm_length_cm=al_cm, inseam_cm=is_cm,
                head_circ_cm=hc_cm,  elbow_breadth_cm=eb_cm,
            )
            kp_side = self._place_skeleton(
                h_cm, chest_depth_cm, hip_half * 0.85, fill_side, h_over_w,
                view="side",
                arm_length_cm=al_cm, inseam_cm=is_cm,
                head_circ_cm=hc_cm,  elbow_breadth_cm=eb_cm,
            )

            m = BodyProxyMeasurements.from_keypoints(kp_front, kp_side)
            if not m.valid:
                continue

            collar_str  = _snap_collar(ne_cm  / 2.54)
            jacket_str  = _snap_jacket(ch_cm  / 2.54)
            trouser_str = _snap_trouser(wa_cm / 2.54)

            records.append({
                "record_type":          "synthetic",
                "shoulder_width_ratio": round(float(m.shoulder_width_ratio), 5),
                "chest_width_ratio":    round(float(m.chest_width_ratio),    5),
                "hip_width_ratio":      round(float(m.hip_width_ratio),      5),
                "torso_depth_ratio":    round(float(m.torso_depth_ratio),    5),
                "shoulder_depth_ratio": round(float(m.shoulder_depth_ratio), 5),
                "torso_height_px":      round(float(m.torso_height_px),      5),
                "elbow_width_ratio":    round(float(m.elbow_width_ratio),    5),
                "arm_span_ratio":       round(float(m.arm_span_ratio),       5),
                "leg_length_ratio":     round(float(m.leg_length_ratio),     5),
                "inseam_ratio":         round(float(m.inseam_ratio),         5),
                "head_width_ratio":     round(float(m.head_width_ratio),     5),
                "hip_shoulder_taper":   round(float(m.hip_shoulder_taper),   5),
                "side_hip_depth_ratio": round(float(m.side_hip_depth_ratio), 5),
                "height_cm":            round(float(h_cm), 1),
                "height_norm":          round(float((h_cm - 175.0) / 12.0), 4),
                "collar":               collar_str,
                "jacket":               jacket_str,
                "trouser_waist":        trouser_str,
                "_phys": {
                    "shoulder_cm": round(float(sh_cm), 1),
                    "chest_cm":    round(float(ch_cm), 1),
                    "waist_cm":    round(float(wa_cm), 1),
                    "neck_cm":     round(float(ne_cm), 1),
                    "arm_length_cm": round(float(al_cm), 1),
                    "inseam_cm":   round(float(is_cm), 1),
                    "head_circ_cm": round(float(hc_cm), 1),
                    "hip_circ_cm": round(float(hipc_cm), 1),
                },
            })

        return records

    def generate(self, n: int, output_path: str | Path) -> int:
        """
        Generate n samples and write to a JSONL file.

        Returns number of valid samples written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate in chunks of 10k for progress reporting
        chunk = 10_000
        written = 0
        needed  = n

        with open(output_path, "w", encoding="utf-8") as fh:
            while written < n:
                batch_n  = min(chunk, needed - written)
                records  = self.generate_batch(int(batch_n * 1.05))  # 5% spare for invalids
                records  = records[: needed - written]
                for r in records:
                    fh.write(json.dumps(r) + "\n")
                written += len(records)
                logger.info("Generated %d / %d samples …", written, n)

        logger.info("Done — wrote %d samples to %s", written, output_path)
        return written

    def print_stats(self, output_path: str | Path) -> None:
        """Print label distribution summary for a generated file."""
        from collections import Counter
        output_path = Path(output_path)
        collars, jackets, trousers = Counter(), Counter(), Counter()
        n = 0
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                collars[r["collar"]]      += 1
                jackets[r["jacket"]]      += 1
                trousers[r["trouser_waist"]] += 1
                n += 1
        print(f"\n── Distribution ({n} samples) ─────────────────────")
        print("Collar :", dict(sorted(collars.items())))
        print("Jacket :", dict(sorted(jackets.items(), key=lambda x: int(x[0]))))
        print("Trouser:", dict(sorted(trousers.items(), key=lambda x: int(x[0]))))
        # Ratio stats
        import json as _j
        ratios = {"sh": [], "ch": [], "hp": [], "dp": []}
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                r = _j.loads(line)
                ratios["sh"].append(r["shoulder_width_ratio"])
                ratios["ch"].append(r["chest_width_ratio"])
                ratios["hp"].append(r["hip_width_ratio"])
                ratios["dp"].append(r["torso_depth_ratio"])
        for k, v in ratios.items():
            arr = np.array(v)
            print(f"  {k}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
                  f"p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    # Suppress per-sample "torso too small" warnings — these are expected invalids
    # that get retried automatically; showing them floods the terminal.
    logging.getLogger("fitengine.measurements").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(
        description="Generate synthetic FitEngine training data (no 3D model required)."
    )
    parser.add_argument("--n",      type=int,   default=50_000, help="samples to generate")
    parser.add_argument("--output", type=str,   required=True,  help="output .jsonl path")
    parser.add_argument("--seed",   type=int,   default=42,     help="random seed")
    parser.add_argument("--stats",  action="store_true",        help="print distribution stats after generation")
    parser.add_argument("--landscape-only", action="store_true",
                        help="force landscape aspect ratio (H/W=0.75) for all samples")
    parser.add_argument("--full-body", action="store_true",
                        help="force full-body crops only (no upper-body/3/4 crops) — use for ratio-regressor training")
    args = parser.parse_args()

    h_over_w = 0.75 if args.landscape_only else None
    gen = SyntheticDataGenerator(seed=args.seed, h_over_w=h_over_w, full_body=args.full_body)
    gen.generate(args.n, args.output)
    if args.stats:
        gen.print_stats(args.output)


if __name__ == "__main__":
    _cli()

"""
Scale-invariant body proxy measurements from keypoints.

All ratios are normalised by torso height (shoulder-to-hip distance) so
they are independent of camera distance and image scale.  This makes the
heuristic estimator robust across different photo setups.

Body-33 landmark indices used:
  5  = left_shoulder   6  = right_shoulder
  11 = left_hip        12 = right_hip
  27 = neck            32 = mid_hip
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BodyProxyMeasurements:
    """
    Scale-invariant body ratios derived from 2-view keypoints.

    All values are relative to torso_height — positive float, dimensionless.
    A value of 0.0 indicates the measurement could not be computed
    (low-confidence or missing keypoints).
    """

    # Front-view ratios
    shoulder_width_ratio:  float = 0.0  # shoulder distance / torso height
    chest_width_ratio:     float = 0.0  # estimated chest span / torso height
    hip_width_ratio:       float = 0.0  # hip distance / torso height
    torso_height_px:       float = 0.0  # torso height in normalised px (0–1)
    elbow_width_ratio:     float = 0.0  # elbow span / torso height (lateral mass)
    arm_span_ratio:        float = 0.0  # shoulder-to-wrist Euclidean / torso (sleeve proxy)
    leg_length_ratio:      float = 0.0  # hip-to-ankle / torso (trouser length proxy)
    inseam_ratio:          float = 0.0  # hip-to-knee / torso (trouser inseam proxy)
    head_width_ratio:      float = 0.0  # ear-to-ear / torso (collar proxy)
    hip_shoulder_taper:    float = 0.0  # hip_width / shoulder_width (body shape)

    # Side-view ratios
    torso_depth_ratio:     float = 0.0  # body width at chest level (side) / torso height
    shoulder_depth_ratio:  float = 0.0  # shoulder depth (side) / torso height
    side_hip_depth_ratio:  float = 0.0  # side-view hip span / torso (waist circ. proxy)

    # Combined
    confidence:            float = 0.0  # mean of contributing joint confidences
    valid:                 bool  = False

    # Raw joint positions for debugging
    _front_kp: Optional[np.ndarray] = field(default=None, repr=False)
    _side_kp:  Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_keypoints(
        cls,
        kp_front: np.ndarray,
        kp_side:  np.ndarray,
    ) -> "BodyProxyMeasurements":
        """
        Compute ratios from normalised [33, 3] keypoint arrays.

        Args:
            kp_front : [33, 3] (x, y, conf) front view, [0, 1] normalised.
            kp_side  : [33, 3] (x, y, conf) side view,  [0, 1] normalised.

        Returns:
            BodyProxyMeasurements instance.
        """
        m = cls(_front_kp=kp_front, _side_kp=kp_side)

        # ── Front view ────────────────────────────────────────────────────
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP,      R_HIP      = 11, 12
        NECK                   = 27
        MID_HIP                = 32

        c_front = [
            kp_front[L_SHOULDER, 2], kp_front[R_SHOULDER, 2],
            kp_front[L_HIP,      2], kp_front[R_HIP,      2],
        ]

        if min(c_front) < 0.25:
            # Not enough confidence — return zero measurements
            return m

        # Torso height: neck Y to mid_hip Y (or shoulder to hip fallback)
        if kp_front[NECK, 2] > 0.25 and kp_front[MID_HIP, 2] > 0.25:
            torso_h = abs(kp_front[MID_HIP, 1] - kp_front[NECK, 1])
        else:
            ls_y = (kp_front[L_SHOULDER, 1] + kp_front[R_SHOULDER, 1]) / 2
            hip_y = (kp_front[L_HIP, 1] + kp_front[R_HIP, 1]) / 2
            torso_h = abs(hip_y - ls_y)

        if torso_h < 1e-4:
            return m

        # Sanity-check: torso height < 10% of frame height means the person
        # is too far from the camera or keypoints collapsed — unreliable ratios.
        if torso_h < 0.10:
            logger.warning(
                "torso_height_px=%.3f < 0.10 — person too small in frame or "
                "keypoint collapse detected; marking invalid.", torso_h
            )
            return m

        shoulder_w = abs(kp_front[R_SHOULDER, 0] - kp_front[L_SHOULDER, 0])
        hip_w      = abs(kp_front[R_HIP,      0] - kp_front[L_HIP,      0])
        # Chest width: slightly wider than shoulder width in most men
        chest_w = shoulder_w * 1.05

        m.shoulder_width_ratio = float(shoulder_w / torso_h)
        m.chest_width_ratio    = float(chest_w    / torso_h)
        m.hip_width_ratio      = float(hip_w      / torso_h)
        m.torso_height_px      = float(torso_h)

        # Hip-shoulder taper (derived — no extra keypoints needed)
        if shoulder_w > 1e-4:
            m.hip_shoulder_taper = float(hip_w / shoulder_w)

        # Elbow width
        L_ELBOW, R_ELBOW = 7, 8
        if kp_front[L_ELBOW, 2] > 0.20 and kp_front[R_ELBOW, 2] > 0.20:
            el_w = abs(kp_front[R_ELBOW, 0] - kp_front[L_ELBOW, 0])
            m.elbow_width_ratio = float(el_w / torso_h)

        # Arm span (shoulder → wrist, Euclidean, avg both sides)
        L_WRIST, R_WRIST = 9, 10
        arm_spans = []
        for sh_idx, wr_idx in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
            if kp_front[sh_idx, 2] > 0.20 and kp_front[wr_idx, 2] > 0.20:
                dx = kp_front[wr_idx, 0] - kp_front[sh_idx, 0]
                dy = kp_front[wr_idx, 1] - kp_front[sh_idx, 1]
                arm_spans.append(float(np.sqrt(dx*dx + dy*dy)))
        if arm_spans:
            m.arm_span_ratio = float(np.mean(arm_spans) / torso_h)

        # Leg length (hip → ankle, Y-axis, avg both sides)
        L_ANKLE, R_ANKLE = 15, 16
        leg_lens = []
        for hip_idx, ank_idx in ((L_HIP, L_ANKLE), (R_HIP, R_ANKLE)):
            if kp_front[hip_idx, 2] > 0.20 and kp_front[ank_idx, 2] > 0.20:
                leg_lens.append(abs(kp_front[ank_idx, 1] - kp_front[hip_idx, 1]))
        if leg_lens:
            m.leg_length_ratio = float(np.mean(leg_lens) / torso_h)

        # Inseam (hip → knee, Y-axis, avg both sides)
        L_KNEE, R_KNEE = 13, 14
        inseams = []
        for hip_idx, kn_idx in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
            if kp_front[hip_idx, 2] > 0.20 and kp_front[kn_idx, 2] > 0.20:
                inseams.append(abs(kp_front[kn_idx, 1] - kp_front[hip_idx, 1]))
        if inseams:
            m.inseam_ratio = float(np.mean(inseams) / torso_h)

        # Head / ear width (ear-to-ear, front view)
        L_EAR, R_EAR = 3, 4
        if kp_front[L_EAR, 2] > 0.15 and kp_front[R_EAR, 2] > 0.15:
            m.head_width_ratio = float(
                abs(kp_front[R_EAR, 0] - kp_front[L_EAR, 0]) / torso_h
            )

        # ── Side view ─────────────────────────────────────────────────────
        c_side = [kp_side[L_SHOULDER, 2], kp_side[R_SHOULDER, 2]]
        if max(c_side) > 0.20:
            # In side view one shoulder may be occluded — take max x span
            # at shoulder level as depth
            shoulder_xs = []
            if kp_side[L_SHOULDER, 2] > 0.20:
                shoulder_xs.append(kp_side[L_SHOULDER, 0])
            if kp_side[R_SHOULDER, 2] > 0.20:
                shoulder_xs.append(kp_side[R_SHOULDER, 0])
            if len(shoulder_xs) >= 2:
                m.torso_depth_ratio = float(
                    abs(shoulder_xs[1] - shoulder_xs[0]) / torso_h
                )
            # Shoulder depth: horizontal extent at shoulder-height scanline
            m.shoulder_depth_ratio = m.torso_depth_ratio

        # Side-view hip depth (same logic as torso depth but at hip level)
        hip_xs_side = []
        if kp_side[L_HIP, 2] > 0.20:
            hip_xs_side.append(kp_side[L_HIP, 0])
        if kp_side[R_HIP, 2] > 0.20:
            hip_xs_side.append(kp_side[R_HIP, 0])
        if len(hip_xs_side) >= 2:
            m.side_hip_depth_ratio = float(
                abs(hip_xs_side[1] - hip_xs_side[0]) / torso_h
            )

        # Confidence: mean of all contributing joints
        conf_vals = c_front + [max(c_side)]
        m.confidence = float(np.mean(conf_vals))
        m.valid = True
        return m

"""
Real-time frame quality checker.

AlignmentGuide analyses a body-33 keypoint array and determines whether
the pose is acceptable for a reliable size estimate.  Used by the Gradio
UI to give live feedback before auto-capture.

Body-33 indices used:
  0  = nose        5  = left_shoulder   6  = right_shoulder
  11 = left_hip   12  = right_hip      15  = left_ankle
  16 = right_ankle 27 = neck           32  = mid_hip
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Configurable thresholds — overridden by default.yaml at runtime
SHOULDER_TILT_MAX_DEG  = 8.0    # relax to 12° for dark-suit pilots
TORSO_CONF_MIN         = 0.4
BODY_ANGLE_TARGET_DEG  = 90.0
BODY_ANGLE_TOL_DEG     = 20.0
TIMEOUT_SECONDS        = 10.0


@dataclass
class CheckResult:
    acceptable: bool
    feedback: str           # empty string if acceptable


class AlignmentGuide:
    """
    Pose quality gate for 2-photo capture flow.

    Usage:
        guide = AlignmentGuide()
        ok, msg = guide.check_front(kp33)
        ok, msg = guide.check_side(kp33)
    """

    def __init__(
        self,
        shoulder_tilt_max_deg: float = SHOULDER_TILT_MAX_DEG,
        torso_conf_min:        float = TORSO_CONF_MIN,
        body_angle_target_deg: float = BODY_ANGLE_TARGET_DEG,
        body_angle_tol_deg:    float = BODY_ANGLE_TOL_DEG,
    ) -> None:
        self.shoulder_tilt_max_deg  = shoulder_tilt_max_deg
        self.torso_conf_min         = torso_conf_min
        self.body_angle_target_deg  = body_angle_target_deg
        self.body_angle_tol_deg     = body_angle_tol_deg

        self._front_stable_since: float | None = None
        self._side_stable_since:  float | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def check_front(self, kp33: np.ndarray) -> Tuple[bool, str]:
        """
        Evaluate front-view frame quality.

        Checks (in priority order):
          1. Torso joint confidence ≥ threshold → "IMPROVE LIGHTING"
          2. Shoulder tilt < max → "STAND STRAIGHT"
          3. Hip midpoint near horizontal centre → "CENTER yourself"
          4. Feet visible (ankle keypoints present) → "STEP BACK"
          5. Arms not touching torso → "Move arms away from body"

        Returns:
            (is_acceptable, feedback_string)
        """
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP,      R_HIP      = 11, 12
        L_ANKLE,    R_ANKLE    = 15, 16
        L_WRIST,    R_WRIST    =  9, 10

        # 1. Lighting / confidence
        torso_kps = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        min_conf = min(float(kp33[i, 2]) for i in torso_kps)
        if min_conf < self.torso_conf_min:
            return False, "IMPROVE LIGHTING — move to a brighter spot"

        # 2. Shoulder tilt
        dy = float(kp33[R_SHOULDER, 1] - kp33[L_SHOULDER, 1])
        dx = float(kp33[R_SHOULDER, 0] - kp33[L_SHOULDER, 0])
        if abs(dx) < 1e-6:
            tilt_deg = 90.0
        else:
            tilt_deg = abs(math.degrees(math.atan(dy / dx)))

        if tilt_deg > self.shoulder_tilt_max_deg:
            return False, "STAND STRAIGHT — shoulders level"

        # 3. Hip centering (mid_hip x near 0.5)
        hip_mx = (float(kp33[L_HIP, 0]) + float(kp33[R_HIP, 0])) / 2.0
        if kp33[L_HIP, 2] > 0.3 and kp33[R_HIP, 2] > 0.3:
            if abs(hip_mx - 0.5) > 0.15:
                direction = "left" if hip_mx > 0.5 else "right"
                return False, f"CENTER yourself — move a step to the {direction}"

        # 4. Feet visible
        feet_vis = (
            kp33[L_ANKLE, 2] > 0.25 or kp33[R_ANKLE, 2] > 0.25
        )
        if not feet_vis:
            return False, "STEP BACK — feet are cropped out of frame"

        # 5. Arms away from torso
        # Wrist x should not be between shoulder x values (too close to body)
        ls_x = float(kp33[L_SHOULDER, 0])
        rs_x = float(kp33[R_SHOULDER, 0])
        x_min, x_max = min(ls_x, rs_x), max(ls_x, rs_x)
        lw_inside = x_min < float(kp33[L_WRIST, 0]) < x_max
        rw_inside = x_min < float(kp33[R_WRIST, 0]) < x_max
        if (
            kp33[L_WRIST, 2] > 0.3 and kp33[R_WRIST, 2] > 0.3
            and lw_inside and rw_inside
        ):
            return False, "Move arms slightly AWAY from body"

        return True, ""

    def check_side(self, kp33: np.ndarray) -> Tuple[bool, str]:
        """
        Evaluate side-view frame quality.

        Checks:
          1. Body not at ~90° to camera → "TURN further"
          2. Hip not centered → "CENTER yourself"

        A simple proxy for body angle: in a true side view, the left and
        right shoulder x-coordinates should be close together (overlapping).
        """
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP,      R_HIP      = 11, 12

        # 1. Body angle — shoulder x overlap
        conf_ls = float(kp33[L_SHOULDER, 2])
        conf_rs = float(kp33[R_SHOULDER, 2])

        if conf_ls > 0.25 and conf_rs > 0.25:
            shoulder_x_spread = abs(
                float(kp33[R_SHOULDER, 0]) - float(kp33[L_SHOULDER, 0])
            )
            # In side view shoulders overlap → spread < 0.08 (normalised)
            if shoulder_x_spread > 0.12:
                return False, "TURN further — show your right side profile"

        # 2. Hip centering
        hip_conf_ok = kp33[L_HIP, 2] > 0.3 or kp33[R_HIP, 2] > 0.3
        if hip_conf_ok:
            hip_xs = []
            if kp33[L_HIP, 2] > 0.3:
                hip_xs.append(float(kp33[L_HIP, 0]))
            if kp33[R_HIP, 2] > 0.3:
                hip_xs.append(float(kp33[R_HIP, 0]))
            hip_mx = sum(hip_xs) / len(hip_xs)
            if abs(hip_mx - 0.5) > 0.20:
                direction = "left" if hip_mx > 0.5 else "right"
                return False, f"CENTER yourself — move a step to the {direction}"

        return True, ""

    def is_stable(self, view: str, is_acceptable: bool, hold_seconds: float = 1.0) -> bool:
        """
        Returns True once the frame has been continuously acceptable for
        `hold_seconds`.  Resets the timer if the frame becomes unacceptable.

        Args:
            view         : 'front' or 'side'
            is_acceptable: result of check_front / check_side
        """
        now = time.monotonic()
        attr = f"_{view}_stable_since"

        if is_acceptable:
            if getattr(self, attr) is None:
                setattr(self, attr, now)
            elapsed = now - getattr(self, attr)
            return elapsed >= hold_seconds
        else:
            setattr(self, attr, None)
            return False

    def reset(self) -> None:
        """Reset stability timers."""
        self._front_stable_since = None
        self._side_stable_since  = None

    @staticmethod
    def draw_overlay(
        img: np.ndarray,
        feedback: str,
        acceptable: bool,
    ) -> np.ndarray:
        """
        Draw a coloured border and feedback text on a copy of img.

        Green border  = acceptable.
        Yellow border = fix needed.
        """
        import cv2
        out = img.copy()
        H, W = out.shape[:2]
        color = (0, 220, 0) if acceptable else (0, 220, 220)
        thickness = 6
        cv2.rectangle(out, (thickness, thickness), (W - thickness, H - thickness),
                      color, thickness)

        if feedback:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(out, (0, H - 50), (W, H), (20, 20, 20), -1)
            cv2.putText(out, feedback, (12, H - 16), font, 0.65,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return out

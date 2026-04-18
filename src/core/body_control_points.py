"""
Body Control Points
===================
Maps the 12 canonical garment control points to 12 corresponding body
positions using MediaPipe pose landmarks detected in the live camera frame.

The points are in the same semantic order as GarmentControlPoints:
    0  L-shoulder        1  R-shoulder
    2  L-collar          3  R-collar
    4  L-underarm        5  R-underarm
    6  L-sleeve-end      7  R-sleeve-end
    8  L-hem             9  R-hem
    10 C-collar          11 C-hem

MediaPipe landmark indices used:
    0  = nose
    11 = left_shoulder   12 = right_shoulder
    13 = left_elbow      14 = right_elbow
    15 = left_wrist      16 = right_wrist
    23 = left_hip        24 = right_hip

Usage
-----
    from src.core.body_control_points import BodyControlPoints
    pts = BodyControlPoints.compute(landmarks, frame_w=640, frame_h=480)
    # returns (12, 2) float32, col/row in frame pixel space
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Garment-type sleeve length fractions
# For 'tshirt' sleeves end 60% from shoulder toward elbow.
# For 'shirt' (full sleeve) ends at wrist via elbow.
_SLEEVE_FRACTIONS: dict[str, float] = {
    'tshirt':       0.60,
    'shortsleeve':  0.55,
    'longsleeve':   1.00,
    'shirt':        1.00,
    'hoodie':       0.95,
    'dress':        0.00,   # no sleeves — use underarm point only
    'tank':         0.00,
    'vest':         0.00,
}


class BodyControlPoints:
    """
    Converts MediaPipe normalized landmarks to 12 frame-pixel control points
    that correspond semantically to the garment's canonical control points.

    All methods are static/class-level — no instance state required.
    """

    @classmethod
    def compute(
        cls,
        landmarks: dict,
        frame_w: int,
        frame_h: int,
        garment_type: str = 'tshirt',
    ) -> Optional[np.ndarray]:
        """
        Compute 12 body control points in frame pixel coordinates.

        Parameters
        ----------
        landmarks : dict
            MediaPipe landmark dict {idx: {'x': float, 'y': float, 'visibility': float}}
            with x,y normalized to [0,1].
        frame_w, frame_h : int
            Camera frame dimensions in pixels.
        garment_type : str
            Controls sleeve endpoint selection.

        Returns
        -------
        (12, 2) float32  (col, row) in pixel space, or None if required
        landmarks are missing / low confidence.
        """
        # Check required landmarks are present and visible enough
        required = [11, 12, 13, 14, 23, 24]
        for idx in required:
            lm = landmarks.get(idx)
            if lm is None or lm.get('visibility', 0) < 0.25:
                logger.debug("[BCP] landmark %d missing or low visibility", idx)
                return None

        def px(idx: int) -> np.ndarray:
            """Convert normalized landmark to pixel (col, row)."""
            lm = landmarks[idx]
            return np.array([lm['x'] * frame_w, lm['y'] * frame_h], dtype=np.float64)

        def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            return a + (b - a) * t

        # --- Key landmark pixel positions ---
        l_shoulder = px(11)
        r_shoulder = px(12)
        l_elbow    = px(13)
        r_elbow    = px(14)
        l_hip      = px(23)
        r_hip      = px(24)

        # MediaPipe shoulder landmarks sit at the glenohumeral joint center,
        # which is ~12% INWARD from the physical shoulder edge where a shirt's
        # shoulder seam actually lands.  Expand the span outward so the
        # warped garment shoulder seams reach the deltoid / outer shoulder.
        _sspan     = np.linalg.norm(r_shoulder - l_shoulder)
        _sdir      = (r_shoulder - l_shoulder) / (_sspan + 1e-6)  # unit vec L→R
        _expand    = _sspan * 0.12          # 12% each side ≈ 24% total expansion
        l_shoulder = l_shoulder - _sdir * _expand
        r_shoulder = r_shoulder + _sdir * _expand
        nose       = px(0) if 0 in landmarks and landmarks[0].get('visibility', 0) > 0.2 \
                     else lerp(l_shoulder, r_shoulder, 0.5) + np.array([0, -(abs(l_shoulder[1] - l_hip[1]) * 0.55)])

        # Optional wrist for long-sleeve endpoint
        l_wrist = px(15) if (15 in landmarks and landmarks[15].get('visibility', 0) > 0.2) \
                  else None
        r_wrist = px(16) if (16 in landmarks and landmarks[16].get('visibility', 0) > 0.2) \
                  else None

        # --- Derived positions ---
        shoulder_mid   = lerp(l_shoulder, r_shoulder, 0.5)
        hip_mid        = lerp(l_hip, r_hip, 0.5)

        # Shoulder tilt angle (radians, positive = right shoulder higher)
        tilt_angle = math.atan2(
            float(r_shoulder[1] - l_shoulder[1]),
            float(r_shoulder[0] - l_shoulder[0]),
        )

        # Collar: midpoint between shoulders, offset toward nose.
        # The glenohumeral joint (MP landmark) is ~armpit height.
        # The collarbone / shirt collar sits ~28% of torso height above it.
        collar_dir = (nose - shoulder_mid)
        collar_dir_n = collar_dir / (np.linalg.norm(collar_dir) + 1e-6)
        torso_h = np.linalg.norm(hip_mid - shoulder_mid)
        collar_offset = collar_dir_n * (torso_h * 0.28)   # was 0.20
        collar_mid = shoulder_mid + collar_offset
        l_collar = lerp(l_shoulder, collar_mid, 0.5)
        r_collar = lerp(r_shoulder, collar_mid, 0.5)

        # Underarm: 30% down from shoulder toward hip
        l_underarm = lerp(l_shoulder, l_hip, 0.30)
        r_underarm = lerp(r_shoulder, r_hip, 0.30)
        # Widen underarm outwards by 8% of shoulder span
        shoulder_span = np.linalg.norm(r_shoulder - l_shoulder)
        l_underarm = l_underarm + np.array([-shoulder_span * 0.08, 0])
        r_underarm = r_underarm + np.array([ shoulder_span * 0.08, 0])

        # Sleeve end
        sleeve_frac = _SLEEVE_FRACTIONS.get(garment_type.lower(), 0.60)
        if sleeve_frac <= 0.0:
            # No sleeve — sleeve-end = underarm
            l_sleeve_end = l_underarm.copy()
            r_sleeve_end = r_underarm.copy()
        elif sleeve_frac >= 1.0 and l_wrist is not None and r_wrist is not None:
            # Full sleeve — elbow then optionally wrist
            l_sleeve_end = lerp(l_elbow, l_wrist, 0.5)
            r_sleeve_end = lerp(r_elbow, r_wrist, 0.5)
        else:
            l_sleeve_end = lerp(l_shoulder, l_elbow, sleeve_frac)
            r_sleeve_end = lerp(r_shoulder, r_elbow, sleeve_frac)

        # Hem: hip level with small outward pad (5 % of shoulder span)
        hem_pad = shoulder_span * 0.05
        l_hem = l_hip + np.array([-hem_pad, 0])
        r_hem = r_hip + np.array([ hem_pad, 0])
        hem_mid = hip_mid

        # --- Apply shoulder tilt correction to all non-shoulder points ---
        # Rotate all points about shoulder_mid by tilt_angle so garment
        # aligns with the actual body lean.
        def _rotate(pt: np.ndarray, pivot: np.ndarray, angle: float) -> np.ndarray:
            s, c = math.sin(angle), math.cos(angle)
            d = pt - pivot
            return pivot + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

        # Only apply tilt to hem/hip points (shoulders already correct from landmarks)
        if abs(tilt_angle) > 0.02:  # > ~1 degree
            l_hem      = _rotate(l_hem,      shoulder_mid, tilt_angle * 0.5)
            r_hem      = _rotate(r_hem,      shoulder_mid, tilt_angle * 0.5)
            hem_mid    = _rotate(hem_mid,    shoulder_mid, tilt_angle * 0.5)

        # --- Assemble 12 points ---
        pts = np.array([
            l_shoulder,      # 0
            r_shoulder,      # 1
            l_collar,        # 2
            r_collar,        # 3
            l_underarm,      # 4
            r_underarm,      # 5
            l_sleeve_end,    # 6
            r_sleeve_end,    # 7
            l_hem,           # 8
            r_hem,           # 9
            collar_mid,      # 10
            hem_mid,         # 11
        ], dtype=np.float32)

        # Clamp to frame
        pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)

        logger.debug("[BCP] %d body control points (tilt=%.1f°, garment=%s)",
                     len(pts), math.degrees(tilt_angle), garment_type)
        return pts

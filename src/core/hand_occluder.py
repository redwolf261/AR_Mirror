"""
Hand Occlusion Mask
===================
Generates a per-frame mask marking the hand and forearm regions so the
compositor can restore the original frame pixels there — making hands appear
in front of the garment rather than disappearing behind it.

The mask is built from MediaPipe hand landmarks (21 points per hand) and the
forearm segment derived from wrist/elbow pose landmarks.

Usage
-----
    from src.core.hand_occluder import HandOccluder

    occluder = HandOccluder()
    mask = occluder.make_mask(
        frame_shape=(480, 640),
        hand_lm_left=holistic_result.left_hand,
        hand_lm_right=holistic_result.right_hand,
        pose_lm=landmarks_dict,
    )
    # mask: (H, W) uint8 — 255 where hands/forearms are, 0 elsewhere
    # After blurring becomes float32 [0,1] soft alpha for blending.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pixels to dilate the convex hull outward before blurring
_DILATE_PX   = 12
# Gaussian blur kernel (must be odd)
_BLUR_K      = 21
# Fraction of forearm length to extend past wrist (captures hand overlap)
_WRIST_EXT   = 0.15


class HandOccluder:
    """
    Builds a soft hand/forearm occlusion mask from pose + hand landmarks.

    The mask is non-zero where the user's hands and forearms are detected.
    It is used to "punch through" the garment so hands in front of the
    torso are rendered on top of the shirt.
    """

    def make_mask(
        self,
        frame_shape: tuple[int, int],
        hand_lm_left:  Optional[list[Any]],
        hand_lm_right: Optional[list[Any]],
        pose_lm:       dict,
    ) -> np.ndarray:
        """
        Build a uint8 binary mask (255 = hand region, 0 = elsewhere).

        Parameters
        ----------
        frame_shape : (H, W) or (H, W, 3)
            Shape of the camera frame.
        hand_lm_left, hand_lm_right : list of MediaPipe NormalizedLandmark
            21-point hand landmark lists from mp.solutions.holistic result.
            Pass None if the hand was not detected.
        pose_lm : dict
            MediaPipe normalized pose landmarks {idx: {'x','y','visibility'}}.

        Returns
        -------
        (H, W) uint8  — 0 or 255 raw; caller may want to convert to float.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw forearm + hand for left arm (landmarks 13=l_elbow, 15=l_wrist)
        self._draw_arm(mask, pose_lm, elbow_idx=13, wrist_idx=15,
                       hand_lm=hand_lm_left, w=w, h=h)
        # Right arm (landmarks 14=r_elbow, 16=r_wrist)
        self._draw_arm(mask, pose_lm, elbow_idx=14, wrist_idx=16,
                       hand_lm=hand_lm_right, w=w, h=h)

        if not mask.any():
            return mask

        # Dilate then blur for soft edges
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_DILATE_PX * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k, iterations=1)
        mask = cv2.GaussianBlur(mask, (_BLUR_K, _BLUR_K), sigmaX=8, sigmaY=8)
        return mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_arm(
        self,
        mask:     np.ndarray,
        pose_lm:  dict,
        elbow_idx: int,
        wrist_idx: int,
        hand_lm:   Optional[list[Any]],
        w: int,
        h: int,
    ) -> None:
        """
        Draw forearm capsule + hand convex hull onto *mask* in-place.
        """
        def _px(idx: int) -> Optional[np.ndarray]:
            lm = pose_lm.get(idx)
            if lm is None or lm.get('visibility', 0) < 0.20:
                return None
            return np.array([int(lm['x'] * w), int(lm['y'] * h)], dtype=np.int32)

        elbow = _px(elbow_idx)
        wrist = _px(wrist_idx)

        if elbow is None and wrist is None and hand_lm is None:
            return

        pts_all: list[np.ndarray] = []

        # ---------- Forearm capsule (elbow → wrist + extension) ----------
        if elbow is not None and wrist is not None:
            arm_vec    = wrist.astype(np.float64) - elbow.astype(np.float64)
            arm_len    = float(np.linalg.norm(arm_vec))
            if arm_len > 1e-3:
                arm_dir = arm_vec / arm_len
                arm_perp = np.array([-arm_dir[1], arm_dir[0]])
                # Capsule width ≈ 14% of arm length
                half_w  = max(8, int(arm_len * 0.14))
                ext_pt  = wrist.astype(np.float64) + arm_dir * (arm_len * _WRIST_EXT)
                for t in np.linspace(0, 1, 6):
                    mid = elbow.astype(np.float64) * (1 - t) + ext_pt * t
                    pts_all.append((mid + arm_perp * half_w).astype(np.int32))
                    pts_all.append((mid - arm_perp * half_w).astype(np.int32))
        elif wrist is not None:
            pts_all.append(wrist)

        # ---------- Hand convex hull (from hand landmarks) ---------------
        if hand_lm is not None:
            for lm in hand_lm:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts_all.append(np.array([px, py], dtype=np.int32))

        if not pts_all:
            return

        hull_pts = np.array(pts_all, dtype=np.int32)
        if len(hull_pts) >= 3:
            hull = cv2.convexHull(hull_pts)
            cv2.fillConvexPoly(mask, hull, 255)
        elif len(hull_pts) == 2:
            cv2.line(mask, tuple(hull_pts[0]), tuple(hull_pts[1]), 255, thickness=20)
        else:
            cv2.circle(mask, tuple(hull_pts[0]), 18, 255, -1)

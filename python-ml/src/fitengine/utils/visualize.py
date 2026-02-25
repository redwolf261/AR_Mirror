"""
Visualisation helpers.

draw_skeleton    : overlays body-33 skeleton on a BGR image.
draw_width_curve : draws silhouette width curve in a separate panel.
draw_size_badge  : renders a size result badge onto a BGR image.
"""

import cv2
import numpy as np
from typing import Optional

# Skeleton connectivity for body-33
_SKELETON = [
    (0, 27),   # nose → neck
    (27, 5),   # neck → left_shoulder
    (27, 6),   # neck → right_shoulder
    (5, 7),    # left_shoulder → left_elbow
    (7, 9),    # left_elbow → left_wrist
    (6, 8),    # right_shoulder → right_elbow
    (8, 10),   # right_elbow → right_wrist
    (5, 11),   # left_shoulder → left_hip
    (6, 12),   # right_shoulder → right_hip
    (11, 12),  # left_hip → right_hip
    (11, 13),  # left_hip → left_knee
    (13, 15),  # left_knee → left_ankle
    (12, 14),  # right_hip → right_knee
    (14, 16),  # right_knee → right_ankle
    (32, 11),  # mid_hip → left_hip
    (32, 12),  # mid_hip → right_hip
]

_CONF_THRESHOLD = 0.3
_JOINT_COLOR = (0, 255, 100)
_BONE_COLOR  = (255, 200, 0)
_LOW_CONF_COLOR = (80, 80, 80)


def draw_skeleton(
    img: np.ndarray,
    kp33: np.ndarray,
    normalized: bool = True,
    thickness: int = 2,
    radius: int = 4,
) -> np.ndarray:
    """
    Draw body-33 skeleton on a copy of img.

    Args:
        img        : BGR image [H, W, 3].
        kp33       : [33, 3] keypoints (x, y, conf).
                     If normalized=True, x/y are in [0,1]; otherwise pixel coords.
        normalized : whether coords are normalised.
        thickness  : bone line thickness in pixels.
        radius     : joint circle radius in pixels.

    Returns:
        Annotated BGR image copy.
    """
    out = img.copy()
    H, W = img.shape[:2]

    def _to_px(kp):
        x = int(kp[0] * W) if normalized else int(kp[0])
        y = int(kp[1] * H) if normalized else int(kp[1])
        return x, y

    # Draw bones
    for a, b in _SKELETON:
        if kp33[a, 2] < _CONF_THRESHOLD or kp33[b, 2] < _CONF_THRESHOLD:
            continue
        pa, pb = _to_px(kp33[a]), _to_px(kp33[b])
        cv2.line(out, pa, pb, _BONE_COLOR, thickness, cv2.LINE_AA)

    # Draw joints
    for i, kp in enumerate(kp33):
        color = _JOINT_COLOR if kp[2] >= _CONF_THRESHOLD else _LOW_CONF_COLOR
        p = _to_px(kp)
        cv2.circle(out, p, radius, color, -1, cv2.LINE_AA)

    return out


def draw_width_curve(
    width_curve: np.ndarray,
    panel_h: int = 200,
    panel_w: int = 120,
    color: tuple = (0, 200, 255),
) -> np.ndarray:
    """
    Render silhouette width curve as a vertical bar chart panel.

    Args:
        width_curve : [32] normalised width values in [0, 1].
        panel_h     : panel height in pixels.
        panel_w     : panel width in pixels.

    Returns:
        BGR panel [panel_h, panel_w, 3].
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    n = len(width_curve)
    bar_h = max(1, panel_h // n)

    for i, w in enumerate(width_curve):
        bar_len = int(w * (panel_w - 4))
        y0 = i * bar_h
        y1 = y0 + bar_h - 1
        cv2.rectangle(panel, (2, y0), (2 + bar_len, y1), color, -1)

    return panel


def draw_size_badge(
    img: np.ndarray,
    size_result: dict,
    position: tuple = (20, 20),
    font_scale: float = 0.7,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Overlay a size result badge on img.

    Args:
        img         : BGR image.
        size_result : dict with keys collar, jacket, trouser_waist, confidence_level.
        position    : top-left (x, y) of badge.
        font_scale  : text scale.
        alpha       : background rectangle transparency.
    """
    out = img.copy()
    x0, y0 = position

    lines = [
        f"Collar : {size_result.get('collar', '?')}",
        f"Jacket : {size_result.get('jacket', '?')}",
        f"Trouser: {size_result.get('trouser_waist', '?')}",
        f"Conf   : {size_result.get('confidence_level', '?')}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(28 * font_scale)
    badge_w = 260
    badge_h = len(lines) * line_h + 16

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    conf = size_result.get("confidence_level", "heuristic")
    text_color = (100, 255, 100) if conf == "High" else (200, 200, 200)

    for i, line in enumerate(lines):
        ty = y0 + 14 + i * line_h
        cv2.putText(out, line, (x0 + 8, ty), font, font_scale, text_color, 1, cv2.LINE_AA)

    return out

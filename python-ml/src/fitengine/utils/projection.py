"""
Camera projection utilities.

weak_perspective_project : front-view projection (scale + translate)
side_view_project         : rotates joints 90° about Y-axis then projects
"""

import numpy as np


def _rotation_y(angle_deg: float) -> np.ndarray:
    """4×4 rotation matrix about the Y axis."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


def weak_perspective_project(
    joints_3d: np.ndarray,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """
    Weak-perspective projection (front view).

    Args:
        joints_3d : [J, 3] or [B, J, 3] — 3D joint positions in metres.
        scale     : camera scale factor.
        tx, ty    : 2D translation.

    Returns:
        kp_2d : same leading dims + [2] (x, y) in normalised image coords.
    """
    batched = joints_3d.ndim == 3
    pts = joints_3d if batched else joints_3d[np.newaxis]  # [B, J, 3]

    x = scale * pts[..., 0] + tx
    y = scale * pts[..., 1] + ty
    kp_2d = np.stack([x, y], axis=-1)

    return kp_2d if batched else kp_2d[0]


def side_view_project(
    joints_3d: np.ndarray,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rotation_deg: float = 90.0,
) -> np.ndarray:
    """
    Side-view projection: rotate joints around Y axis then weak-perspective.

    Args:
        joints_3d    : [J, 3] or [B, J, 3].
        scale        : camera scale factor.
        tx, ty       : 2D translation.
        rotation_deg : angle to rotate about Y (default 90° = right-side view).

    Returns:
        kp_2d : same leading dims + [2].
    """
    batched = joints_3d.ndim == 3
    pts = joints_3d if batched else joints_3d[np.newaxis]  # [B, J, 3]

    R = _rotation_y(rotation_deg)[:3, :3].astype(np.float32)  # [3,3]
    rotated = pts @ R.T  # [B, J, 3]

    x = scale * rotated[..., 0] + tx
    y = scale * rotated[..., 1] + ty
    kp_2d = np.stack([x, y], axis=-1)

    return kp_2d if batched else kp_2d[0]

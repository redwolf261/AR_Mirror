"""
Garment Control Points
======================
Analyses the cloth mask (loaded once per garment) to compute 12 canonical
control points in garment-image pixel coordinates.

These points are matched 1-to-1 with the 12 body control points produced
by BodyControlPoints, and the pair drives the TPS warp.

Control-point layout (indices):
    0  L-shoulder        1  R-shoulder
    2  L-collar          3  R-collar
    4  L-underarm        5  R-underarm
    6  L-sleeve-end      7  R-sleeve-end
    8  L-hem             9  R-hem
    10 C-collar          11 C-hem

Usage
-----
    from src.core.garment_control_points import GarmentControlPoints
    gcp = GarmentControlPoints(cloth_mask)
    pts = gcp.compute()   # (12, 2) float32, (col, row)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Number of canonical control points
N_PTS = 12


class GarmentControlPoints:
    """
    Derives 12 canonical garment control points from the cloth binary mask.

    The mask is analysed once and the result cached.  Calling compute()
    multiple times returns the same array without recomputing.

    Parameters
    ----------
    cloth_mask : (H, W) or (H, W, 1) float32 [0, 1]
        White-on-black garment silhouette mask.
    """

    def __init__(self, cloth_mask: np.ndarray) -> None:
        m = cloth_mask.squeeze() if cloth_mask.ndim == 3 else cloth_mask
        self._mask: np.ndarray = (m > 0.5).astype(np.uint8)
        self._pts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self) -> np.ndarray:
        """Return (12, 2) float32 control points in (col, row) pixel coords."""
        if self._pts is not None:
            return self._pts
        self._pts = self._extract_points()
        return self._pts

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_points(self) -> np.ndarray:
        mask = self._mask
        h, w = mask.shape

        # -- Bounding box of the mask --------------------------------
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            # Degenerate mask — return a uniform grid
            logger.warning("[GCP] Empty cloth mask — using uniform grid")
            return self._uniform_fallback(h, w)

        r_min, r_max = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        c_min, c_max = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        cx = (c_min + c_max) // 2   # horizontal centre

        # -- Row-wise horizontal extents for structural analysis -----
        # Sample 5 horizontal slices to find shoulder / underarm / hem rows
        top_row     = r_min
        bot_row     = r_max
        q25_row     = r_min + (r_max - r_min) // 4     # ~underarm level
        q50_row     = (r_min + r_max) // 2             # ~waist level
        collar_row  = r_min + int((r_max - r_min) * 0.07)  # just below top

        def _row_extent(row: int) -> tuple[int, int]:
            """Return (left_col, right_col) of mask at given row."""
            if 0 <= row < h:
                cols_at = np.where(mask[row] > 0)[0]
                if len(cols_at):
                    return int(cols_at[0]), int(cols_at[-1])
            return c_min, c_max

        # Shoulder edges (topmost non-empty row)
        shl, shr = _row_extent(top_row)
        # Collar edges (~7% from top)
        cll, clr  = _row_extent(collar_row)
        # Underarm edges (~25% from top, usually the narrowest horizontal span)
        ual, uar  = _row_extent(q25_row)

        # Sleeve ends: scan down left/right columns from shoulder to find
        # where the mask first narrows significantly (sleeve terminus)
        sleeve_row = self._find_sleeve_end_row(mask, r_min, q25_row)

        sl_sleeve_l, _ = _row_extent(sleeve_row)
        _,  sl_sleeve_r = _row_extent(sleeve_row)

        # Hem edges, bottom row
        heml, hemr = _row_extent(bot_row)
        hem_cx = (heml + hemr) // 2

        # Collar centre (slightly inset from top)
        collar_cx = cx

        # -- Assemble 12 points (col, row) --------------------------
        pts = np.array([
            [shl,        top_row],      # 0  L-shoulder
            [shr,        top_row],      # 1  R-shoulder
            [cll + (cx - cll) // 2, collar_row],   # 2  L-collar (mid between edge and center)
            [clr - (clr - cx) // 2, collar_row],   # 3  R-collar
            [ual,        q25_row],      # 4  L-underarm
            [uar,        q25_row],      # 5  R-underarm
            [sl_sleeve_l, sleeve_row],  # 6  L-sleeve-end
            [sl_sleeve_r, sleeve_row],  # 7  R-sleeve-end
            [heml,       bot_row],      # 8  L-hem
            [hemr,       bot_row],      # 9  R-hem
            [collar_cx,  collar_row],   # 10 C-collar
            [hem_cx,     bot_row],      # 11 C-hem
        ], dtype=np.float32)

        # Clamp to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        logger.debug("[GCP] Computed %d garment control points — bbox (%d,%d)→(%d,%d)",
                     N_PTS, c_min, r_min, c_max, r_max)
        return pts

    def _find_sleeve_end_row(self, mask: np.ndarray, r_start: int, r_end: int) -> int:
        """
        Find the row where horizontal span reaches its minimum (sleeve terminus).
        Falls back to midpoint between r_start and r_end.
        """
        if r_end <= r_start:
            return r_start
        widths = []
        for r in range(r_start, r_end + 1):
            cols_at = np.where(mask[r] > 0)[0]
            widths.append(int(cols_at[-1] - cols_at[0]) if len(cols_at) >= 2 else 0)
        min_idx = int(np.argmin(widths))
        return r_start + min_idx

    def _uniform_fallback(self, h: int, w: int) -> np.ndarray:
        """Return a simple 3×4 regular grid as a last-resort fallback."""
        cols = np.linspace(w * 0.2, w * 0.8, 4, dtype=np.float32)
        rows = np.linspace(h * 0.1, h * 0.9, 3, dtype=np.float32)
        pts_list = []
        for r in rows:
            for c in cols:
                pts_list.append([c, r])
        return np.array(pts_list[:N_PTS], dtype=np.float32)

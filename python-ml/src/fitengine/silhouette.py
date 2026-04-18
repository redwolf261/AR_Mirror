"""
Silhouette width curve extraction.

SilhouetteExtractor converts a binary body mask (or projected 3D vertices)
into a 32-bin normalised horizontal width curve used as input features for
the DualViewRegressor.

Each bin represents the horizontal body extent at a relative vertical
position — scale-invariant by design.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


_N_BINS = 32


class SilhouetteExtractor:
    """
    Extracts a 32-bin width curve from a binary body mask or 3D vertices.
    """

    def __init__(self, n_bins: int = _N_BINS) -> None:
        self.n_bins = n_bins

    def from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract width curve from a binary mask.

        Args:
            mask : [H, W] uint8 — 255 = body, 0 = background.

        Returns:
            width_curve : [n_bins] float32 in [0, 1].
                Each bin is horizontal body width / total image width at
                that vertical slice. 0 if no foreground in that slice.
        """
        H, W = mask.shape
        fg = (mask > 128)

        # Find body bounding box rows
        row_has_fg = fg.any(axis=1)
        if not row_has_fg.any():
            return np.zeros(self.n_bins, dtype=np.float32)

        y_min = int(np.argmax(row_has_fg))
        y_max = int(len(row_has_fg) - 1 - np.argmax(row_has_fg[::-1]))
        body_h = max(y_max - y_min, 1)

        curve = np.zeros(self.n_bins, dtype=np.float32)
        bin_h = body_h / self.n_bins

        for i in range(self.n_bins):
            row_start = int(y_min + i * bin_h)
            row_end   = int(y_min + (i + 1) * bin_h)
            row_end   = min(row_end, H)
            if row_start >= H:
                break

            slice_fg = fg[row_start:row_end, :]
            if not slice_fg.any():
                continue

            # Width = rightmost − leftmost foreground pixel
            col_any = slice_fg.any(axis=0)
            x_min = int(np.argmax(col_any))
            x_max = int(len(col_any) - 1 - np.argmax(col_any[::-1]))
            width_px = x_max - x_min + 1
            curve[i] = width_px / float(W)

        return curve

    def from_vertices(
        self,
        vertices: np.ndarray,
        scale: float = 1.0,
        tx: float = 0.0,
    ) -> np.ndarray:
        """
        Approximate width curve directly from projected 3D vertices.

        Used in synthetic data generation (Phase 2) — faster than
        rasterising a full mask.

        Args:
            vertices : [V, 3] — 3D body vertices in normalised camera space.
            scale    : projection scale.
            tx       : horizontal translation.

        Returns:
            width_curve : [n_bins] float32 in [0, 1].
        """
        x_proj = scale * vertices[:, 0] + tx  # [V]
        y_proj = scale * vertices[:, 1]        # [V]

        if (y_proj.max() - y_proj.min()) < 1e-6:
            return np.zeros(self.n_bins, dtype=np.float32)

        y_min, y_max = y_proj.min(), y_proj.max()
        bin_h = (y_max - y_min) / self.n_bins
        x_range = x_proj.max() - x_proj.min()

        curve = np.zeros(self.n_bins, dtype=np.float32)
        for i in range(self.n_bins):
            y_lo = y_min + i * bin_h
            y_hi = y_lo + bin_h
            in_bin = (y_proj >= y_lo) & (y_proj < y_hi)
            if not in_bin.any():
                continue
            x_slice = x_proj[in_bin]
            span = x_slice.max() - x_slice.min()
            curve[i] = span / (x_range + 1e-8)

        return curve

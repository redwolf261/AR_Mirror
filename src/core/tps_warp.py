"""
Thin Plate Spline Warp Engine
==============================
Pure-numpy TPS solver. Given N source control points on the garment image
and N destination control points on the live camera frame, computes the
(H, W, 2) displacement field and applies it via cv2.remap.

No scipy required — the (N+3)×(N+3) linear system is solved with
numpy.linalg.solve, which takes ~0.2 ms for N=12.

Usage
-----
    from src.core.tps_warp import TPSWarp

    warp = TPSWarp()
    warp.fit(src_pts, dst_pts)          # solve once per detection update
    warped_cloth = warp.warp_image(cloth_rgb)
    warped_mask  = warp.warp_mask(cloth_mask)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TPSWarp:
    """
    Thin Plate Spline image warp.

    Mathematical basis
    ------------------
    For 2-D warping we solve independently for each output axis (x, y).
    The TPS mapping f: R²→R is:

        f(p) = a₀ + a₁·u + a₂·v + Σᵢ wᵢ·ϕ(‖p − pᵢ‖)

    where ϕ(r) = r²·log(r²+ε) is the TPS radial basis function and
    {pᵢ} are the N source control points.

    The coefficients (w₁…wN, a₀, a₁, a₂) are found by solving the
    (N+3)×(N+3) system:

        ⎡ K   P ⎤ ⎡ w ⎤   ⎡ dst_col ⎤
        ⎣ Pᵀ  0 ⎦ ⎣ a ⎦ = ⎣   0     ⎦

    where K[i,j] = ϕ(‖pᵢ−pⱼ‖) and P = [1, u, v] for each source point.

    Parameters
    ----------
    regularization : float
        Ridge term added to K diagonal to avoid numerical singularity
        when control points are close or coincident.  1e-3 is a safe
        default that has negligible visual effect.
    """

    def __init__(self, regularization: float = 1e-3) -> None:
        self._reg = regularization
        self._coeffs_x: Optional[np.ndarray] = None   # (N+3,)
        self._coeffs_y: Optional[np.ndarray] = None
        self._src_pts:  Optional[np.ndarray] = None   # (N, 2) source control pts
        self._map_x:    Optional[np.ndarray] = None   # cached remap grid
        self._map_y:    Optional[np.ndarray] = None
        self._grid_shape: tuple[int, int] = (0, 0)
        self.is_fitted  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> None:
        """
        Compute TPS coefficients mapping src_pts → dst_pts.

        Parameters
        ----------
        src_pts : (N, 2) float32
            Control points in garment-image pixel space (col, row).
        dst_pts : (N, 2) float32
            Corresponding points in the camera-frame pixel space (col, row).
        """
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
            raise ValueError(f"src/dst must be (N,2); got {src.shape}, {dst.shape}")
        n = src.shape[0]

        # Build kernel matrix K
        K = self._rbf_matrix(src)         # (N, N)

        # Build polynomial matrix P  [1, u, v]
        P = np.hstack([np.ones((n, 1), dtype=np.float64), src])  # (N, 3)

        # Assemble block system
        zeros33 = np.zeros((3, 3), dtype=np.float64)
        top    = np.hstack([K, P])         # (N, N+3)
        bottom = np.hstack([P.T, zeros33]) # (3, N+3)
        A = np.vstack([top, bottom])       # (N+3, N+3)

        # Right-hand sides
        rhs_x = np.concatenate([dst[:, 0], np.zeros(3)])
        rhs_y = np.concatenate([dst[:, 1], np.zeros(3)])

        self._coeffs_x = np.linalg.solve(A, rhs_x)
        self._coeffs_y = np.linalg.solve(A, rhs_y)
        self._src_pts  = src
        self._map_x    = None   # invalidate cached grid
        self._map_y    = None
        self.is_fitted = True
        logger.debug("[TPS] fit: %d control points", n)

    def warp_image(self, img: np.ndarray) -> np.ndarray:
        """
        Warp a float32 RGB image using the fitted TPS.

        Parameters
        ----------
        img : (H, W, 3) float32 [0, 1]

        Returns
        -------
        warped : (H, W, 3) float32 [0, 1]
        """
        if not self.is_fitted:
            return img.copy()
        h, w = img.shape[:2]
        mx, my = self._get_remap(h, w)
        warped = cv2.remap(
            img, mx, my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.clip(warped.astype(np.float32), 0.0, 1.0)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Warp a float32 mask using the fitted TPS.

        Parameters
        ----------
        mask : (H, W) or (H, W, 1) float32 [0, 1]

        Returns
        -------
        warped_mask : (H, W) float32 [0, 1]
        """
        if not self.is_fitted:
            m = mask.squeeze() if mask.ndim == 3 else mask
            return m.copy().astype(np.float32)
        h, w = mask.shape[:2]
        m2d = mask.squeeze().astype(np.float32)
        mx, my = self._get_remap(h, w)
        warped = cv2.remap(
            m2d, mx, my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        return np.clip(warped, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_remap(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (map_x, map_y) for cv2.remap, computing and caching if stale."""
        if self._map_x is not None and self._grid_shape == (h, w):
            return self._map_x, self._map_y  # type: ignore[return-value]
        self._map_x, self._map_y = self._build_grid(h, w)
        self._grid_shape = (h, w)
        return self._map_x, self._map_y

    def _build_grid(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """
        For every output pixel (col, row) in the frame, compute where it
        maps to in the garment image.  This is the *inverse* warp (required
        by cv2.remap): for each destination pixel we ask "where in the
        source does it come from?"

        Strategy: We fit TPS from garment→frame (forward). To get the
        inverse we evaluate the forward map at a dense grid of garment
        pixels and then use the forward map coefficients to define map_x/y
        as frame→garment via the same coefficients evaluated at (col,row)
        in frame space, treating dst_pts as the new "source" in the
        system — equivalent to fitting the inverse TPS with src↔dst swapped
        but using the same already-solved coefficients to define the mapping
        from frame coordinates back to garment coordinates.

        Implementation: directly evaluate the TPS function whose coefficients
        were solved for the forward direction at every (col, row) of the
        output frame. This gives map_x[row,col] = garment_col and
        map_y[row,col] = garment_row, which is exactly what cv2.remap needs.
        We swap src ↔ dst so the coefficients map frame coords → garment coords.
        """
        assert self._src_pts is not None
        assert self._coeffs_x is not None
        assert self._coeffs_y is not None

        # Create a dense pixel grid in garment (source) space
        # We want: for each output (frame) pixel, find the garment pixel.
        # Equivalently: densely sample garment space, compute frame coords
        # via forward TPS, then invert via scattered interpolation.
        # For speed we instead re-solve: fit inverse TPS (dst→src) using
        # the same N control points but with roles swapped.
        src = self._src_pts                        # garment pts
        n   = src.shape[0]
        cx  = self._coeffs_x                       # (N+3,) maps garment→frame_x
        cy  = self._coeffs_y                       # (N+3,) maps garment→frame_y

        # Build a dense 2-D grid in garment pixel space
        gx = np.arange(w, dtype=np.float64)        # garment cols
        gy = np.arange(h, dtype=np.float64)        # garment rows
        gc, gr = np.meshgrid(gx, gy)               # each (H, W)
        pts = np.stack([gc.ravel(), gr.ravel()], axis=1)  # (H*W, 2)

        # Evaluate RBF kernel at every grid point against all control points
        # K_dense[i, j] = phi(||pts[i] - src[j]||)
        diff = pts[:, np.newaxis, :] - src[np.newaxis, :, :]   # (H*W, N, 2)
        r2   = (diff ** 2).sum(axis=2)                          # (H*W, N)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi  = np.where(r2 > 0, r2 * np.log(r2 + 1e-10), 0.0)  # (H*W, N)

        # Polynomial part: [1, col, row]
        poly = np.hstack([np.ones((pts.shape[0], 1)), pts])   # (H*W, 3)
        basis = np.hstack([phi, poly])                         # (H*W, N+3)

        # Forward map: garment coords → frame coords
        frame_col = (basis @ cx).reshape(h, w)  # (H, W) output frame col
        frame_row = (basis @ cy).reshape(h, w)  # (H, W) output frame row

        # cv2.remap: for each OUTPUT pixel (frame), map_x/y gives the SOURCE
        # (garment) coordinates. But we just computed the forward map (garment→frame).
        # Create the inverse by using OpenCV's perspective inversion through
        # a fine scattered interpolation: sample forward map densely in garment
        # space → get (frame_col, frame_row) → insert (garment col, garment row)
        # at those (fractional) frame positions.
        #
        # Practical shortcut that works well for near-affine TPS:
        # Fit the INVERSE TPS (swapping src↔dst) from the same control points.
        inv_warp = TPSWarp(regularization=self._reg)
        # dst (garment) control points come from self._src_pts
        # src (frame) control points: evaluate the forward TPS at src_pts to
        # get where each garment control point lands in the frame — these are
        # just dst_pts from the original fit.  We'll derive them from the
        # already-solved coefficients evaluated at the control points.
        n_src = n
        K_ctrl = self._rbf_matrix(src)
        P_ctrl = np.hstack([np.ones((n_src, 1)), src])
        basis_ctrl = np.hstack([K_ctrl, P_ctrl])  # (N, N+3)
        frame_ctrl_col = basis_ctrl @ cx
        frame_ctrl_row = basis_ctrl @ cy
        frame_ctrl_pts = np.stack([frame_ctrl_col, frame_ctrl_row], axis=1)  # (N, 2)

        # Now fit inverse: frame_ctrl_pts → garment src_pts
        inv_warp.fit(frame_ctrl_pts, src)
        map_x_inv, map_y_inv = inv_warp._build_grid_direct(h, w, factor=4)
        return map_x_inv.astype(np.float32), map_y_inv.astype(np.float32)

    def _build_grid_direct(self, h: int, w: int, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the TPS forward map on a coarse grid then bilinear-upsample.

        Evaluating at every `factor`-th pixel gives a factor² speedup.
        TPS is a smooth function, so bilinear upsampling of the flow map
        introduces negligible error (sub-pixel) for factor ≤ 8.

        Parameters
        ----------
        h, w : int   Full output dimensions.
        factor : int  Downsampling factor for the evaluation grid (default 4).
        """
        assert self._src_pts is not None
        assert self._coeffs_x is not None
        assert self._coeffs_y is not None
        src = self._src_pts
        cx  = self._coeffs_x
        cy  = self._coeffs_y

        # Coarse grid: every `factor`-th pixel at actual frame pixel positions
        h_g = max(1, (h + factor - 1) // factor)
        w_g = max(1, (w + factor - 1) // factor)
        gc = np.arange(w_g, dtype=np.float64) * factor
        gr = np.arange(h_g, dtype=np.float64) * factor
        gc_2d, gr_2d = np.meshgrid(gc, gr)
        pts = np.stack([gc_2d.ravel(), gr_2d.ravel()], axis=1)   # (h_g*w_g, 2)

        diff = pts[:, np.newaxis, :] - src[np.newaxis, :, :]
        r2   = (diff ** 2).sum(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = np.where(r2 > 0, r2 * np.log(r2 + 1e-10), 0.0)

        poly  = np.hstack([np.ones((pts.shape[0], 1)), pts])
        basis = np.hstack([phi, poly])

        map_x_small = (basis @ cx).reshape(h_g, w_g).astype(np.float32)
        map_y_small = (basis @ cy).reshape(h_g, w_g).astype(np.float32)

        # Bilinear upsample to full resolution
        if factor > 1:
            map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            map_x, map_y = map_x_small, map_y_small
        return map_x, map_y

    def _rbf_matrix(self, pts: np.ndarray) -> np.ndarray:
        """
        Build the N×N kernel matrix K where K[i,j] = ϕ(‖pᵢ−pⱼ‖).
        Regularization term (λI) added to the diagonal.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        r2   = (diff ** 2).sum(axis=2)                         # (N, N)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.where(r2 > 0, r2 * np.log(r2 + 1e-10), 0.0)
        K += np.eye(len(pts)) * self._reg
        return K

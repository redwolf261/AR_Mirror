#!/usr/bin/env python3
"""
Temporal Coherence Cache for Real-Time 3D Try-On

Exploits the fact that the human body moves slowly relative to
the camera frame rate. Instead of recomputing the full SMPL
reconstruction + mesh wrapping every frame, we:

1. Detect when landmarks change significantly (motion threshold)
2. Cache SMPL body mesh and reuse for N frames
3. Interpolate between cached keyframes for smooth motion
4. Only update physics simulation every K frames

This reduces per-frame cost from ~80ms to ~5ms for static poses.

Performance budget at 30 FPS (33ms/frame):
  - Landmark detection:  4ms  (always runs)
  - Motion check:        0.1ms
  - SMPL reconstruct:    5ms  (every Nth frame)
  - Mesh wrap:           5ms  (every Nth frame)
  - Physics step:        3ms  (every Kth frame)
  - GPU render:          2ms  (always runs)
  - Total (cache hit):   ~6ms
  - Total (cache miss):  ~19ms

Author: AR Mirror Pipeline
Date: February 15, 2026
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CachedFrame:
    """Snapshot of one keyframe's computation results."""
    timestamp: float                    # time.perf_counter()
    landmarks: Dict                     # raw MediaPipe landmarks dict
    body_mesh: Any = None               # SMPLMeshResult
    wrapped_mesh: Any = None            # WrappedGarmentMesh
    rendered: Optional[np.ndarray] = None  # RGBA image
    quality_score: float = 0.0


class TemporalCache:
    """
    Smart cache that decides when to recompute vs reuse.

    Motion detection compares the L2 norm of landmark deltas
    against a configurable threshold. When the body is mostly
    still, we reuse the previous mesh and only re-render.

    Keyframe interpolation blends between the two nearest
    cached body meshes using LERP on vertex positions (simple
    but effective for small motions).
    """

    def __init__(
        self,
        motion_threshold: float = 0.015,
        max_reuse_frames: int = 5,
        physics_interval: int = 2,
        cache_size: int = 30,
        interpolate: bool = True,
    ):
        """
        Args:
            motion_threshold: Normalised landmark delta that triggers
                              a full recompute (0.015 ≈ 1.5% of frame).
            max_reuse_frames: Maximum number of consecutive frames that
                              can reuse the cache before forcing refresh.
            physics_interval: Run physics every N frames.
            cache_size:       Number of keyframes to keep for interpolation.
            interpolate:      If True, LERP vertex positions between
                              keyframes; otherwise snap to last keyframe.
        """
        self.motion_threshold = motion_threshold
        self.max_reuse_frames = max_reuse_frames
        self.physics_interval = physics_interval
        self.interpolate = interpolate

        self._cache: deque[CachedFrame] = deque(maxlen=cache_size)
        self._frames_since_update: int = 0
        self._frame_count: int = 0
        self._last_landmarks: Optional[Dict] = None

        # Telemetry
        self._hits = 0
        self._misses = 0

        logger.info(
            f"TemporalCache: threshold={motion_threshold}, "
            f"max_reuse={max_reuse_frames}, physics_every={physics_interval}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def should_recompute(self, landmarks: Dict) -> bool:
        """
        Determine whether we need to run the full 3D pipeline.

        Returns True if:
        - This is the first frame
        - Landmark delta exceeds motion threshold
        - We've reused the cache for too many consecutive frames
        """
        self._frame_count += 1

        if self._last_landmarks is None:
            return True

        if self._frames_since_update >= self.max_reuse_frames:
            return True

        delta = self._landmark_delta(self._last_landmarks, landmarks)
        return delta > self.motion_threshold

    def should_run_physics(self) -> bool:
        """Whether to run a physics simulation step this frame."""
        return (self._frame_count % self.physics_interval) == 0

    def store(self, frame: CachedFrame):
        """Push a new keyframe into the cache."""
        self._cache.append(frame)
        self._last_landmarks = frame.landmarks
        self._frames_since_update = 0
        self._misses += 1

    def get_cached(self) -> Optional[CachedFrame]:
        """Return the most recent cached frame (for reuse)."""
        if not self._cache:
            return None
        self._frames_since_update += 1
        self._hits += 1
        return self._cache[-1]

    def get_interpolated_vertices(
        self,
        landmarks: Dict,
    ) -> Optional[np.ndarray]:
        """
        Interpolate vertex positions between the two nearest keyframes.

        Uses landmark similarity as the blending weight.
        Returns None if <2 keyframes are available.
        """
        if len(self._cache) < 2 or not self.interpolate:
            if self._cache:
                last = self._cache[-1]
                return last.wrapped_mesh.vertices if last.wrapped_mesh else None
            return None

        prev = self._cache[-2]
        curr = self._cache[-1]

        if prev.wrapped_mesh is None or curr.wrapped_mesh is None:
            return curr.wrapped_mesh.vertices if curr.wrapped_mesh else None

        # Blend weight: how far are incoming landmarks from the two keyframes?
        d_prev = self._landmark_delta(prev.landmarks, landmarks)
        d_curr = self._landmark_delta(curr.landmarks, landmarks)
        total = d_prev + d_curr + 1e-8
        alpha = d_prev / total  # closer to curr → higher alpha

        v = (1 - alpha) * prev.wrapped_mesh.vertices + alpha * curr.wrapped_mesh.vertices
        return v.astype(np.float32)

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict:
        return {
            'total_frames': self._frame_count,
            'cache_hits': self._hits,
            'cache_misses': self._misses,
            'hit_rate': f"{self.hit_rate:.1%}",
            'keyframes_stored': len(self._cache),
        }

    def reset(self):
        """Clear cache state."""
        self._cache.clear()
        self._frames_since_update = 0
        self._frame_count = 0
        self._last_landmarks = None
        self._hits = 0
        self._misses = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _landmark_delta(lm_a: Dict, lm_b: Dict) -> float:
        """
        Compute normalised L2 distance between two landmark dicts.

        Landmarks are expected as {idx: {'x': float, 'y': float, ...}}.
        Returns mean Euclidean distance across matching indices.
        """
        common = set(lm_a.keys()) & set(lm_b.keys())
        if not common:
            return 1.0  # force recompute

        deltas = []
        for idx in common:
            a = lm_a[idx]
            b = lm_b[idx]
            dx = a.get('x', 0) - b.get('x', 0)
            dy = a.get('y', 0) - b.get('y', 0)
            deltas.append(np.sqrt(dx * dx + dy * dy))

        return float(np.mean(deltas))


# ═══════════════════════════════════════════════════════════════════════════════
#  Integrated Cache Manager
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineCacheManager:
    """
    High-level orchestrator that wraps TemporalCache with
    the full 3D pipeline (SMPL → mesh → physics → render).

    Usage::

        mgr = PipelineCacheManager(smpl, wrapper, renderer, physics)
        for landmarks in live_stream:
            result = mgr.process_frame(landmarks, cloth_rgb, cloth_mask)
            # result is always a rendered RGBA image
    """

    def __init__(
        self,
        smpl_reconstructor=None,
        mesh_wrapper=None,
        renderer=None,
        physics_sim=None,
        cache_config: Optional[Dict] = None,
    ):
        self.smpl = smpl_reconstructor
        self.wrapper = mesh_wrapper
        self.renderer = renderer
        self.physics = physics_sim

        cfg = cache_config or {}
        self.cache = TemporalCache(
            motion_threshold=cfg.get('motion_threshold', 0.015),
            max_reuse_frames=cfg.get('max_reuse_frames', 5),
            physics_interval=cfg.get('physics_interval', 2),
            cache_size=cfg.get('cache_size', 30),
            interpolate=cfg.get('interpolate', True),
        )

    def process_frame(
        self,
        landmarks: Dict,
        cloth_rgb: np.ndarray,
        cloth_mask: np.ndarray,
        frame_shape: Tuple[int, int] = (480, 640),
        garment_type: str = 'tshirt',
        camera_matrix: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Process a single frame with temporal caching.

        Returns RGBA rendered image, or None if pipeline unavailable.
        """
        if self.smpl is None or self.wrapper is None:
            return None

        t0 = time.perf_counter()

        if self.cache.should_recompute(landmarks):
            # Full pipeline
            body_mesh = self.smpl.reconstruct(landmarks, frame_shape)
            if body_mesh is None:
                cached = self.cache.get_cached()
                return cached.rendered if cached else None

            from src.core.mesh_garment_wrapper import GarmentMesh
            garment_mesh = GarmentMesh.from_image(cloth_rgb, cloth_mask)
            wrapped = self.wrapper.wrap_garment(garment_mesh, body_mesh, garment_type)

            # Optional physics
            if self.physics and self.cache.should_run_physics():
                wrapped = self.physics.simulate_step(wrapped, body_mesh, num_iterations=5)

            # Render
            rendered = self._render(wrapped, camera_matrix, frame_shape)

            # Cache
            self.cache.store(CachedFrame(
                timestamp=time.perf_counter(),
                landmarks=landmarks,
                body_mesh=body_mesh,
                wrapped_mesh=wrapped,
                rendered=rendered,
            ))

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Cache MISS – full pipeline: {elapsed:.1f}ms")
            return rendered

        else:
            # Cache hit – reuse or interpolate
            cached = self.cache.get_cached()
            if cached is None or cached.rendered is None:
                return None

            if self.cache.interpolate and cached.wrapped_mesh is not None:
                interp_v = self.cache.get_interpolated_vertices(landmarks)
                if interp_v is not None:
                    cached.wrapped_mesh.vertices = interp_v
                    rendered = self._render(cached.wrapped_mesh, camera_matrix, frame_shape)
                    elapsed = (time.perf_counter() - t0) * 1000
                    logger.debug(f"Cache HIT (interpolated): {elapsed:.1f}ms")
                    return rendered

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Cache HIT (reuse): {elapsed:.1f}ms")
            return cached.rendered

    def _render(self, wrapped_mesh, camera_matrix, frame_shape):
        """Render using GPU or software renderer."""
        if self.renderer:
            return self.renderer.render_wrapped_mesh(
                wrapped_mesh,
                camera_matrix=camera_matrix,
                image_size=frame_shape,
            )
        else:
            return wrapped_mesh.render_to_image(
                camera_matrix=camera_matrix or np.eye(3, dtype=np.float32),
                image_size=frame_shape,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── Quick self-test ───────────────────────────────────────────────────────
    print("=" * 50)
    print("Temporal Cache Self-Test")
    print("=" * 50)

    cache = TemporalCache(motion_threshold=0.01, max_reuse_frames=3)

    # Simulate static landmark stream
    lm_static = {i: {'x': 0.5, 'y': float(i) / 33, 'visibility': 1.0} for i in range(33)}

    for frame_idx in range(10):
        recompute = cache.should_recompute(lm_static)
        if recompute:
            cache.store(CachedFrame(
                timestamp=time.perf_counter(),
                landmarks=lm_static,
            ))
            print(f"  Frame {frame_idx}: MISS (recompute)")
        else:
            cache.get_cached()
            print(f"  Frame {frame_idx}: HIT (reuse)")

    print(f"\nStats: {cache.stats}")

    # Simulate motion
    cache.reset()
    for frame_idx in range(10):
        # Move landmarks progressively
        lm_moving = {
            i: {'x': 0.5 + 0.05 * frame_idx, 'y': float(i) / 33, 'visibility': 1.0}
            for i in range(33)
        }
        recompute = cache.should_recompute(lm_moving)
        if recompute:
            cache.store(CachedFrame(timestamp=time.perf_counter(), landmarks=lm_moving))
            print(f"  Frame {frame_idx}: MISS (motion)")
        else:
            cache.get_cached()
            print(f"  Frame {frame_idx}: HIT")

    print(f"\nStats: {cache.stats}")
    print("Done.")

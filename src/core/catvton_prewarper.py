#!/usr/bin/env python3
"""
CatVTON Pre-Warper
==================
Uses the CatVTON diffusion-based virtual try-on model to produce a
high-quality body-conforming garment warp **once** per person/garment pair,
then caches the result for real-time compositing.

Why offline pre-warp?
- CatVTON produces far better side/curvature wrapping than TPS-based models.
- At ~3-5 s per inference on RTX 2050 it is too slow for per-frame use.
- Wrapping the garment once when the person enters the frame, caching it,
  and recompositing it in real-time gives the best quality-vs-latency tradeoff.

Pipeline
--------
1. Caller detects a new person / significant pose change via ``needs_rewarp()``.
2. Caller calls ``prewarp(person_image, garment_image)`` — blocks ~3-5 s once.
3. Caller uses ``cached_result`` each subsequent frame for compositing.

Install (optional — module degrades gracefully if absent):
    pip install diffusers transformers accelerate
    # Weights are downloaded automatically from HuggingFace on first call.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── dependency checks ──────────────────────────────────────────────────────────
_CATVTON_AVAILABLE = False
try:
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    _CATVTON_AVAILABLE = True
except ImportError:
    logger.info("CatVTON: diffusers / transformers not installed — pre-warper disabled. "
                "Install with:  pip install diffusers transformers accelerate")

# CatVTON repo ID — correct as of 2026 (no hyphen, lowercase).
# Alternatives: "zhengchong/CatVTON-MaskFree"
_MODEL_ID = "zhengchong/CatVTON"


# ── SSIM-based pose-change detector ───────────────────────────────────────────

def _ssim_patch(a: np.ndarray, b: np.ndarray) -> float:
    """Fast approximate SSIM over downscaled grayscale patch."""
    size = (64, 64)
    ga = cv2.cvtColor(cv2.resize(a, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cv2.resize(b, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu_a, mu_b = ga.mean(), gb.mean()
    sigma_a = ga.std() + 1e-6
    sigma_b = gb.std() + 1e-6
    sigma_ab = float(np.mean((ga - mu_a) * (gb - mu_b)))
    C1, C2 = 6.5025, 58.5225
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
           ((mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2))
    return float(ssim)


class CatVTONPrewarper:
    """
    Offline CatVTON warper with SSIM-based pose-change detection.

    Parameters
    ----------
    rewarp_threshold : float
        SSIM drop below this value triggers a new pre-warp (default 0.82).
        Lower = rewarp more often (higher quality but more latency events).
    half_precision : bool
        Use float16 to halve VRAM usage (recommended on RTX 2050).
    device : str
        'cuda' or 'cpu'.  'cuda' strongly recommended.
    """

    def __init__(
        self,
        rewarp_threshold: float = 0.82,
        half_precision: bool = True,
        device: str = "cuda",
    ) -> None:
        self.rewarp_threshold = rewarp_threshold
        self.half_precision = half_precision
        self.device = device

        self._pipeline = None          # loaded lazily
        self._cached_result: Optional[np.ndarray] = None   # (H, W, 3) RGB float32
        self._cached_mask:   Optional[np.ndarray] = None   # (H, W) float32
        self._ref_frame:     Optional[np.ndarray] = None   # last frame used for warp
        self._ref_garment_hash: Optional[int] = None
        self._warp_count: int = 0
        self._is_available: bool = _CATVTON_AVAILABLE

        if not _CATVTON_AVAILABLE:
            logger.warning("CatVTON pre-warper disabled (diffusers not installed).")
        else:
            # Quick local-cache check — avoids a blocking network call during the
            # render loop.  If the model is not cached we disable immediately.
            self._is_available = self._check_local_cache()
            if not self._is_available:
                logger.info(
                    "CatVTON model not found in local HuggingFace cache. "
                    "To enable, either:\n"
                    "  1) Accept terms and login: huggingface-cli login\n"
                    "     then run: python -c \"from diffusers import DiffusionPipeline; "
                    "DiffusionPipeline.from_pretrained('%s')\"\n"
                    "  2) Or download manually and place weights in:"
                    " ~/.cache/huggingface/hub/",
                    _MODEL_ID,
                )

    # ── public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def _check_local_cache() -> bool:
        """Return True only if the model weights are already cached locally."""
        try:
            from huggingface_hub import scan_cache_dir  # type: ignore
            hf_cache = scan_cache_dir()
            for repo in hf_cache.repos:
                if repo.repo_id.lower() == _MODEL_ID.lower():
                    return True
            return False
        except Exception:
            # If we can't scan the cache, let the later network call decide.
            return True

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def cached_result(self) -> Optional[np.ndarray]:
        """Last pre-warped garment image as float32 RGB [0,1], shape (H,W,3)."""
        return self._cached_result

    @property
    def cached_mask(self) -> Optional[np.ndarray]:
        """Mask for last pre-warped garment, float32 [0,1], shape (H,W)."""
        return self._cached_mask

    def needs_rewarp(
        self,
        current_frame: np.ndarray,
        current_garment: np.ndarray,
    ) -> bool:
        """
        Return True if a new pre-warp should be triggered.

        Triggers when:
        - No cached result exists yet, OR
        - Garment has changed (different numpy data), OR
        - Person's pose has shifted significantly (SSIM < threshold).
        """
        if self._cached_result is None:
            return True

        # Check garment identity via hash of first 1024 bytes
        garment_hash = hash(current_garment.tobytes()[:1024])
        if garment_hash != self._ref_garment_hash:
            return True

        # Pose-change check via SSIM
        if self._ref_frame is not None:
            ssim = _ssim_patch(current_frame, self._ref_frame)
            if ssim < self.rewarp_threshold:
                logger.debug(f"CatVTON: pose change detected (SSIM={ssim:.3f}), rewarp needed")
                return True

        return False

    def prewarp(
        self,
        person_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        output_size: tuple[int, int] = (512, 384),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run CatVTON inference once and cache the result.

        Parameters
        ----------
        person_bgr  : BGR uint8 camera frame
        garment_bgr : BGR uint8 garment image
        output_size : (width, height) for output — default (512, 384)

        Returns
        -------
        warped_rgb  : float32 [0,1] (H, W, 3)
        warped_mask : float32 [0,1] (H, W)
        """
        if not self._is_available:
            logger.warning("CatVTON not available — returning resize fallback.")
            return self._fallback_resize(person_bgr, garment_bgr, output_size)

        try:
            pipeline = self._get_pipeline()
            t0 = time.perf_counter()

            # Prepare inputs — CatVTON expects PIL images
            from PIL import Image as PILImage

            ow, oh = output_size
            person_rgb = cv2.cvtColor(
                cv2.resize(person_bgr, (ow, oh)), cv2.COLOR_BGR2RGB
            )
            garment_rgb = cv2.cvtColor(
                cv2.resize(garment_bgr, (ow, oh)), cv2.COLOR_BGR2RGB
            )

            person_pil  = PILImage.fromarray(person_rgb)
            garment_pil = PILImage.fromarray(garment_rgb)

            # ── Run CatVTON pipeline ──────────────────────────────────────────
            result_pil = pipeline(
                person_image=person_pil,
                garment_image=garment_pil,
                num_inference_steps=20,       # balanced quality / speed
                guidance_scale=2.5,
            ).images[0]

            elapsed = time.perf_counter() - t0
            logger.info(f"CatVTON pre-warp completed in {elapsed:.1f}s")

            # Convert to float32 numpy
            warped_rgb = np.array(result_pil).astype(np.float32) / 255.0
            # Build mask: pixels that differ significantly from original person
            diff = np.abs(warped_rgb - person_rgb.astype(np.float32) / 255.0)
            warped_mask = np.clip(diff.mean(axis=2) * 3.0, 0.0, 1.0)

            # Cache
            self._cached_result = warped_rgb
            self._cached_mask   = warped_mask
            self._ref_frame     = person_bgr.copy()
            self._ref_garment_hash = hash(garment_bgr.tobytes()[:1024])
            self._warp_count   += 1

            return warped_rgb, warped_mask

        except Exception as exc:
            # Disable permanently so we don't retry every frame
            self._is_available = False
            logger.warning(f"CatVTON unavailable (disabling): {exc}")
            return self._fallback_resize(person_bgr, garment_bgr, output_size)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_pipeline(self):
        """Lazy-load the CatVTON pipeline from HuggingFace (first call only)."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info(f"Loading CatVTON from '{_MODEL_ID}' (local cache only — ~2 GB)")

        # Import here to avoid hard dependency at module load time
        from diffusers import DiffusionPipeline  # type: ignore
        import torch

        dtype = torch.float16 if (self.half_precision and self.device == "cuda") else torch.float32

        try:
            # local_files_only=True — never attempt a network download during
            # the render loop.  If this fails the _check_local_cache guard
            # should have set _is_available=False already.
            pipe = DiffusionPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
            pipe = pipe.to(self.device)
            # Memory optimisations for 4 GB VRAM (RTX 2050)
            if self.device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass

            self._pipeline = pipe
            logger.info("CatVTON pipeline loaded successfully.")
        except Exception as exc:
            _short = str(exc).split("\n")[0][:120]
            logger.warning(f"Failed to load CatVTON pipeline: {_short}")
            self._is_available = False
            raise

        return self._pipeline

    @staticmethod
    def _fallback_resize(
        person_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        output_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple resize fallback when CatVTON is unavailable."""
        ow, oh = output_size
        g_rgb = cv2.cvtColor(cv2.resize(garment_bgr, (ow, oh)), cv2.COLOR_BGR2RGB)
        warped_rgb = g_rgb.astype(np.float32) / 255.0
        # Mask: anything non-white in the garment
        gray = cv2.cvtColor(g_rgb, cv2.COLOR_RGB2GRAY)
        warped_mask = (gray < 240).astype(np.float32)
        return warped_rgb, warped_mask


# ── module-level singleton (shared across all GarmentRenderer instances) ──────
_global_prewarper: Optional[CatVTONPrewarper] = None


def get_prewarper(
    rewarp_threshold: float = 0.82,
    half_precision: bool = True,
    device: str = "cuda",
) -> CatVTONPrewarper:
    """Return (or create) the module-level CatVTON pre-warper singleton."""
    global _global_prewarper
    if _global_prewarper is None:
        _global_prewarper = CatVTONPrewarper(
            rewarp_threshold=rewarp_threshold,
            half_precision=half_precision,
            device=device,
        )
    return _global_prewarper

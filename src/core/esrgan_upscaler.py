"""
ESRGAN Garment Texture Upscaler  (Phase A)
==========================================
Lazy singleton that wraps Real-ESRGAN x4plus to increase garment texture
resolution before it reaches the GPU renderer.

Usage
-----
    from src.core.esrgan_upscaler import get_upscaler
    upscaler = get_upscaler()
    cloth_rgb_hi = upscaler.upscale(cloth_rgb)   # float32 RGB [0,1]

Fallback
--------
If the ``realesrgan`` package is not installed, or the model weights cannot be
downloaded, the upscaler silently falls back to Lanczos×4 resizing so the
pipeline never breaks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchvision ≥ 0.17 compatibility shim
# basicsr/realesrgan still imports the removed functional_tensor submodule.
# We re-expose all functions from torchvision.transforms.functional under
# that legacy name so basicsr does not crash on import.
# ---------------------------------------------------------------------------
def _patch_torchvision_functional_tensor() -> None:
    import sys, types
    _LEGACY = "torchvision.transforms.functional_tensor"
    if _LEGACY in sys.modules:
        return
    try:
        import torchvision.transforms.functional as _F
        _ft = types.ModuleType(_LEGACY)
        for _name in dir(_F):
            if not _name.startswith("__"):
                setattr(_ft, _name, getattr(_F, _name))
        sys.modules[_LEGACY] = _ft
    except Exception:
        pass  # If torchvision itself isn't installed, the import below will also fail cleanly.

_patch_torchvision_functional_tensor()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_WEIGHT_FILENAME = "RealESRGAN_x4plus.pth"
_WEIGHT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
_SCALE = 4

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional["ESRGANUpscaler"] = None


def get_upscaler() -> "ESRGANUpscaler":
    """Return the process-wide ESRGANUpscaler singleton (lazy init)."""
    global _instance
    if _instance is None:
        _instance = ESRGANUpscaler()
    return _instance


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class ESRGANUpscaler:
    """
    4× garment texture upscaler backed by Real-ESRGAN x4plus.

    The class tries, in order:
      1. Import ``realesrgan`` and load weights (auto-download if absent).
      2. Lanczos×4 fallback (always available).

    After construction, ``self.available`` indicates which backend is active.
    """

    def __init__(self) -> None:
        self._model = None
        self.available = False
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upscale(self, img_float32_rgb: np.ndarray) -> np.ndarray:
        """
        Upscale a garment image 4×.

        Parameters
        ----------
        img_float32_rgb:
            float32 numpy array, shape (H, W, 3), values in [0, 1], RGB.

        Returns
        -------
        float32 numpy array, shape (H×4, W×4, 3), values in [0, 1], RGB.
        """
        if img_float32_rgb is None:
            return img_float32_rgb

        h, w = img_float32_rgb.shape[:2]

        if self.available and self._model is not None:
            return self._upscale_esrgan(img_float32_rgb)

        # Lanczos fallback
        return self._upscale_lanczos(img_float32_rgb, w * _SCALE, h * _SCALE)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to import realesrgan and load the x4plus weights."""
        try:
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            import torch

            weight_path = _MODEL_DIR / _WEIGHT_FILENAME
            if not weight_path.exists():
                self._download_weights(weight_path)

            if not weight_path.exists():
                logger.warning("ESRGAN weights not found; using Lanczos fallback.")
                return

            device = "cuda" if self._cuda_available() else "cpu"

            # Skip ESRGAN on CPU — 4× upscale takes 3+ minutes per garment.
            # The Lanczos fallback produces good-enough quality in milliseconds.
            if device == "cpu":
                logger.info(
                    "ESRGANUpscaler: CUDA not available — using Lanczos fallback "
                    "(ESRGAN skipped to avoid multi-minute blocking on CPU)."
                )
                return

            arch = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=_SCALE
            )
            self._model = RealESRGANer(
                scale=_SCALE,
                model_path=str(weight_path),
                model=arch,
                tile=256,           # tile to avoid OOM on large garments
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                device=device,
            )
            self.available = True
            logger.info("ESRGANUpscaler: Real-ESRGAN x4plus loaded on %s.", device)

        except ImportError:
            logger.info(
                "ESRGANUpscaler: 'realesrgan' package not installed; "
                "using Lanczos×4 fallback. "
                "Install with: pip install realesrgan"
            )
        except Exception as exc:
            logger.warning("ESRGANUpscaler: failed to load model (%s); using fallback.", exc)

    def _upscale_esrgan(self, img: np.ndarray) -> np.ndarray:
        """Run Real-ESRGAN inference."""
        try:
            # realesrgan expects uint8 BGR
            img_uint8_bgr = cv2.cvtColor(
                (img * 255.0).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
            output_bgr, _ = self._model.enhance(img_uint8_bgr, outscale=_SCALE)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            return output_rgb.astype(np.float32) / 255.0
        except Exception as exc:
            logger.warning("ESRGAN inference failed (%s); falling back to Lanczos.", exc)
            h, w = img.shape[:2]
            return self._upscale_lanczos(img, w * _SCALE, h * _SCALE)

    @staticmethod
    def _upscale_lanczos(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """High-quality bicubic upscale as a drop-in fallback."""
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        return resized.astype(np.float32)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _download_weights(dest: Path) -> None:
        """Try to download model weights with urllib."""
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading ESRGAN weights to %s …", dest)
        try:
            urllib.request.urlretrieve(_WEIGHT_URL, str(dest))
            logger.info("ESRGAN weights downloaded successfully.")
        except Exception as exc:
            logger.warning("Could not download ESRGAN weights: %s", exc)

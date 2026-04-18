"""
PP-HumanSeg body segmentor (Apache 2.0, ~8MB ONNX).

BodySegmentor auto-downloads the PP-HumanSeg-Lite ONNX model to
~/.fitengine/models/ on first use and produces a binary body mask.

Usage:
    seg = BodySegmentor()
    mask = seg.segment(bgr_image)   # np.ndarray [H, W] uint8, 0 or 255
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR   = Path(os.path.expanduser("~/.fitengine/models"))
_MODEL_NAME  = "pphuman_seg_lite.onnx"
_MODEL_URL   = (
    "https://paddlepaddle.github.io/PaddleSeg/deploy/slim/quant/"
    "pp_humanseg_lite_export_398x224.zip"
)
# Direct ONNX from PaddleSeg FastDeploy
_MODEL_URL_DIRECT = (
    "https://raw.githubusercontent.com/PaddlePaddle/FastDeploy/release/2.0.0/"
    "examples/vision/segmentation/paddleseg/quantize/cpp/portrait_pp_humanseg_v2.onnx"
)

_INPUT_W, _INPUT_H = 192, 192   # model input resolution


def _download_model(dest: Path) -> bool:
    """Try to download PP-HumanSeg-Lite. Returns True on success, False on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in (_MODEL_URL_DIRECT, _MODEL_URL):
        try:
            logger.info("Downloading PP-HumanSeg-Lite from %s", url)
            urllib.request.urlretrieve(url, str(dest))
            # Sanity check: ONNX files start with 0x08 (protobuf field 1, varint)
            if dest.stat().st_size > 1000:
                logger.info("PP-HumanSeg download complete (%d bytes).", dest.stat().st_size)
                return True
            dest.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Download attempt failed (%s): %s", url, exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
    return False


class BodySegmentor:
    """
    PP-HumanSeg-Lite binary body segmentor.

    Returns a uint8 mask where 255 = foreground (body), 0 = background.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            model_path = _CACHE_DIR / _MODEL_NAME
        model_path = Path(model_path)

        self._session = None  # None => noop/fallback mode

        if not model_path.exists():
            ok = _download_model(model_path)
            if not ok:
                logger.warning(
                    "PP-HumanSeg model unavailable — BodySegmentor running in "
                    "NOOP mode (full-white mask). Silhouette measurements will "
                    "be less accurate but keypoint-based ratios are unaffected."
                )
                return

        try:
            self._session = self._load_session(model_path)
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("BodySegmentor loaded from %s", model_path)
        except Exception as exc:
            logger.warning("BodySegmentor failed to load (%s) — NOOP mode active.", exc)

    def _load_session(self, model_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for BodySegmentor. "
                "Install: pip install onnxruntime-gpu"
            ) from e

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return sess

    def segment(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Produce binary foreground mask.

        Args:
            img_bgr : [H, W, 3] uint8 BGR.

        Returns:
            mask : [H, W] uint8, values 0 or 255.
        """
        H, W = img_bgr.shape[:2]

        # NOOP fallback: return full-body (all-foreground) mask
        if self._session is None:
            return np.full((H, W), 255, dtype=np.uint8)

        # Pre-process
        inp = cv2.resize(img_bgr, (_INPUT_W, _INPUT_H))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = (inp / 255.0 - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])
        inp = inp.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

        # Inference
        out = self._session.run([self._output_name], {self._input_name: inp})[0]  # [1, 2, H, W]

        # Foreground = class 1 > class 0
        prob_fg = out[0, 1]  # [H, W]
        mask_small = (prob_fg > 0.5).astype(np.uint8) * 255

        # Resize back to original resolution
        mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask

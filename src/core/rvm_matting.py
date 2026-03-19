"""
RobustVideoMatting Body Matting  (Phase C)
==========================================
Wraps the RVM MobileNetV3 ONNX model for per-frame alpha-matte generation.
The recurrent hidden states are maintained across frames so the model produces
temporally stable mattes without per-frame flicker.

Usage
-----
    from src.core.rvm_matting import RVMMatting
    rvm = RVMMatting()
    alpha = rvm.matte(bgr_frame)   # float32 (H, W) in [0, 1]

Fallback
--------
Returns None when ONNX Runtime is unavailable or the model fails to load,
signalling the caller to fall back to its existing segmentation backend.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_ONNX_FILENAME = "rvm_mobilenetv3_fp32.onnx"
_ONNX_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
    "v1.0.0/rvm_mobilenetv3_fp32.onnx"
)

# Scene-cut threshold: if mean alpha changes by more than this, reset hidden state
_SCENE_CUT_THRESHOLD = 0.30

# Inference resolution (model accepts any, but we cap for speed)
_INF_W = 640
_INF_H = 480


class RVMMatting:
    """
    Real-time alpha-matte using RobustVideoMatting (MobileNetV3 ONNX).

    The four recurrent hidden-state tensors (r1..r4) are kept in memory and
    passed back into each inference call, giving the model temporal context
    for artifact-free mattes at video rates.

    Attributes
    ----------
    available : bool
        True if the ONNX model was loaded successfully.
    """

    def __init__(self, downsample_ratio: float = 0.25) -> None:
        """
        Parameters
        ----------
        downsample_ratio:
            Passed to the model as the ``downsample_ratio`` input.
            Lower values → faster, less fine-detail.  0.25 is good at 30 FPS.
        """
        self.downsample_ratio = float(downsample_ratio)
        self.available = False
        self._session = None
        self._hidden: Optional[list[np.ndarray]] = None  # [r1, r2, r3, r4]
        self._prev_alpha_mean: Optional[float] = None
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matte(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a soft alpha matte for the person in the frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            uint8 BGR frame from OpenCV, shape (H, W, 3).

        Returns
        -------
        alpha : np.ndarray | None
            float32 array of shape (H, W), values in [0, 1], where 1 = fully
            foreground (person).  Returns None if the model is unavailable.
        """
        if not self.available or self._session is None:
            return None

        try:
            orig_h, orig_w = bgr_frame.shape[:2]

            # ── Preprocess ──────────────────────────────────────────────
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(rgb, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)
            inp_f32 = inp.astype(np.float32) / 255.0                  # [0,1]
            inp_nchw = inp_f32.transpose(2, 0, 1)[None, ...]          # (1,3,H,W)

            # ── Build hidden-state inputs ────────────────────────────────
            if self._hidden is None:
                self._hidden = self._zero_hidden()

            r1, r2, r3, r4 = self._hidden

            # ── Run ONNX inference ───────────────────────────────────────
            outputs = self._session.run(
                None,
                {
                    "src": inp_nchw,
                    "r1i": r1, "r2i": r2, "r3i": r3, "r4i": r4,
                    "downsample_ratio": np.array(
                        [self.downsample_ratio], dtype=np.float32
                    ),
                },
            )
            # outputs: [fgr, pha, r1o, r2o, r3o, r4o]
            fgr_out, pha_out, r1o, r2o, r3o, r4o = outputs

            # ── Detect scene cut → reset hidden state ────────────────────
            alpha_mean = float(pha_out.mean())
            if (
                self._prev_alpha_mean is not None
                and abs(alpha_mean - self._prev_alpha_mean) > _SCENE_CUT_THRESHOLD
            ):
                logger.debug("RVM: scene cut detected — resetting hidden state.")
                self._hidden = self._zero_hidden()
            else:
                self._hidden = [r1o, r2o, r3o, r4o]
            self._prev_alpha_mean = alpha_mean

            # ── Postprocess ──────────────────────────────────────────────
            alpha_small = pha_out[0, 0]                # (inf_H, inf_W)
            alpha_full = cv2.resize(
                alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            return np.clip(alpha_full, 0.0, 1.0).astype(np.float32)

        except Exception as exc:
            logger.warning("RVMMatting inference error: %s", exc)
            self._hidden = None   # reset so next frame re-initialises cleanly
            return None

    def reset(self) -> None:
        """Manually reset hidden state (call on garment change or scene jump)."""
        self._hidden = None
        self._prev_alpha_mean = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Load (and optionally download) the ONNX model."""
        try:
            import onnxruntime as ort  # type: ignore

            weight_path = _MODEL_DIR / _ONNX_FILENAME
            if not weight_path.exists():
                self._download_model(weight_path)

            if not weight_path.exists():
                logger.warning("RVM ONNX model not found; matting unavailable.")
                return

            # Prefer CUDA EP, fall back to CPU
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._cuda_available()
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(str(weight_path), providers=providers)
            self.available = True
            active_ep = self._session.get_providers()[0]
            logger.info("RVMMatting: model loaded (EP=%s).", active_ep)

        except ImportError:
            logger.info(
                "RVMMatting: 'onnxruntime' not installed; matting unavailable. "
                "Install with: pip install onnxruntime-gpu"
            )
        except Exception as exc:
            logger.warning("RVMMatting: failed to load model (%s).", exc)

    def _zero_hidden(self) -> list[np.ndarray]:
        """Return zero-initialised hidden states that broadcast to any shape.

        The ONNX Expand node inside RVM broadcasts the recurrent state to its
        internally-computed spatial size.  Using (1,C,1,1) tensors avoids any
        shape-prediction arithmetic — they broadcast cleanly to whatever spatial
        dimensions the model requires at each encoder level.
        """
        return [
            np.zeros((1, 16, 1, 1), dtype=np.float32),
            np.zeros((1, 20, 1, 1), dtype=np.float32),
            np.zeros((1, 40, 1, 1), dtype=np.float32),
            np.zeros((1, 64, 1, 1), dtype=np.float32),
        ]

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _download_model(dest: Path) -> None:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading RVM ONNX model to %s …", dest)
        try:
            urllib.request.urlretrieve(_ONNX_URL, str(dest))
            logger.info("RVM model downloaded successfully.")
        except Exception as exc:
            logger.warning("Could not download RVM model: %s", exc)

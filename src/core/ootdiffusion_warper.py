"""
OOTDiffusion Async Warper  (Phase E)
=====================================
Runs OOTDiffusion in a background thread so the main rendering loop is never
blocked.  The caller submits a (person, garment) pair and polls for the result
asynchronously.

Usage
-----
    from src.core.ootdiffusion_warper import OOTDiffusionWarper
    warper = OOTDiffusionWarper()
    warper.submit(person_bgr, garment_bgr, garment_type="upper_body")
    ...
    result = warper.poll()   # None until ready, then np.ndarray BGR uint8

Architecture
------------
* Background daemon thread consumes a ``queue.Queue`` of (person, garment, type)
  tuples and writes results to ``_result_box`` (a 1-slot deque).
* Every ~90 frames (≈3 s @ 30 FPS) the pipeline triggers a new submission.
* On garment change the frame counter is reset so a fresh diffusion starts
  immediately.
* A 3-level compositing waterfall is used at render time:
    1. OOTDiffusion cached result  (highest realism)
    2. CatVTON cached result       (medium realism, faster)
    3. Phase-3 SMPL mesh warp      (real-time fallback)

Fallback
--------
If ``diffusers`` / ``transformers`` are not installed, ``available`` is False
and all calls are no-ops so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HF_MODEL_ID = "levihsu/OOTDiffusion"
_NUM_INFERENCE_STEPS = 20
_GUIDANCE_SCALE = 2.0
_OUTPUT_SIZE = (768, 1024)   # (W, H)  – OOTDiffusion canonical size
_TRIGGER_EVERY_N_FRAMES = 90


class OOTDiffusionWarper:
    """
    Asynchronous OOTDiffusion virtual try-on.

    The diffusion pipeline runs in a dedicated daemon thread.  The main render
    loop calls :meth:`submit` (non-blocking) and :meth:`poll` (returns the
    latest diffusion result or ``None``).
    """

    def __init__(self, garment_type: str = "upper_body") -> None:
        """
        Parameters
        ----------
        garment_type:
            OOTDiffusion category: ``"upper_body"``, ``"lower_body"``, or
            ``"dress"``.
        """
        self.garment_type = garment_type
        self.available = False
        self._pipeline = None

        # Thread-safe communication
        self._task_queue: queue.Queue = queue.Queue(maxsize=1)   # drop if busy
        self._result_box: deque = deque(maxlen=1)                 # latest result

        self._frame_counter: int = 0
        self._last_garment_hash: Optional[int] = None

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._try_init_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_submit(
        self,
        garment_bgr: np.ndarray,
        frame_counter: Optional[int] = None
    ) -> bool:
        """
        Return True when the caller should call :meth:`submit`.

        Triggers on:
        * First call (no cached result yet).
        * Garment change detected (pixel-hash differs).
        * Every ``_TRIGGER_EVERY_N_FRAMES`` frames.
        """
        if not self.available:
            return False

        g_hash = hash(garment_bgr.tobytes()) if garment_bgr is not None else None
        garment_changed = g_hash != self._last_garment_hash

        if frame_counter is not None:
            self._frame_counter = frame_counter

        trigger = (
            garment_changed
            or len(self._result_box) == 0
            or (self._frame_counter % _TRIGGER_EVERY_N_FRAMES == 0)
        )

        if garment_changed:
            self._last_garment_hash = g_hash
            self._frame_counter = 0

        self._frame_counter += 1
        return trigger

    def submit(
        self,
        person_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        garment_type: Optional[str] = None,
    ) -> None:
        """
        Enqueue a diffusion job (non-blocking).

        If a job is already queued, the old one is discarded in favour of the
        latest frames (only the most recent context is useful).
        """
        if not self.available:
            return

        try:
            # Discard any pending job (put_nowait raises Full if queue is full)
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except queue.Empty:
                    break

            self._task_queue.put_nowait(
                (
                    person_bgr.copy(),
                    garment_bgr.copy(),
                    garment_type or self.garment_type,
                )
            )
        except queue.Full:
            pass  # Worker is busy; skip this frame

    def poll(self) -> Optional[np.ndarray]:
        """
        Return the most recent OOTDiffusion result, or ``None`` if not yet ready.

        Returns
        -------
        np.ndarray | None
            uint8 BGR image of shape (H, W, 3), or None.
        """
        if self._result_box:
            return self._result_box[-1]
        return None

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def __del__(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_init_pipeline(self) -> None:
        """Load the diffusion pipeline in the background thread."""
        try:
            import diffusers  # noqa: F401 – check import works
            import transformers  # noqa: F401

            self.available = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="ootd-diffuser",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("OOTDiffusionWarper: background thread started.")

        except ImportError:
            logger.info(
                "OOTDiffusionWarper: 'diffusers'/'transformers' not installed; "
                "async diffusion disabled. "
                "Install with: pip install diffusers transformers accelerate"
            )
        except Exception as exc:
            logger.warning("OOTDiffusionWarper: init failed (%s).", exc)

    def _worker_loop(self) -> None:
        """Background thread: load model once, then process jobs."""
        pipeline = self._load_pipeline()
        if pipeline is None:
            self.available = False
            return

        self._pipeline = pipeline
        logger.info("OOTDiffusionWarper: pipeline ready.")

        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            person_bgr, garment_bgr, g_type = task
            result_bgr = self._run_diffusion(pipeline, person_bgr, garment_bgr, g_type)
            if result_bgr is not None:
                self._result_box.append(result_bgr)

        logger.info("OOTDiffusionWarper: worker thread exiting.")

    def _load_pipeline(self):
        """Load OOTDiffusion pipeline (runs once in worker thread)."""
        try:
            # OOTDiffusion is distributed as a custom pipeline through diffusers
            from diffusers import DiffusionPipeline  # type: ignore
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(
                "OOTDiffusionWarper: loading %s on %s …", _HF_MODEL_ID, device
            )
            pipe = DiffusionPipeline.from_pretrained(
                _HF_MODEL_ID,
                custom_pipeline=_HF_MODEL_ID,
                torch_dtype=dtype,
            )
            pipe = pipe.to(device)
            # Memory efficiency: only needed when not using FP16 flash-attn
            if device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            return pipe

        except Exception as exc:
            logger.error("OOTDiffusionWarper: could not load pipeline: %s", exc)
            return None

    def _run_diffusion(
        self,
        pipeline,
        person_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        garment_type: str,
    ) -> Optional[np.ndarray]:
        """Run one diffusion pass and return BGR uint8 result."""
        try:
            from PIL import Image  # type: ignore
            import torch

            ow, oh = _OUTPUT_SIZE
            person_rgb = cv2.cvtColor(
                cv2.resize(person_bgr, (ow, oh), interpolation=cv2.INTER_LANCZOS4),
                cv2.COLOR_BGR2RGB,
            )
            garment_rgb = cv2.cvtColor(
                cv2.resize(garment_bgr, (ow, oh), interpolation=cv2.INTER_LANCZOS4),
                cv2.COLOR_BGR2RGB,
            )

            person_pil = Image.fromarray(person_rgb)
            garment_pil = Image.fromarray(garment_rgb)

            with torch.inference_mode():
                output = pipeline(
                    image=person_pil,
                    cloth_image=garment_pil,
                    cloth_type=garment_type,
                    num_inference_steps=_NUM_INFERENCE_STEPS,
                    guidance_scale=_GUIDANCE_SCALE,
                ).images[0]

            result_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
            return result_bgr

        except Exception as exc:
            logger.warning("OOTDiffusion inference failed: %s", exc)
            return None

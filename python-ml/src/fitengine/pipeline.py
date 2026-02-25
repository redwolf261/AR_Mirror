"""
FitEnginePipeline — end-to-end size prediction.

Phase 1:  front_img + side_img + height_cm
          → PoseDetector → BodySegmentor → AlignmentGuide
          → SilhouetteExtractor → HeuristicEstimator → SizeChart → JSON

Phase 2+: HeuristicEstimator is silently replaced by DualViewRegressor
          via self._estimator.  Callers (ui.py, /fit-size endpoint) see
          no interface change.

Usage:
    pipeline = FitEnginePipeline(chart="generic")
    result   = pipeline.predict(front_bgr, side_bgr, height_cm=178)
    # → {"collar": "16.0", "jacket": "42R", "trouser_waist": "34",
    #    "confidence_level": "heuristic"}
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .alignment import AlignmentGuide
from .detector import PoseDetector
from .heuristic.estimator import HeuristicEstimator
from .heuristic.pilot_logger import PilotLogger
from .measurements import BodyProxyMeasurements
from .segmentor import BodySegmentor
from .silhouette import SilhouetteExtractor
from .size_chart import SizeChart

logger = logging.getLogger(__name__)


class FitEnginePipeline:
    """
    Single entry point for size prediction.

    The _estimator attribute is the only point that changes between
    Phase 1 (heuristic) and Phase 2 (DualViewRegressor).

    Args:
        chart            : brand name or SizeChart instance.
        log_path         : path to pilot_log.jsonl (None = no logging).
        score_threshold  : minimum keypoint confidence passed to PoseDetector.
    """

    def __init__(
        self,
        chart: str | SizeChart = "generic",
        log_path: Optional[str | Path] = "data/pilot_log.jsonl",
        score_threshold: float = 0.3,
    ) -> None:
        self._chart = chart if isinstance(chart, SizeChart) else SizeChart(chart)
        self._detector   = PoseDetector(score_threshold=score_threshold)
        self._segmentor  = BodySegmentor()
        self._silhouette = SilhouetteExtractor()
        self._guide      = AlignmentGuide()
        self._logger     = PilotLogger(log_path) if log_path else None

        # Estimator priority: DualStreamCNN → RatioRegressor → Heuristic
        from .image_net import load_dual_stream
        from .heuristic.ratio_regressor import load_ratio_regressor

        cnn_estimator = load_dual_stream()
        ml_estimator  = load_ratio_regressor()

        if cnn_estimator is not None:
            self._estimator = cnn_estimator
            logger.info("Using DualStreamCNN estimator (CNN model loaded)")
        elif ml_estimator is not None:
            self._estimator = ml_estimator
            logger.info("Using RatioRegressorEstimator (ML model loaded)")
        else:
            self._estimator = HeuristicEstimator(chart=self._chart)
            logger.info("Using HeuristicEstimator (no ML checkpoint found)")

    def predict(
        self,
        front_bgr: np.ndarray,
        side_bgr:  np.ndarray,
        height_cm: float,
        session_id: Optional[str] = None,
        skip_alignment_check: bool = False,
    ) -> dict:
        """
        Predict men's formalwear sizes from a front + side photo pair.

        Args:
            front_bgr            : [H, W, 3] uint8 BGR front-view image.
            side_bgr             : [H, W, 3] uint8 BGR side-view image.
            height_cm            : user-reported height in cm.
            session_id           : if provided, logs to pilot_log.jsonl.
            skip_alignment_check : if True, skip alignment quality gate
                                   (used for batch evaluation).

        Returns:
            {
                "collar":           "16.0",
                "jacket":           "42R",
                "trouser_waist":    "34",
                "confidence_level": "heuristic" | "High" | "Medium" | "Low",
                "alignment_ok":     True | False,
                "alignment_hint":   str,
            }
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # 1. Detect keypoints in both views
        kp_front = self._detector.detect(front_bgr)
        kp_side  = self._detector.detect(side_bgr)

        # 2. Alignment quality check (front view)
        alignment_ok, alignment_hint = self._guide.check_front(kp_front)
        if not skip_alignment_check and not alignment_ok:
            logger.debug("Alignment check failed: %s", alignment_hint)

        side_ok, side_hint = self._guide.check_side(kp_side)
        if not skip_alignment_check and not side_ok:
            logger.debug("Side alignment check failed: %s", side_hint)

        # 3. Segment + silhouette curves
        mask_front = self._segmentor.segment(front_bgr)
        mask_side  = self._segmentor.segment(side_bgr)
        width_front = self._silhouette.from_mask(mask_front)
        width_side  = self._silhouette.from_mask(mask_side)

        # 4. Scale-invariant proxy measurements
        m = BodyProxyMeasurements.from_keypoints(kp_front, kp_side)

        # 5. Size estimation (heuristic Phase 1 / regressor Phase 2+)
        result = self._estimator.predict(kp_front, kp_side, height_cm)

        # 6. Log session
        if self._logger:
            measurement_ratios = {
                "shoulder_width_ratio": m.shoulder_width_ratio,
                "chest_width_ratio":    m.chest_width_ratio,
                "hip_width_ratio":      m.hip_width_ratio,
                "torso_depth_ratio":    m.torso_depth_ratio,
                "height_norm":          (height_cm - 175.0) / 12.0,
            }
            self._logger.log_session(
                session_id=session_id,
                kp_front=kp_front,
                kp_side=kp_side,
                height_cm=height_cm,
                predicted_size=result,
                measurement_ratios=measurement_ratios,
            )

        result["alignment_ok"]   = alignment_ok and side_ok
        result["alignment_hint"] = alignment_hint or side_hint
        return result

    def swap_estimator(self, estimator) -> None:
        """
        Phase 2 silent swap: replace heuristic with the ML regressor.

        Args:
            estimator : any object implementing
                        .predict(kp_front, kp_side, height_cm) → dict
        """
        self._estimator = estimator
        logger.info(
            "FitEnginePipeline: estimator swapped to %s",
            type(estimator).__name__,
        )

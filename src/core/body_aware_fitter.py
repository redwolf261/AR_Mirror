#!/usr/bin/env python3
"""
Body-Aware Garment Fitting Module
Uses MediaPipe pose detection to fit garments to actual body size
NO ADDITIONAL ML MODELS NEEDED - Just geometric transformations!
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  # type: ignore
from typing import Optional, Tuple, Dict, Any
import os
import time
import logging
from collections import deque

from src.core.landmark_logger import LandmarkStabilityLogger
from src.core.landmark_smoother import LandmarkSmoother

logger = logging.getLogger(__name__)


class BodyAwareGarmentFitter:
    """
    Fits garments to user's actual body using MediaPipe pose detection
    Uses simple geometry - no additional ML models required!

    Camera calibration notes
    ------------------------
    Raw pixel measurements shrink / grow with subject distance.  We apply a
    perspective-scale normalisation using the known average shoulder-to-hip
    distance (≈55 cm) as a depth anchor so that garment sizing is stable
    over a ±50 % change in subject distance.

    Focal length is estimated once per session from the first stable frame
    (shoulder_width ≥ CALIBRATION_MIN_PX) and cached.  If the focal length
    estimate changes by more than FOCAL_RECAL_THRESHOLD between frames the
    running average is updated slowly (EMA α=0.05) so a single noisy frame
    doesn't corrupt the calibration.
    """

    # Average human shoulder width used as real-world reference (metres)
    _REAL_SHOULDER_WIDTH_M: float = 0.42
    # Minimum pixel shoulder width before we trust a calibration frame
    _CALIBRATION_MIN_PX: int = 60
    # EMA blending factor for focal-length running average (small = slow update)
    _FOCAL_EMA_ALPHA: float = 0.05
    # If estimated focal length deviates by more than this fraction, run EMA
    _FOCAL_RECAL_THRESHOLD: float = 0.15
    # Observation-layer quality gates
    _CRITICAL_VISIBILITY: float = 0.35
    _MIN_BRIGHTNESS: float = 28.0
    _MAX_YAW_DEG: float = 25.0

    def __init__(self, model_path: str = 'pose_landmarker_lite.task', output_segmentation_masks: bool = True):
        """Initialize MediaPipe pose detector"""
        self.model_path = model_path
        self.output_segmentation_masks = bool(output_segmentation_masks)
        self.detector = None
        self._start_time_ns = time.monotonic_ns()  # Real-time base for timestamps
        
        # Detection quality tracking
        self.last_detection_status: str = 'not_started'
        self.last_confidence: float = 0.0
        self.consecutive_failures: int = 0
        self.total_detections: int = 0
        self.successful_detections: int = 0
        
        # Instrumentation: landmark stability tracking
        self.landmark_logger = LandmarkStabilityLogger()
        
        # Phase B: pre-warp landmark smoothing
        self.landmark_smoother = LandmarkSmoother()
        
        # Phase B: static pose lock
        self._static_threshold_px = 2.0   # below this displacement = static
        self._static_lock_frames = 5      # consecutive static frames to lock
        self._consecutive_static = 0
        self._locked_measurements = None   # cached measurements during lock

        # Camera calibration state
        # Estimated focal length (pixels).  None = not yet calibrated.
        self._focal_length_px: Optional[float] = None
        self._user_height_cm: Optional[float] = None
        self._scale_px_to_cm: Optional[float] = None
        self._calibration_square_cm: float = 10.0
        self._calibration_source: str = "uninitialized"

        # Per-metric robust stabilisation (median over short window)
        self._metric_windows = {
            'shoulder_width': deque(maxlen=10),
            'torso_height': deque(maxlen=10),
            'chest_width': deque(maxlen=10),
            'hip_width': deque(maxlen=10),
            'waist_width': deque(maxlen=10),
        }
        self._stable_metrics: Dict[str, float] = {}
        self._metric_var_thresholds = {
            'shoulder_width': 80.0,
            'torso_height': 120.0,
            'chest_width': 90.0,
            'hip_width': 100.0,
            'waist_width': 90.0,
        }

        # Feature-flagged non-rigid warp path (enabled by default).
        env_nonrigid = os.getenv('USE_NONRIGID_WARP', '0').strip().lower()
        self.use_nonrigid_warp = env_nonrigid not in {'0', 'false', 'off', 'no'}
        self._nonrigid_update_interval = max(1, int(os.getenv('NONRIGID_WARP_UPDATE_INTERVAL', '2')))
        self._nonrigid_backoff_interval = max(1, int(os.getenv('NONRIGID_WARP_BACKOFF_INTERVAL', '3')))
        self._nonrigid_budget_ms = float(os.getenv('NONRIGID_FRAME_BUDGET_MS', '33.0'))
        self._nonrigid_margin_ms = float(os.getenv('NONRIGID_FRAME_MARGIN_MS', '5.0'))
        self._runtime_last_frame_ms = 0.0
        self._runtime_frame_ema_ms = 0.0
        self._runtime_frame_ema_alpha = float(np.clip(float(os.getenv('NONRIGID_FRAME_EMA_ALPHA', '0.2')), 0.01, 1.0))
        self._nonrigid_frame_counter = 0
        self._nonrigid_frames_since_refresh = 0
        self._cached_nonrigid_rgb: Optional[np.ndarray] = None
        self._cached_nonrigid_mask: Optional[np.ndarray] = None
        self._nonrigid_refresh_blend_alpha = 0.35
        self._nonrigid_blend_px = 14
        self._nonrigid_warp_scale = float(np.clip(float(os.getenv('NONRIGID_WARP_SCALE', '0.5')), 0.25, 1.0))
        env_nr_profile = os.getenv('NONRIGID_WARP_PROFILE', '0').strip().lower()
        self._nonrigid_profile_enabled = env_nr_profile not in {'0', 'false', 'off', 'no'}
        self._nonrigid_profile_every = max(1, int(os.getenv('NONRIGID_WARP_PROFILE_EVERY', '20')))
        self._nonrigid_profile_counter = 0
        self._nonrigid_post_profile_every = max(1, int(os.getenv('NONRIGID_POST_PROFILE_EVERY', '20')))
        self._nonrigid_post_profile_counter = 0
        self._nonrigid_min_scale = 0.65
        self._nonrigid_max_scale = 1.35
        self._blend_weight_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._last_fit_diag: Dict[str, float] = {
            'roi_area': 0.0,
            'roi_ratio': 0.0,
            'post_ms': 0.0,
            'post_mask_ms': 0.0,
            'post_compose_ms': 0.0,
            'post_cvt_ms': 0.0,
            'post_writeback_ms': 0.0,
        }
        self._scratch_buffers: Dict[Tuple[str, Tuple[int, ...], str], np.ndarray] = {}
        
        # Download model if needed
        if not os.path.exists(model_path):
            print(f"Downloading pose model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, model_path)
            print(f"[OK] Model downloaded")
        
        # Create detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=self.output_segmentation_masks,
            running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        print("[OK] Body-aware fitter initialized")

    def set_user_height_cm(self, height_cm: Optional[float]) -> None:
        """Set user height for scale calibration (when available from UI/workflow)."""
        if height_cm is None:
            self._user_height_cm = None
            return
        try:
            h = float(height_cm)
            if 130.0 <= h <= 240.0:
                self._user_height_cm = h
        except (TypeError, ValueError):
            return

    def set_runtime_frame_time_ms(self, frame_time_ms: Optional[float]) -> None:
        """Provide last frame time to adaptive non-rigid scheduler."""
        if frame_time_ms is None:
            return
        try:
            v = float(frame_time_ms)
            if np.isfinite(v) and v >= 0.0:
                self._runtime_last_frame_ms = v
                if self._runtime_frame_ema_ms <= 0.0:
                    self._runtime_frame_ema_ms = v
                else:
                    a = self._runtime_frame_ema_alpha
                    self._runtime_frame_ema_ms = (a * v) + ((1.0 - a) * self._runtime_frame_ema_ms)
        except (TypeError, ValueError):
            return

    def set_calibration_square_cm(self, square_cm: float) -> None:
        try:
            v = float(square_cm)
            if 2.0 <= v <= 50.0:
                self._calibration_square_cm = v
        except (TypeError, ValueError):
            return

    def set_scale_px_to_cm(self, scale_px_to_cm: Optional[float], source: str = "manual") -> None:
        if scale_px_to_cm is None:
            self._scale_px_to_cm = None
            self._calibration_source = "unset"
            return
        try:
            s = float(scale_px_to_cm)
            if 0.5 <= s <= 100.0:
                self._scale_px_to_cm = s
                self._calibration_source = source
        except (TypeError, ValueError):
            return

    def _try_red_square_calibration(self, frame_bgr: np.ndarray) -> None:
        """Estimate px/cm from a red square marker with known side length."""
        if self._scale_px_to_cm is not None:
            return
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Red wraps HSV hue space
        mask1 = cv2.inRange(hsv, np.array([0, 90, 70], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array([168, 90, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 500:
            return
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            return
        x, y, rw, rh = cv2.boundingRect(approx)
        ar = rw / max(rh, 1)
        if not (0.80 <= ar <= 1.20):
            return
        side_px = float((rw + rh) * 0.5)
        if side_px < 10:
            return
        self._scale_px_to_cm = side_px / max(self._calibration_square_cm, 1e-6)
        self._calibration_source = "red_square"

    @staticmethod
    def _landmark_px(landmark, frame_w: int, frame_h: int) -> np.ndarray:
        return np.array([landmark.x * frame_w, landmark.y * frame_h], dtype=np.float32)

    @staticmethod
    def _safe_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _clip_conf(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _get_scratch(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        key = (name, tuple(shape), np.dtype(dtype).str)
        buf = self._scratch_buffers.get(key)
        if buf is None or buf.shape != tuple(shape) or buf.dtype != np.dtype(dtype):
            buf = np.empty(shape, dtype=dtype)
            self._scratch_buffers[key] = buf
        return buf

    @staticmethod
    def _as_2d_mask(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            return mask[:, :, 0]
        return mask

    def _robust_metric(self, key: str, value: float, confidence: float, min_conf: float = 0.45) -> float:
        """Keep a median-stable value per metric; freeze on low confidence frames."""
        window = self._metric_windows[key]
        if confidence >= min_conf and np.isfinite(value) and value > 0:
            window.append(float(value))
            if len(window) >= 5:
                var = float(np.var(np.array(window, dtype=np.float32)))
                if var > self._metric_var_thresholds.get(key, 100.0):
                    # High temporal variance -> hold last stable value
                    return self._stable_metrics.get(key, float(np.median(window)))
            self._stable_metrics[key] = float(np.median(window))
        if key in self._stable_metrics:
            return self._stable_metrics[key]
        return float(value)

    def _sample_mask_band_width(self, body_mask: np.ndarray, y_px: int) -> Tuple[float, float]:
        """Sample silhouette width around a row band; returns (width_px, confidence)."""
        if body_mask is None or body_mask.size == 0:
            return 0.0, 0.0
        h, _ = body_mask.shape[:2]
        y0 = max(0, y_px - 2)
        y1 = min(h, y_px + 3)
        widths = []
        for y in range(y0, y1):
            row = body_mask[y]
            xs = np.where(row > 0)[0]
            if xs.size > 1:
                widths.append(float(xs[-1] - xs[0]))
        if not widths:
            return 0.0, 0.0
        width = float(np.median(widths))
        coverage = min(1.0, len(widths) / 5.0)
        return width, coverage
    
    def extract_body_measurements(self, frame: np.ndarray) -> Optional[dict]:
        """
        Extract body measurements from frame
        Returns dict with shoulder_width, torso_height, torso_box, body_mask
        """
        h, w = frame.shape[:2]

        # Observation gate 1: lighting quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_light = float(np.mean(gray)) < self._MIN_BRIGHTNESS

        # Opportunistic physical calibration (runs until a valid red square is seen)
        self._try_red_square_calibration(frame)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        self.total_detections += 1
        t_detect = time.perf_counter()
        detection_result = self.detector.detect(mp_image)
        detect_ms = (time.perf_counter() - t_detect) * 1000
        if detect_ms > 10:  # Only log if significant
            logger.debug(f"[Pose] MediaPipe detection: {detect_ms:.1f}ms")
        
        if not detection_result.pose_landmarks:
            self.consecutive_failures += 1
            self.last_detection_status = 'no_person'
            self.last_confidence = 0.0
            # Reset smoother state when person disappears
            if self.consecutive_failures > 10:
                self.landmark_smoother.reset()
                self._consecutive_static = 0
                self._locked_measurements = None
            logger.debug(f"Pose detection failed (consecutive: {self.consecutive_failures})")
            return None
        
        self.consecutive_failures = 0
        self.successful_detections += 1
        
        # Get landmarks
        landmarks = detection_result.pose_landmarks[0]
        
        # Instrumentation: log RAW landmark positions BEFORE smoothing
        self.landmark_logger.log_frame(landmarks, frame_shape=(h, w))
        
        # Phase B1: Apply per-joint weighted EMA smoothing
        landmarks = self.landmark_smoother.smooth(landmarks, frame_shape=(h, w))
        
        # Key points (MediaPipe landmark indices)
        left_shoulder = landmarks[11]   # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        left_hip = landmarks[23]        # Left hip
        right_hip = landmarks[24]       # Right hip
        
        # Track landmark confidence (critical set)
        nose = landmarks[0] if len(landmarks) > 0 else None
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None

        # Upper-body try-on should remain usable when lower-body landmarks are
        # cropped. Hard gate only on shoulder visibility.
        critical_vis = [left_shoulder.visibility, right_shoulder.visibility]

        avg_confidence = float(np.mean(np.array(critical_vis, dtype=np.float32)))
        self.last_confidence = avg_confidence

        if min(critical_vis) < self._CRITICAL_VISIBILITY:
            self.last_detection_status = 'low_confidence'
            logger.debug(f"Low landmark confidence: {avg_confidence:.2f}")
            return None
        
        self.last_detection_status = 'detected_low_light' if low_light else 'detected'
        
        # Geometric points in pixel space
        ls = self._landmark_px(left_shoulder, w, h)
        rs = self._landmark_px(right_shoulder, w, h)

        shoulder_width_raw = abs(right_shoulder.x - left_shoulder.x) * w

        # If hips are unreliable, infer an upper-body hip band from shoulder
        # geometry so garment placement still tracks movement.
        hips_reliable = min(left_hip.visibility, right_hip.visibility) >= 0.35
        if hips_reliable:
            lh = self._landmark_px(left_hip, w, h)
            rh = self._landmark_px(right_hip, w, h)
            hip_width_raw = abs(right_hip.x - left_hip.x) * w
            hip_vis_proxy = min(left_hip.visibility, right_hip.visibility)
        else:
            shoulder_mid_y = float((ls[1] + rs[1]) * 0.5)
            inferred_drop = max(shoulder_width_raw * 1.05, h * 0.20)
            inferred_y = float(min(h - 2, shoulder_mid_y + inferred_drop))
            lh = np.array([ls[0], inferred_y], dtype=np.float32)
            rh = np.array([rs[0], inferred_y], dtype=np.float32)
            hip_width_raw = max(shoulder_width_raw * 0.82, 1.0)
            hip_vis_proxy = min(left_shoulder.visibility, right_shoulder.visibility) * 0.65

        shoulder_mid = (ls + rs) * 0.5
        hip_mid = (lh + rh) * 0.5

        # Rotation compensation from pseudo-3D shoulder yaw
        yaw_rad = float(np.arctan2((right_shoulder.z - left_shoulder.z), (right_shoulder.x - left_shoulder.x + 1e-6)))
        yaw_deg = float(np.degrees(yaw_rad))
        cos_yaw = max(0.55, float(np.cos(np.clip(abs(yaw_rad), 0.0, np.deg2rad(65.0)))))

        # Raw measurements in pixels (frontal-corrected where relevant)
        shoulder_width = shoulder_width_raw / cos_yaw
        torso_height_raw = self._safe_distance(shoulder_mid, hip_mid)
        torso_height = torso_height_raw
        hip_width = hip_width_raw / cos_yaw

        # --- Depth normalisation DISABLED ---
        # Normalising to a "canonical" shoulder width caused the garment to
        # appear undersized when the person stands close to the camera and
        # oversized when they stand far away.  The garment should always be
        # sized to the ACTUAL detected body size in the current frame.
        # shoulder_width, torso_height = self._apply_depth_normalisation(
        #     shoulder_width, torso_height, w, h
        # )

        # Calculate torso bounding box
        torso_x1 = int(max(0, min(ls[0], rs[0], lh[0], rh[0])))
        torso_y1 = int(max(0, min(left_shoulder.y, right_shoulder.y) * h))
        torso_x2 = int(min(w - 1, max(ls[0], rs[0], lh[0], rh[0])))
        torso_y2 = int(min(h - 1, max(lh[1], rh[1])))
        
        # Get body segmentation mask (conditioned silhouette)
        body_mask = None
        if detection_result.segmentation_masks:
            mask = detection_result.segmentation_masks[0].numpy_view()
            body_mask = (mask > 0.5).astype(np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, k, iterations=1)
            body_mask = cv2.GaussianBlur((body_mask * 255).astype(np.uint8), (7, 7), 1.5)
            body_mask = (body_mask > 120).astype(np.uint8)
        
        # Measurement confidence per metric
        torso_sym = self._safe_distance(ls, lh) / max(self._safe_distance(rs, rh), 1e-6)
        torso_sym_score = self._clip_conf(1.0 - min(abs(1.0 - torso_sym), 1.0))
        frontality_score = self._clip_conf((cos_yaw - 0.55) / 0.45)

        shoulder_conf = self._clip_conf(min(left_shoulder.visibility, right_shoulder.visibility) * (0.55 + 0.45 * frontality_score))
        torso_conf = self._clip_conf(min(left_shoulder.visibility, right_shoulder.visibility, left_hip.visibility, right_hip.visibility) * torso_sym_score)
        hip_conf = self._clip_conf(hip_vis_proxy * (0.55 + 0.45 * frontality_score))

        # Mask-based chest / waist estimation
        chest_y = int(np.clip((shoulder_mid[1] + torso_height * 0.20), 0, h - 1))
        waist_y = int(np.clip((hip_mid[1] - torso_height * 0.20), 0, h - 1))
        chest_width_mask, chest_mask_conf = self._sample_mask_band_width(body_mask, chest_y) if body_mask is not None else (0.0, 0.0)
        waist_width_mask, waist_mask_conf = self._sample_mask_band_width(body_mask, waist_y) if body_mask is not None else (0.0, 0.0)

        # Multi-signal fusion (landmark trusted > silhouette)
        chest_landmark = shoulder_width * 1.10
        waist_landmark = hip_width * 0.88
        looseness = (waist_width_mask / max(waist_landmark, 1e-6)) if waist_width_mask > 0 else 1.0
        clothing_correction = 0.90 if looseness <= 1.15 else 0.85
        silhouette_w_chest = chest_width_mask * clothing_correction if chest_width_mask > 0 else chest_landmark
        silhouette_w_waist = waist_width_mask * clothing_correction if waist_width_mask > 0 else waist_landmark
        chest_width_raw = 0.70 * chest_landmark + 0.30 * silhouette_w_chest
        waist_width_raw = 0.75 * waist_landmark + 0.25 * silhouette_w_waist

        # Clothing inflation compensation for silhouette-derived waist
        waist_width_raw *= clothing_correction

        chest_conf = self._clip_conf(min(torso_conf, chest_mask_conf if chest_width_mask > 0 else torso_conf * 0.7))
        waist_conf = self._clip_conf(min(torso_conf, waist_mask_conf if waist_width_mask > 0 else torso_conf * 0.6) * (0.55 + 0.45 * frontality_score))

        # Robust temporal stabilisation per metric
        shoulder_width = self._robust_metric('shoulder_width', shoulder_width, shoulder_conf)
        torso_height = self._robust_metric('torso_height', torso_height, torso_conf)
        chest_width = self._robust_metric('chest_width', chest_width_raw / cos_yaw, chest_conf)
        hip_width = self._robust_metric('hip_width', hip_width, hip_conf)
        waist_width = self._robust_metric('waist_width', waist_width_raw / cos_yaw, waist_conf)

        # Scale calibration priority:
        # 1) Red square physical calibration
        # 2) User height normalization
        # 3) Shoulder prior fallback
        px_per_cm: Optional[float] = None
        est_height_cm: Optional[float] = None

        if self._scale_px_to_cm is not None:
            px_per_cm = float(self._scale_px_to_cm)
            self._calibration_source = self._calibration_source or "red_square"

        if px_per_cm is None and nose and left_ankle and right_ankle and self._user_height_cm is not None:
            if min(nose.visibility, left_ankle.visibility, right_ankle.visibility) > self._CRITICAL_VISIBILITY:
                nose_px = self._landmark_px(nose, w, h)
                ankle_mid_px = (self._landmark_px(left_ankle, w, h) + self._landmark_px(right_ankle, w, h)) * 0.5
                full_body_px = self._safe_distance(nose_px, ankle_mid_px)
                if full_body_px > 50:
                    px_per_cm = full_body_px / self._user_height_cm
                    est_height_cm = float(self._user_height_cm)
                    self._calibration_source = "user_height"

        if px_per_cm is None:
            # Shoulder prior (approximate fallback, explicitly lower confidence)
            px_per_cm = shoulder_width / 42.0
            est_height_cm = None
            self._calibration_source = "shoulder_prior"

        def _to_cm(px: float) -> float:
            return float(px / max(px_per_cm or 1e-6, 1e-6))

        shoulder_width_cm = _to_cm(shoulder_width)
        chest_cm = _to_cm(chest_width)
        torso_length_cm = _to_cm(torso_height)
        hip_cm = _to_cm(hip_width)
        waist_cm = _to_cm(waist_width)

        # Hidden-depth estimate and ellipse circumference model
        depth_factor = 0.72 if looseness <= 1.15 else 0.66
        chest_depth_cm = chest_cm * depth_factor
        waist_depth_cm = waist_cm * depth_factor

        def _ellipse_circumference(width_cm: float, depth_cm: float) -> float:
            a = max(width_cm * 0.5, 0.1)
            b = max(depth_cm * 0.5, 0.1)
            return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))

        chest_circumference_cm = _ellipse_circumference(chest_cm, chest_depth_cm)
        waist_circumference_cm = _ellipse_circumference(waist_cm, waist_depth_cm)

        metric_confidence = {
            'shoulder': shoulder_conf,
            'chest': chest_conf,
            'torso': torso_conf,
            'hip': hip_conf,
            'waist': waist_conf,
            'frontality': frontality_score,
        }

        pose_alignment_ok = abs(yaw_deg) <= self._MAX_YAW_DEG
        if not pose_alignment_ok:
            # Reduce confidence when side rotation is too high
            for k in ('shoulder', 'chest', 'waist', 'hip'):
                metric_confidence[k] *= 0.65

        if low_light:
            for k in ('shoulder', 'chest', 'torso', 'hip', 'waist'):
                metric_confidence[k] *= 0.85

        # Import size recommendation system
        try:
            from src.core.size_recommendation import get_size_recommendation, format_size_recommendation

            measurements = {
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'chest_width': chest_width,
                'waist_width': waist_width,
                'hip_width': hip_width,
                'torso_box': (torso_x1, torso_y1, torso_x2, torso_y2),
                'body_mask': body_mask,
                'landmarks': landmarks,
                'confidence': float(np.mean(list(metric_confidence.values())[:5])),
                'measurement_confidence': metric_confidence,
                'yaw_deg': yaw_deg,
                'user_height_cm': self._user_height_cm,
                'height_cm': est_height_cm,
                'shoulder_width_cm': shoulder_width_cm,
                'chest_cm': chest_cm,
                'torso_length_cm': torso_length_cm,
                'hip_cm': hip_cm,
                'waist_cm': waist_cm,
                'chest_circumference_cm': chest_circumference_cm,
                'waist_circumference_cm': waist_circumference_cm,
                'depth_factor': depth_factor,
                'looseness_ratio': float(looseness),
                'pose_alignment_ok': pose_alignment_ok,
                'alignment_warning': None if pose_alignment_ok else 'Please face front (yaw > 25 deg)',
                'cm_scale_source': self._calibration_source,
            }

            # Get size recommendation
            size_rec = get_size_recommendation(measurements)

            measurements.update({
                'size_recommendation': size_rec['recommended_size'],
                'size_confidence': size_rec['confidence'],
                'size_alternatives': size_rec['all_sizes'],
                'fit_classification': size_rec.get('fit_classification'),
                'measurements_cm': size_rec['measurements_cm'],
                'size_description': format_size_recommendation(size_rec)
            })

            # Add individual cm measurements for compatibility
            measurements.update({
                'shoulder_width_cm': size_rec['measurements_cm']['shoulder_width'],
                'chest_cm': size_rec['measurements_cm']['chest_width'],
                'torso_length_cm': size_rec['measurements_cm']['torso_length'],
            })

        except Exception as e:
            # Fallback if size recommendation fails
            logger.debug(f"Size recommendation failed: {e}")
            measurements = {
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'chest_width': chest_width,
                'waist_width': waist_width,
                'hip_width': hip_width,
                'torso_box': (torso_x1, torso_y1, torso_x2, torso_y2),
                'body_mask': body_mask,
                'landmarks': landmarks,
                'confidence': float(np.mean(list(metric_confidence.values())[:5])),
                'measurement_confidence': metric_confidence,
                'yaw_deg': yaw_deg,
                'user_height_cm': self._user_height_cm,
                'height_cm': est_height_cm,
                'shoulder_width_cm': shoulder_width_cm,
                'chest_cm': chest_cm,
                'torso_length_cm': torso_length_cm,
                'hip_cm': hip_cm,
                'waist_cm': waist_cm,
                'chest_circumference_cm': chest_circumference_cm,
                'waist_circumference_cm': waist_circumference_cm,
                'depth_factor': depth_factor,
                'looseness_ratio': float(looseness),
                'pose_alignment_ok': pose_alignment_ok,
                'alignment_warning': None if pose_alignment_ok else 'Please face front (yaw > 25 deg)',
                'cm_scale_source': self._calibration_source,
                'size_recommendation': 'M',  # Default fallback
                'size_confidence': 0.5,
                'fit_classification': 'Perfect',
                'size_description': 'Size estimation unavailable'
            }
        
        # Phase B3: Static pose lock
        last_disp = self.landmark_logger.get_last_displacement()
        if last_disp < self._static_threshold_px:
            self._consecutive_static += 1
        else:
            self._consecutive_static = 0
        
        if self._consecutive_static >= self._static_lock_frames:
            # Locked: return cached measurements (skip re-warp)
            if self._locked_measurements is not None:
                # Update body_mask from current frame (it changes with lighting)
                self._locked_measurements['body_mask'] = body_mask

                # IMPORTANT: Also update size recommendation in cached measurements
                # In case the user's distance/pose has changed slightly
                try:
                    if ('size_recommendation' not in self._locked_measurements or
                        'size_confidence' not in self._locked_measurements):
                        # Recalculate size recommendation for cached measurements
                        from src.core.size_recommendation import get_size_recommendation, format_size_recommendation
                        size_rec = get_size_recommendation(self._locked_measurements)
                        self._locked_measurements.update({
                            'size_recommendation': size_rec['recommended_size'],
                            'size_confidence': size_rec['confidence'],
                            'size_alternatives': size_rec['all_sizes'],
                            'measurements_cm': size_rec['measurements_cm'],
                            'size_description': format_size_recommendation(size_rec),
                            'shoulder_width_cm': size_rec['measurements_cm']['shoulder_width'],
                            'chest_cm': size_rec['measurements_cm']['chest_width'],
                            'torso_length_cm': size_rec['measurements_cm']['torso_length']
                        })
                except Exception as e:
                    logger.debug(f"Size recommendation update failed in cache: {e}")
                    # Ensure fallback values exist
                    if 'size_recommendation' not in self._locked_measurements:
                        self._locked_measurements.update({
                            'size_recommendation': 'M',
                            'size_confidence': 0.5,
                            'size_description': 'Size estimation unavailable'
                        })

                return self._locked_measurements
            else:
                # First lock — cache current measurements (with size data)
                self._locked_measurements = measurements
        else:
            # Moving — update cache (with size data)
            self._locked_measurements = measurements
        
        return measurements

    # ------------------------------------------------------------------
    # Camera calibration helpers
    # ------------------------------------------------------------------

    def _apply_depth_normalisation(
        self,
        shoulder_width_px: float,
        torso_height_px: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float]:
        """
        Normalise pixel measurements to a canonical subject distance so that
        garment scaling is stable when the user moves closer or farther from
        the camera.

        Strategy
        --------
        We treat the shoulder width as a known real-world reference
        (self._REAL_SHOULDER_WIDTH_M = 0.42 m) and estimate the camera focal
        length f (in pixels) as:

            f = shoulder_px * Z / shoulder_real

        where Z is the unknown subject distance.  Instead of solving for Z
        we pick a *canonical* shoulder width (40 % of frame width) and scale
        all measurements so they appear at that canonical size.  The focal
        length estimate is used only to detect drift; a running EMA keeps it
        stable.

        For frames where the shoulder is too narrow (cropped / too far) we
        skip normalisation and return raw measurements.

        Returns
        -------
        (normalised_shoulder_px, normalised_torso_px)
        """
        if shoulder_width_px < self._CALIBRATION_MIN_PX:
            # Subject too far or partially out of frame — skip correction
            return shoulder_width_px, torso_height_px

        # Canonical shoulder width: 40 % of frame width
        canonical_shoulder_px = frame_w * 0.40

        # Scale factor to bring current measurements to canonical distance
        scale = canonical_shoulder_px / shoulder_width_px

        # Estimate focal length for calibration tracking (f proportional to
        # shoulder_px; using frame width as a proxy for sensor size gives a
        # unitless relative focal length)
        focal_estimate = shoulder_width_px / self._REAL_SHOULDER_WIDTH_M  # px / m

        if self._focal_length_px is None:
            # First frame — seed the estimate
            self._focal_length_px = focal_estimate
            logger.info(f"Camera calibration seeded: focal_est={focal_estimate:.1f} px/m")
        else:
            deviation = abs(focal_estimate - self._focal_length_px) / self._focal_length_px
            if deviation > self._FOCAL_RECAL_THRESHOLD:
                # Gradual EMA update to reject single noisy frames
                self._focal_length_px = (
                    self._FOCAL_EMA_ALPHA * focal_estimate
                    + (1.0 - self._FOCAL_EMA_ALPHA) * self._focal_length_px
                )
                logger.debug(
                    f"Focal length drifted {deviation*100:.1f}%, "
                    f"EMA update → {self._focal_length_px:.1f} px/m"
                )

        # Apply scale normalisation while keeping measurements within the frame
        norm_shoulder = min(shoulder_width_px * scale, frame_w * 0.95)
        norm_torso = min(torso_height_px * scale, frame_h * 0.90)

        return float(norm_shoulder), float(norm_torso)

    def _get_blend_weights(self, height: int, blend_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create cached vertical blend weights for upper/mid/lower regions."""
        key = (int(height), int(blend_px))
        cached = self._blend_weight_cache.get(key)
        if cached is not None:
            return cached

        h = max(2, int(height))
        b = max(1, int(blend_px))
        y = np.arange(h, dtype=np.float32)
        y1 = float(h) / 3.0
        y2 = float(2.0 * h) / 3.0

        def _smoothstep(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        t1 = _smoothstep((y - (y1 - b)) / max(1.0, 2.0 * b))
        t2 = _smoothstep((y - (y2 - b)) / max(1.0, 2.0 * b))

        w_upper = (1.0 - t1)
        w_mid = t1 * (1.0 - t2)
        w_lower = t2

        # Normalize row-wise to keep total energy stable.
        s = np.maximum(1e-6, w_upper + w_mid + w_lower)
        w_upper = (w_upper / s).reshape(h, 1, 1).astype(np.float32)
        w_mid = (w_mid / s).reshape(h, 1, 1).astype(np.float32)
        w_lower = (w_lower / s).reshape(h, 1, 1).astype(np.float32)

        self._blend_weight_cache[key] = (w_upper, w_mid, w_lower)
        return self._blend_weight_cache[key]

    def _fit_piecewise_affine(
        self,
        garment_rgb: np.ndarray,
        garment_mask: np.ndarray,
        landmarks,
        garment_x1: int,
        garment_y1: int,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Apply 3-region piecewise affine warp driven by shoulders/hips."""
        full_h, full_w = garment_rgb.shape[:2]
        if full_h < 2 or full_w < 2:
            return garment_rgb, garment_mask, {'down_ms': 0.0, 'warp_ms': 0.0, 'up_ms': 0.0}

        t_down_ms = 0.0
        t_warp_ms = 0.0
        t_up_ms = 0.0

        warp_scale = float(np.clip(getattr(self, '_nonrigid_warp_scale', 1.0), 0.25, 1.0))
        if warp_scale < 0.999:
            t_down = time.perf_counter()
            low_w = max(2, int(round(full_w * warp_scale)))
            low_h = max(2, int(round(full_h * warp_scale)))
            work_rgb = self._get_scratch("nr_down_rgb", (low_h, low_w, garment_rgb.shape[2]), garment_rgb.dtype)
            cv2.resize(garment_rgb, (low_w, low_h), dst=work_rgb, interpolation=cv2.INTER_AREA)
            mask_2d = self._as_2d_mask(garment_mask)
            work_mask = self._get_scratch("nr_down_mask", (low_h, low_w), mask_2d.dtype)
            cv2.resize(mask_2d, (low_w, low_h), dst=work_mask, interpolation=cv2.INTER_AREA)
            t_down_ms = (time.perf_counter() - t_down) * 1000.0
            sx = float(low_w) / float(max(1, full_w))
            sy = float(low_h) / float(max(1, full_h))
        else:
            low_w, low_h = full_w, full_h
            work_rgb = garment_rgb
            work_mask = self._as_2d_mask(garment_mask)
            sx, sy = 1.0, 1.0

        gh, gw = work_rgb.shape[:2]

        ls = np.array([
            float((landmarks[11].x * frame_w - garment_x1) * sx),
            float((landmarks[11].y * frame_h - garment_y1) * sy)
        ], dtype=np.float32)
        rs = np.array([
            float((landmarks[12].x * frame_w - garment_x1) * sx),
            float((landmarks[12].y * frame_h - garment_y1) * sy)
        ], dtype=np.float32)
        lh = np.array([
            float((landmarks[23].x * frame_w - garment_x1) * sx),
            float((landmarks[23].y * frame_h - garment_y1) * sy)
        ], dtype=np.float32)
        rh = np.array([
            float((landmarks[24].x * frame_w - garment_x1) * sx),
            float((landmarks[24].y * frame_h - garment_y1) * sy)
        ], dtype=np.float32)

        shoulder_y = float(np.clip((ls[1] + rs[1]) * 0.5, 0.0, gh - 1.0))
        hip_y = float(np.clip((lh[1] + rh[1]) * 0.5, shoulder_y + 1.0, gh - 1.0))
        torso_span = max(8.0, hip_y - shoulder_y)

        # Region boundaries in output garment space.
        yb = np.array([0.0, gh / 3.0, 2.0 * gh / 3.0, gh - 1.0], dtype=np.float32)

        shoulder_width = float(max(2.0, np.linalg.norm(rs - ls)))
        hip_width = float(max(2.0, np.linalg.norm(rh - lh)))

        def _edge_x(yv: float, left: bool) -> float:
            t = (yv - shoulder_y) / max(1e-6, torso_span)
            t = float(np.clip(t, 0.0, 1.0))
            if left:
                x = ls[0] + t * (lh[0] - ls[0])
                width = shoulder_width + t * (hip_width - shoulder_width)
                return float(x - 0.04 * width)
            x = rs[0] + t * (rh[0] - rs[0])
            width = shoulder_width + t * (hip_width - shoulder_width)
            return float(x + 0.04 * width)

        xl = np.array([_edge_x(float(v), True) for v in yb], dtype=np.float32)
        xr = np.array([_edge_x(float(v), False) for v in yb], dtype=np.float32)

        # Clamp per-region width scaling to avoid overstretch artifacts.
        src_width = float(max(2, gw - 1))
        for i in range(3):
            top_w = max(2.0, xr[i] - xl[i])
            bot_w = max(2.0, xr[i + 1] - xl[i + 1])
            avg_w = 0.5 * (top_w + bot_w)
            scale = float(np.clip(avg_w / src_width, self._nonrigid_min_scale, self._nonrigid_max_scale))
            target_w = scale * src_width
            c_top = 0.5 * (xl[i] + xr[i])
            c_bot = 0.5 * (xl[i + 1] + xr[i + 1])
            xl[i] = c_top - 0.5 * target_w
            xr[i] = c_top + 0.5 * target_w
            xl[i + 1] = c_bot - 0.5 * target_w
            xr[i + 1] = c_bot + 0.5 * target_w

        def _warp_region(src_img: np.ndarray, region_idx: int, is_mask: bool = False) -> np.ndarray:
            y0 = float(yb[region_idx])
            y1 = float(yb[region_idx + 1])
            src_tri = np.array(
                [[0.0, y0], [gw - 1.0, y0], [0.0, y1]], dtype=np.float32
            )
            dst_tri = np.array(
                [[xl[region_idx], y0], [xr[region_idx], y0], [xl[region_idx + 1], y1]], dtype=np.float32
            )
            m = cv2.getAffineTransform(src_tri, dst_tri)
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
            border = 0 if is_mask else 0
            out_shape = (gh, gw) if is_mask else (gh, gw, work_rgb.shape[2])
            dst = self._get_scratch(
                f"nr_region_{'mask' if is_mask else 'rgb'}_{region_idx}",
                out_shape,
                src_img.dtype,
            )
            cv2.warpAffine(
                src_img,
                m,
                (gw, gh),
                dst=dst,
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border,
            )
            return dst

        t_warp = time.perf_counter()
        rg0 = _warp_region(work_rgb, 0, is_mask=False)
        rg1 = _warp_region(work_rgb, 1, is_mask=False)
        rg2 = _warp_region(work_rgb, 2, is_mask=False)

        rm0 = _warp_region(work_mask, 0, is_mask=True)
        rm1 = _warp_region(work_mask, 1, is_mask=True)
        rm2 = _warp_region(work_mask, 2, is_mask=True)

        if rg0.ndim == 2:
            rg0 = rg0[:, :, np.newaxis]
        if rg1.ndim == 2:
            rg1 = rg1[:, :, np.newaxis]
        if rg2.ndim == 2:
            rg2 = rg2[:, :, np.newaxis]

        warped_rgb = self._get_scratch("nr_warp_rgb", rg0.shape, rg0.dtype)
        warped_mask = self._get_scratch("nr_warp_mask", rm0.shape, rm0.dtype)
        np.copyto(warped_rgb, rg0)
        np.copyto(warped_mask, rm0)

        b1 = int(gh / 3)
        b2 = int(2 * gh / 3)
        blend = int(max(4, min(self._nonrigid_blend_px, gh // 8)))

        m0_s = max(0, b1 + blend)
        m0_e = min(gh, b2 - blend)
        if m0_e > m0_s:
            warped_rgb[m0_s:m0_e] = rg1[m0_s:m0_e]
            warped_mask[m0_s:m0_e] = rm1[m0_s:m0_e]

        l_s = min(gh, max(0, b2 + blend))
        if l_s < gh:
            warped_rgb[l_s:] = rg2[l_s:]
            warped_mask[l_s:] = rm2[l_s:]

        s1 = max(0, b1 - blend)
        e1 = min(gh, b1 + blend)
        if e1 > s1:
            t = np.linspace(0.0, 1.0, e1 - s1, dtype=np.float32).reshape(-1, 1, 1)
            warped_rgb[s1:e1] = rg0[s1:e1] * (1.0 - t) + rg1[s1:e1] * t
            warped_mask[s1:e1] = rm0[s1:e1] * (1.0 - t) + rm1[s1:e1] * t

        s2 = max(0, b2 - blend)
        e2 = min(gh, b2 + blend)
        if e2 > s2:
            t = np.linspace(0.0, 1.0, e2 - s2, dtype=np.float32).reshape(-1, 1, 1)
            warped_rgb[s2:e2] = rg1[s2:e2] * (1.0 - t) + rg2[s2:e2] * t
            warped_mask[s2:e2] = rm1[s2:e2] * (1.0 - t) + rm2[s2:e2] * t
        t_warp_ms = (time.perf_counter() - t_warp) * 1000.0

        warped_rgb = warped_rgb.astype(np.float32)
        warped_mask = np.clip(warped_mask.astype(np.float32), 0.0, 1.0)

        if warp_scale < 0.999:
            t_up = time.perf_counter()
            up_rgb = self._get_scratch("nr_up_rgb", (full_h, full_w, warped_rgb.shape[2]), warped_rgb.dtype)
            cv2.resize(warped_rgb, (full_w, full_h), dst=up_rgb, interpolation=cv2.INTER_LINEAR)
            up_mask = self._get_scratch("nr_up_mask", (full_h, full_w), warped_mask.dtype)
            cv2.resize(warped_mask, (full_w, full_h), dst=up_mask, interpolation=cv2.INTER_LINEAR)
            warped_rgb = up_rgb
            warped_mask = up_mask
            t_up_ms = (time.perf_counter() - t_up) * 1000.0

        return (
            warped_rgb.astype(np.float32, copy=False),
            np.clip(warped_mask.astype(np.float32, copy=False), 0.0, 1.0),
            {'down_ms': float(t_down_ms), 'warp_ms': float(t_warp_ms), 'up_ms': float(t_up_ms)},
        )

    def fit_garment_to_body(
        self,
        frame: np.ndarray,
        garment_rgb: np.ndarray,
        garment_mask: np.ndarray,
        body_measurements: dict
    ) -> np.ndarray:
        """
        Fit garment to body using geometric transformations
        NO ML MODEL NEEDED - just scaling and positioning!
        
        Args:
            frame: Camera frame (BGR)
            garment_rgb: Garment image (RGB, float32, 0-1)
            garment_mask: Garment mask (float32, 0-1)
            body_measurements: Dict from extract_body_measurements()
            
        Returns:
            Frame with fitted garment (BGR)
        """
        h, w = frame.shape[:2]
        
        # Extract measurements
        shoulder_width = body_measurements['shoulder_width']
        torso_height = body_measurements['torso_height']
        torso_x1, torso_y1, torso_x2, torso_y2 = body_measurements['torso_box']
        body_mask = body_measurements['body_mask']
        landmarks = body_measurements.get('landmarks')
        
        # Calculate garment scaling
        # Keep the garment slightly narrower than the raw shoulder span so it
        # doesn't read as oversized, but extend the height enough to cover the
        # lower torso and reduce the visible tee underneath.
        target_width = int(max(shoulder_width * 1.02, (torso_x2 - torso_x1) * 0.92))
        target_height = int(max(torso_height * 1.28, target_width * 1.18))
        target_width = max(2, target_width)
        target_height = max(2, target_height)

        shoulder_mid_x = (torso_x1 + torso_x2) // 2
        shoulder_top_y = torso_y1
        if landmarks is not None and len(landmarks) > 24:
            ls = landmarks[11]
            rs = landmarks[12]
            lh = landmarks[23]
            rh = landmarks[24]
            shoulder_mid_x = int(((ls.x + rs.x) * 0.5) * w)
            shoulder_top_y = int(min(ls.y, rs.y) * h)
            # Use the shoulder/hip span to place the shirt collar lower than
            # the raw torso box top.  This keeps the garment out of the face.
            torso_span = int(max(abs(lh.x - ls.x), abs(rh.x - rs.x)) * w)
            target_width = max(target_width, int(torso_span * 0.94))
            target_height = max(target_height, int(torso_height * 1.22))

        collar_drop = max(8, int(torso_height * 0.08))
        
        # Resize garment to match body size
        garment_resized = self._get_scratch("fit_garment_rgb", (target_height, target_width, garment_rgb.shape[2]), garment_rgb.dtype)
        cv2.resize(garment_rgb, (target_width, target_height), dst=garment_resized, interpolation=cv2.INTER_LINEAR)
        mask_2d = self._as_2d_mask(garment_mask)
        mask_resized = self._get_scratch("fit_garment_mask", (target_height, target_width), mask_2d.dtype)
        cv2.resize(mask_2d, (target_width, target_height), dst=mask_resized, interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask has correct shape
        # Calculate position (center on the upper torso, but keep the collar
        # below the face and slightly lower than the raw shoulder line).
        garment_x1 = shoulder_mid_x - target_width // 2
        garment_y1 = shoulder_top_y + collar_drop
        garment_x2 = garment_x1 + target_width
        garment_y2 = garment_y1 + target_height
        
        # Clip to frame boundaries
        garment_x1 = max(0, garment_x1)
        garment_y1 = max(0, garment_y1)
        garment_x2 = min(w, garment_x2)
        garment_y2 = min(h, garment_y2)
        
        # Adjust garment size if clipped
        actual_width = garment_x2 - garment_x1
        actual_height = garment_y2 - garment_y1
        
        # Guard: skip if region is too small or empty
        if actual_width < 2 or actual_height < 2:
            self._last_fit_diag = {
                'roi_area': 0.0,
                'roi_ratio': 0.0,
                'post_ms': 0.0,
                'post_mask_ms': 0.0,
                'post_compose_ms': 0.0,
                'post_cvt_ms': 0.0,
                'post_writeback_ms': 0.0,
            }
            return frame
        
        if actual_width != target_width or actual_height != target_height:
            garment_resized = garment_resized[:actual_height, :actual_width]
            mask_resized = mask_resized[:actual_height, :actual_width]

        if (
            self.use_nonrigid_warp
            and landmarks is not None
            and len(landmarks) > 24
        ):
            self._nonrigid_frame_counter += 1
            self._nonrigid_frames_since_refresh += 1

            cache_invalid = (
                self._cached_nonrigid_rgb is None
                or self._cached_nonrigid_mask is None
                or self._cached_nonrigid_rgb.shape != garment_resized.shape
                or self._cached_nonrigid_mask.shape != mask_resized.shape
            )

            headroom_threshold = self._nonrigid_budget_ms - self._nonrigid_margin_ms
            decision_frame_ms = self._runtime_frame_ema_ms if self._runtime_frame_ema_ms > 0.0 else self._runtime_last_frame_ms
            has_headroom = decision_frame_ms <= headroom_threshold
            target_interval = self._nonrigid_update_interval if has_headroom else self._nonrigid_backoff_interval
            should_refresh = cache_invalid or (self._nonrigid_frames_since_refresh >= target_interval)

            if should_refresh:
                warped_rgb, warped_mask, nr_stage = self._fit_piecewise_affine(
                    garment_resized,
                    mask_resized,
                    landmarks,
                    garment_x1,
                    garment_y1,
                    w,
                    h,
                )

                blend_ms = 0.0
                if (
                    self._cached_nonrigid_rgb is not None
                    and self._cached_nonrigid_mask is not None
                    and self._cached_nonrigid_rgb.shape == warped_rgb.shape
                    and self._cached_nonrigid_mask.shape == warped_mask.shape
                ):
                    t_blend = time.perf_counter()
                    a = float(np.clip(self._nonrigid_refresh_blend_alpha, 0.0, 1.0))
                    warped_rgb = self._cached_nonrigid_rgb * (1.0 - a) + warped_rgb * a
                    warped_mask = self._cached_nonrigid_mask * (1.0 - a) + warped_mask * a
                    blend_ms = (time.perf_counter() - t_blend) * 1000.0

                self._cached_nonrigid_rgb = warped_rgb.astype(np.float32, copy=False)
                self._cached_nonrigid_mask = np.clip(warped_mask.astype(np.float32, copy=False), 0.0, 1.0)
                self._nonrigid_frames_since_refresh = 0

                if self._nonrigid_profile_enabled:
                    self._nonrigid_profile_counter += 1
                    if self._nonrigid_profile_counter % self._nonrigid_profile_every == 0:
                        logger.info(
                            "[NONRIGID PROFILE] down=%.2fms warp=%.2fms up=%.2fms blend=%.2fms scale=%.2f interval=%d frame_ms=%.2f ema_ms=%.2f",
                            nr_stage.get('down_ms', 0.0),
                            nr_stage.get('warp_ms', 0.0),
                            nr_stage.get('up_ms', 0.0),
                            blend_ms,
                            float(self._nonrigid_warp_scale),
                            int(target_interval),
                            float(self._runtime_last_frame_ms),
                            float(self._runtime_frame_ema_ms),
                        )

            garment_resized = self._cached_nonrigid_rgb
            mask_resized = self._cached_nonrigid_mask

        roi_area = float(actual_width * actual_height)
        roi_ratio = float(roi_area / max(1.0, float(w * h)))
        t_post0 = time.perf_counter()
        
        # Extract frame region
        frame_region = frame[garment_y1:garment_y2, garment_x1:garment_x2]
        if frame_region.size == 0:
            self._last_fit_diag = {
                'roi_area': roi_area,
                'roi_ratio': roi_ratio,
                'post_ms': 0.0,
                'post_mask_ms': 0.0,
                'post_compose_ms': 0.0,
                'post_cvt_ms': 0.0,
                'post_writeback_ms': 0.0,
            }
            return frame
        
        # Apply body mask if available
        t_mask = time.perf_counter()
        if body_mask is not None:
            body_mask_region = body_mask[garment_y1:garment_y2, garment_x1:garment_x2]
            
            # Combine garment mask with body mask
            combined_mask = mask_resized * body_mask_region
        else:
            combined_mask = mask_resized
        mask_u8 = (combined_mask > 0.5).astype(np.uint8) * 255
        mask_ms = (time.perf_counter() - t_mask) * 1000.0
        
        output = self._get_scratch("fit_output", frame.shape, frame.dtype)
        np.copyto(output, frame)
        output_roi = output[garment_y1:garment_y2, garment_x1:garment_x2]
        warped_u8 = np.clip(garment_resized * 255.0, 0.0, 255.0).astype(np.uint8)

        # cvtColor-only timing retained for diagnostics parity.
        t_cvt = time.perf_counter()
        warped_bgr = self._get_scratch("fit_warp_bgr", warped_u8.shape, warped_u8.dtype)
        cv2.cvtColor(warped_u8, cv2.COLOR_RGB2BGR, dst=warped_bgr)
        cvt_ms = (time.perf_counter() - t_cvt) * 1000.0

        # Fast compose path: binary masked copy avoids float blend passes.
        t_comp = time.perf_counter()
        cv2.copyTo(warped_bgr, mask_u8, output_roi)
        comp_ms = (time.perf_counter() - t_comp) * 1000.0
        
        # writeback is done in-place through output ROI view above.
        t_write = time.perf_counter()
        _ = output_roi
        write_ms = (time.perf_counter() - t_write) * 1000.0

        post_ms = (time.perf_counter() - t_post0) * 1000.0
        self._last_fit_diag = {
            'roi_area': roi_area,
            'roi_ratio': roi_ratio,
            'post_ms': float(post_ms),
            'post_mask_ms': float(mask_ms),
            'post_compose_ms': float(comp_ms),
            'post_cvt_ms': float(cvt_ms),
            'post_writeback_ms': float(write_ms),
        }

        if self._nonrigid_profile_enabled and post_ms > 25.0:
            logger.warning(
                "[NONRIGID POST] roi_area=%.0f roi_ratio=%.3f post=%.2fms",
                roi_area,
                roi_ratio,
                post_ms,
            )

        if self._nonrigid_profile_enabled:
            self._nonrigid_post_profile_counter += 1
            if self._nonrigid_post_profile_counter % self._nonrigid_post_profile_every == 0:
                logger.info(
                    "[NONRIGID POST PROFILE] mask=%.2fms compose=%.2fms cvt=%.2fms write=%.2fms post=%.2fms roi_ratio=%.3f",
                    mask_ms,
                    comp_ms,
                    cvt_ms,
                    write_ms,
                    post_ms,
                    roi_ratio,
                )
        
        return output

    def get_last_fit_diag(self) -> Dict[str, float]:
        return dict(self._last_fit_diag)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic info about MediaPipe detection quality"""
        success_rate = (
            self.successful_detections / self.total_detections * 100
            if self.total_detections > 0 else 0
        )
        return {
            'status': self.last_detection_status,
            'confidence': self.last_confidence,
            'consecutive_failures': self.consecutive_failures,
            'success_rate': success_rate,
            'total_frames': self.total_detections,
            'successful_frames': self.successful_detections,
        }

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        body_measurements: Optional[dict] = None,
        show_box: bool = True,
        show_measurements: bool = True,
        show_skeleton: bool = True
    ) -> np.ndarray:
        """Draw debug information on frame including detection status"""
        output = self._get_scratch("debug_output", frame.shape, frame.dtype)
        np.copyto(output, frame)
        h, w = output.shape[:2]
        
        # Always show detection status (even when detection fails)
        diag = self.get_diagnostics()
        status = diag['status']
        confidence = diag['confidence']
        success_rate = diag['success_rate']
        
        # Status color: green=detected, yellow=low confidence, red=no person
        status_colors = {
            'detected': (0, 255, 0),
            'low_confidence': (0, 200, 255),
            'no_person': (0, 0, 255),
            'not_started': (128, 128, 128),
        }
        color = status_colors.get(status, (128, 128, 128))
        
        # Detection status badge (bottom-left)
        badge_y = h - 100
        cv2.rectangle(output, (10, badge_y), (320, h - 10), (0, 0, 0), -1)
        cv2.rectangle(output, (10, badge_y), (320, h - 10), color, 2)
        
        # Status indicator dot
        cv2.circle(output, (30, badge_y + 20), 8, color, -1)
        cv2.putText(output, f"Pose: {status.upper()}", (50, badge_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(output, f"Confidence: {confidence:.0%}", (20, badge_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(output, f"Success rate: {success_rate:.0f}%", (20, badge_y + 72),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if body_measurements is None:
            return output
        
        if show_box:
            # Draw torso box
            x1, y1, x2, y2 = body_measurements['torso_box']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if show_measurements:
            shoulder_width = body_measurements['shoulder_width']
            torso_height = body_measurements['torso_height']
            
            cv2.putText(output, f"Shoulder: {shoulder_width:.0f}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output, f"Torso: {torso_height:.0f}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if show_skeleton and 'landmarks' in body_measurements:
            landmarks = body_measurements['landmarks']
            # Draw key skeleton connections
            connections = [
                (11, 12), (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),              # Right arm
                (11, 23), (12, 24),              # Torso sides
                (23, 24),                        # Hips
                (23, 25), (25, 27),              # Left leg
                (24, 26), (26, 28),              # Right leg
            ]
            for (i, j) in connections:
                if i < len(landmarks) and j < len(landmarks):
                    pt1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
                    pt2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
                    vis = min(landmarks[i].visibility, landmarks[j].visibility)
                    if vis > 0.3:
                        line_color = (0, 255, 0) if vis > 0.6 else (0, 200, 255)
                        cv2.line(output, pt1, pt2, line_color, 2)
            
            # Draw landmark dots
            for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                if idx < len(landmarks) and landmarks[idx].visibility > 0.3:
                    pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                    cv2.circle(output, pt, 4, (255, 0, 255), -1)
        
        return output
    
    def __del__(self):
        """Cleanup"""
        if getattr(self, 'detector', None):
            self.detector.close()


# Simple test
if __name__ == "__main__":
    print("Body-Aware Garment Fitter - Ready!")
    print("This uses ONLY geometric transformations - no additional ML models!")
    print()
    print("How it works:")
    print("1. MediaPipe detects body landmarks")
    print("2. Extract shoulder width and torso height")
    print("3. Scale garment to match measurements")
    print("4. Position garment at correct location")
    print("5. Apply body mask for clean overlay")
    print()
    print("Result: Garments that fit YOUR actual body size!")

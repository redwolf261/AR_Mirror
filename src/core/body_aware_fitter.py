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

    def __init__(self, model_path: str = 'pose_landmarker_lite.task'):
        """Initialize MediaPipe pose detector"""
        self.model_path = model_path
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
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.VIDEO)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        print("[OK] Body-aware fitter initialized")
    
    def extract_body_measurements(self, frame: np.ndarray) -> Optional[dict]:
        """
        Extract body measurements from frame
        Returns dict with shoulder_width, torso_height, torso_box, body_mask
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Use REAL elapsed time for timestamps (fixes frame rejection at non-30fps)
        elapsed_ns = time.monotonic_ns() - self._start_time_ns
        timestamp_ms = elapsed_ns // 1_000_000
        
        self.total_detections += 1
        t_detect = time.perf_counter()
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
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
        
        # Track landmark confidence
        avg_confidence = (
            left_shoulder.visibility + right_shoulder.visibility +
            left_hip.visibility + right_hip.visibility
        ) / 4.0
        self.last_confidence = avg_confidence
        
        if avg_confidence < 0.3:
            self.last_detection_status = 'low_confidence'
            logger.debug(f"Low landmark confidence: {avg_confidence:.2f}")
            return None
        
        self.last_detection_status = 'detected'
        
        # Calculate measurements in pixels
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
        torso_height = abs(left_shoulder.y - left_hip.y) * h

        # --- Depth normalisation DISABLED ---
        # Normalising to a "canonical" shoulder width caused the garment to
        # appear undersized when the person stands close to the camera and
        # oversized when they stand far away.  The garment should always be
        # sized to the ACTUAL detected body size in the current frame.
        # shoulder_width, torso_height = self._apply_depth_normalisation(
        #     shoulder_width, torso_height, w, h
        # )

        # Calculate torso bounding box
        torso_x1 = int(min(left_shoulder.x, left_hip.x) * w)
        torso_y1 = int(left_shoulder.y * h)
        torso_x2 = int(max(right_shoulder.x, right_hip.x) * w)
        torso_y2 = int(left_hip.y * h)
        
        # Get body segmentation mask
        body_mask = None
        if detection_result.segmentation_masks:
            mask = detection_result.segmentation_masks[0].numpy_view()
            body_mask = (mask > 0.5).astype(np.uint8)
        
        # Import size recommendation system
        try:
            print(f"DEBUG: Starting size recommendation for measurements: shoulder_width={shoulder_width:.1f}, torso_height={torso_height:.1f}")
            from src.core.size_recommendation import get_size_recommendation, format_size_recommendation

            measurements = {
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'torso_box': (torso_x1, torso_y1, torso_x2, torso_y2),
                'body_mask': body_mask,
                'landmarks': landmarks,
                'confidence': avg_confidence
            }

            print(f"DEBUG: Calling get_size_recommendation...")
            # Get size recommendation
            size_rec = get_size_recommendation(measurements)
            print(f"DEBUG: Size recommendation result: {size_rec['recommended_size']} (confidence: {size_rec['confidence']:.2f})")

            measurements.update({
                'size_recommendation': size_rec['recommended_size'],
                'size_confidence': size_rec['confidence'],
                'size_alternatives': size_rec['all_sizes'],
                'measurements_cm': size_rec['measurements_cm'],
                'size_description': format_size_recommendation(size_rec)
            })

            # Add individual cm measurements for compatibility
            measurements.update({
                'shoulder_width_cm': size_rec['measurements_cm']['shoulder_width'],
                'chest_cm': size_rec['measurements_cm']['chest_width'],
                'torso_length_cm': size_rec['measurements_cm']['torso_length']
            })

            print(f"DEBUG: Final measurements with size data: size={measurements.get('size_recommendation')}")

        except Exception as e:
            # Fallback if size recommendation fails
            print(f"DEBUG: Size recommendation FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            logger.debug(f"Size recommendation failed: {e}")
            measurements = {
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'torso_box': (torso_x1, torso_y1, torso_x2, torso_y2),
                'body_mask': body_mask,
                'landmarks': landmarks,
                'confidence': avg_confidence,
                'size_recommendation': 'M',  # Default fallback
                'size_confidence': 0.5,
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
        
        # Calculate garment scaling
        # Add 10% padding for natural fit
        target_width = int(shoulder_width * 1.1)
        target_height = int(torso_height * 1.2)
        
        # Resize garment to match body size
        garment_resized = cv2.resize(garment_rgb, (target_width, target_height), 
                                     interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(garment_mask, (target_width, target_height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask has correct shape
        if mask_resized.ndim == 2:
            mask_resized = np.expand_dims(mask_resized, axis=-1)
        
        # Calculate position (center on torso)
        torso_center_x = (torso_x1 + torso_x2) // 2
        torso_center_y = (torso_y1 + torso_y2) // 2
        
        # Position garment
        garment_x1 = torso_center_x - target_width // 2
        garment_y1 = torso_y1  # Align with shoulders
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
            return frame
        
        if actual_width != target_width or actual_height != target_height:
            garment_resized = garment_resized[:actual_height, :actual_width]
            mask_resized = mask_resized[:actual_height, :actual_width]
        
        # Extract frame region
        frame_region = frame[garment_y1:garment_y2, garment_x1:garment_x2]
        if frame_region.size == 0:
            return frame
        frame_region_rgb = cv2.cvtColor(frame_region, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Apply body mask if available
        if body_mask is not None:
            body_mask_region = body_mask[garment_y1:garment_y2, garment_x1:garment_x2]
            if body_mask_region.ndim == 2:
                body_mask_region = np.expand_dims(body_mask_region, axis=-1)
            
            # Combine garment mask with body mask
            combined_mask = mask_resized * body_mask_region
        else:
            combined_mask = mask_resized
        
        # Alpha composite: garment * mask + background * (1 - mask)
        composite = garment_resized * combined_mask + frame_region_rgb * (1 - combined_mask)
        
        # Convert back to BGR uint8
        composite_bgr = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Blend into frame
        output = frame.copy()
        output[garment_y1:garment_y2, garment_x1:garment_x2] = composite_bgr
        
        return output
    
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
        output = frame.copy()
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

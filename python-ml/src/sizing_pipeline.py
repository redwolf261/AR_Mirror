import numpy as np
import cv2
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
        mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
        MEDIAPIPE_LEGACY = True
    else:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_LEGACY = False
except ImportError:
    print("Error: MediaPipe not installed. Run: pip install mediapipe")
    exit(1)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, List, Set
import json
import time
from pathlib import Path
from collections import deque
import csv
from .garment_visualizer import GarmentVisualizer, GarmentAssetManager


class BodyRegion(Enum):
    FACE = "FACE"
    HANDS = "HANDS"
    UPPER_BODY = "UPPER_BODY"
    LOWER_BODY = "LOWER_BODY"
    FULL_BODY = "FULL_BODY"


class ProductCategory(Enum):
    # Face accessories
    HAT = "HAT"
    SUNGLASSES = "SUNGLASSES"
    EARRINGS = "EARRINGS"
    MASK = "MASK"
    
    # Hand accessories
    WATCH = "WATCH"
    BRACELET = "BRACELET"
    RING = "RING"
    GLOVES = "GLOVES"
    
    # Upper body
    SHIRT = "SHIRT"
    JACKET = "JACKET"
    TOP = "TOP"
    BLOUSE = "BLOUSE"
    
    # Lower body
    PANTS = "PANTS"
    SHORTS = "SHORTS"
    SKIRT = "SKIRT"
    
    # Full body
    DRESS = "DRESS"
    SUIT = "SUIT"
    JUMPSUIT = "JUMPSUIT"


REGION_TO_PRODUCTS = {
    BodyRegion.FACE: [ProductCategory.HAT, ProductCategory.SUNGLASSES, 
                      ProductCategory.EARRINGS, ProductCategory.MASK],
    BodyRegion.HANDS: [ProductCategory.WATCH, ProductCategory.BRACELET, 
                       ProductCategory.RING, ProductCategory.GLOVES],
    BodyRegion.UPPER_BODY: [ProductCategory.SHIRT, ProductCategory.JACKET, 
                            ProductCategory.TOP, ProductCategory.BLOUSE],
    BodyRegion.LOWER_BODY: [ProductCategory.PANTS, ProductCategory.SHORTS, 
                            ProductCategory.SKIRT],
    BodyRegion.FULL_BODY: [ProductCategory.DRESS, ProductCategory.SUIT, 
                           ProductCategory.JUMPSUIT]
}


class FitDecision(Enum):
    TIGHT = "TIGHT"
    GOOD = "GOOD"
    LOOSE = "LOOSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class DetectionResult:
    detected_regions: Set[BodyRegion]
    available_categories: List[ProductCategory]
    landmarks: Dict
    confidence: float


@dataclass
class BodyMeasurements:
    shoulder_width_cm: float
    chest_width_cm: float
    torso_length_cm: float
    confidence: float
    timestamp: float
    detected_regions: Set[BodyRegion]
    head_width_cm: Optional[float] = None
    hand_length_cm: Optional[float] = None
    leg_length_cm: Optional[float] = None
    stability_score: Optional[float] = None  # NEW: 0-1, how stable measurements are


@dataclass
class GarmentSpec:
    sku: str
    shoulder_cm: float
    chest_cm: float
    length_cm: float
    size_label: str
    category: str = "SHIRT"


@dataclass
class FitResult:
    decision: FitDecision
    measurements: BodyMeasurements
    garment: GarmentSpec
    component_fits: Dict[str, FitDecision]
    confidence: float
    detected_regions: Set[BodyRegion]
    confidence_level: Optional[str] = None  # NEW: "HIGH", "MEDIUM", "LOW"
    alt_recommendation: Optional[str] = None  # NEW: "Try size M-L" for edge cases


@dataclass
class LightingQuality:
    """PHASE 1A: Lighting robustness metrics"""
    mean_brightness: float
    dark_ratio: float  # % pixels in dark range
    contrast: float
    is_acceptable: bool
    enhancement_applied: bool


class FramePreprocessor:
    def __init__(self, target_width=640, target_height=480):
        self.target_width = target_width
        self.target_height = target_height
        self.brightness_low = 40
        self.brightness_high = 220
        self.dark_threshold = 50  # Pixel values below this = dark
        self.dark_ratio_max = 0.4  # Max 40% dark pixels acceptable
        
    def analyze_lighting(self, frame: np.ndarray) -> LightingQuality:
        """PHASE 1A: Detailed lighting analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_ratio = np.sum(hist[:self.dark_threshold]) / total_pixels
        
        # Calculate contrast (std dev)
        contrast = np.std(gray)
        
        # Determine if lighting is acceptable
        is_acceptable = (
            self.brightness_low <= mean_brightness <= self.brightness_high
            and dark_ratio <= self.dark_ratio_max
            and contrast > 20  # Minimum contrast threshold
        )
        
        return LightingQuality(
            mean_brightness=float(mean_brightness),
            dark_ratio=float(dark_ratio),
            contrast=float(contrast),
            is_acceptable=bool(is_acceptable),
            enhancement_applied=False
        )
        
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, LightingQuality]:
        """Enhanced preprocessing with lighting quality feedback"""
        if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Analyze lighting first
        lighting = self.analyze_lighting(frame)
        
        # Reject if too dark/bright
        if not lighting.is_acceptable:
            # Try enhancement if too dark
            if lighting.mean_brightness < 80 and lighting.dark_ratio > 0.3:
                frame = self._enhance_dark_frame(frame)
                lighting = self.analyze_lighting(frame)
                lighting.enhancement_applied = True
                
                # Check if enhancement helped
                if not lighting.is_acceptable:
                    return frame, False, lighting
            else:
                return frame, False, lighting
        
        # Apply CLAHE for low-light optimization
        if lighting.mean_brightness < 80:
            frame = self._enhance_dark_frame(frame)
            lighting.enhancement_applied = True
        
        return frame, True, lighting
    
    def _enhance_dark_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement for dark frames"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class PoseDetector:
    def __init__(self):
        if MEDIAPIPE_LEGACY:
            self.mp_pose = mp_pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        else:
            # New MediaPipe API (0.10.30+) uses IMAGE mode for frame-by-frame
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6)
            self.pose = vision.PoseLandmarker.create_from_options(options)
        
        # Landmark groups for body region detection
        self.face_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Face points
        self.hand_landmarks = [15, 16, 17, 18, 19, 20, 21, 22]  # Wrists and hands
        self.upper_body_landmarks = [11, 12, 13, 14]  # Shoulders and elbows
        self.lower_body_landmarks = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Hips, knees, ankles
        
        self.min_visibility = 0.5  # Lowered for partial detection
        self.legacy_mode = MEDIAPIPE_LEGACY
        
    def detect_regions(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Detect which body regions are visible"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.legacy_mode:
            results = self.pose.process(rgb_frame)
            if not results.pose_landmarks:
                return None
            pose_landmarks = results.pose_landmarks.landmark
        else:
            # New API: create mp.Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.pose.detect(mp_image)
            if not results.pose_landmarks or len(results.pose_landmarks) == 0:
                return None
            pose_landmarks = results.pose_landmarks[0]
        
        # Extract all landmarks
        all_landmarks = {}
        for idx in range(len(pose_landmarks)):
            lm = pose_landmarks[idx]
            all_landmarks[idx] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }
        
        # Determine which regions are visible
        detected_regions = set()
        
        # Check face (need at least 3 face landmarks)
        face_visible = sum(1 for idx in self.face_landmarks 
                          if all_landmarks.get(idx, {}).get('visibility', 0) > self.min_visibility) >= 3
        if face_visible:
            detected_regions.add(BodyRegion.FACE)
        
        # Check hands (need at least 1 wrist)
        hands_visible = sum(1 for idx in self.hand_landmarks 
                           if all_landmarks.get(idx, {}).get('visibility', 0) > self.min_visibility) >= 1
        if hands_visible:
            detected_regions.add(BodyRegion.HANDS)
        
        # Check upper body (need both shoulders)
        upper_visible = sum(1 for idx in self.upper_body_landmarks 
                           if all_landmarks.get(idx, {}).get('visibility', 0) > self.min_visibility) >= 2
        if upper_visible:
            detected_regions.add(BodyRegion.UPPER_BODY)
        
        # Check lower body (need hips and at least one leg)
        lower_visible = sum(1 for idx in self.lower_body_landmarks 
                           if all_landmarks.get(idx, {}).get('visibility', 0) > self.min_visibility) >= 4
        if lower_visible:
            detected_regions.add(BodyRegion.LOWER_BODY)
        
        # Full body = upper + lower
        if BodyRegion.UPPER_BODY in detected_regions and BodyRegion.LOWER_BODY in detected_regions:
            detected_regions.add(BodyRegion.FULL_BODY)
        
        # Determine available product categories
        available_categories = []
        for region in detected_regions:
            if region in REGION_TO_PRODUCTS:
                available_categories.extend(REGION_TO_PRODUCTS[region])
        
        # Calculate overall confidence
        visible_count = sum(1 for lm in all_landmarks.values() 
                          if lm.get('visibility', 0) > self.min_visibility)
        confidence = visible_count / len(all_landmarks) if len(all_landmarks) > 0 else 0
        
        return DetectionResult(
            detected_regions=detected_regions,
            available_categories=list(set(available_categories)),
            landmarks=all_landmarks,
            confidence=confidence
        )
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Legacy method for backwards compatibility"""
        result = self.detect_regions(frame)
        if result is None:
            return None
        
        # Return only required landmarks for full body measurement
        required = [0, 11, 12, 23, 24]
        if all(result.landmarks.get(idx, {}).get('visibility', 0) > 0.7 for idx in required):
            return {idx: result.landmarks[idx] for idx in required}
        return None
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


class MeasurementEstimator:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reference_head_height_cm = 23.0
        self.valid_head_height_range = (80, 180)
        self.max_shoulder_tilt = 0.05
    
    def estimate_from_detection(self, detection: DetectionResult) -> Optional[BodyMeasurements]:
        """New method: estimate measurements based on detected regions"""
        landmarks = detection.landmarks
        detected_regions = detection.detected_regions
        
        # Initialize measurements with None
        shoulder_width = None
        chest_width = None
        torso_length = None
        head_width = None
        hand_length = None
        leg_length = None
        
        # Calculate scale factor if we have face
        scale_factor = None
        if BodyRegion.FACE in detected_regions or BodyRegion.UPPER_BODY in detected_regions:
            scale_factor = self._compute_scale_factor_flexible(landmarks)
        
        if scale_factor is None:
            return None
        
        # Measure based on available regions
        if BodyRegion.FACE in detected_regions:
            head_width = self._compute_head_width(landmarks, scale_factor)
        
        if BodyRegion.HANDS in detected_regions:
            hand_length = self._compute_hand_length(landmarks, scale_factor)
        
        if BodyRegion.UPPER_BODY in detected_regions:
            shoulder_width = self._compute_shoulder_width_safe(landmarks, scale_factor)
            chest_width = self._compute_chest_width_safe(landmarks, scale_factor)
        
        if BodyRegion.FULL_BODY in detected_regions:
            torso_length = self._compute_torso_length_safe(landmarks, scale_factor)
            leg_length = self._compute_leg_length(landmarks, scale_factor)
        
        confidence = detection.confidence
        
        return BodyMeasurements(
            shoulder_width_cm=shoulder_width or 0.0,
            chest_width_cm=chest_width or 0.0,
            torso_length_cm=torso_length or 0.0,
            confidence=confidence,
            timestamp=time.time(),
            detected_regions=detected_regions,
            head_width_cm=head_width,
            hand_length_cm=hand_length,
            leg_length_cm=leg_length
        )
    
    def estimate(self, landmarks: Dict) -> Optional[BodyMeasurements]:
        """Legacy method for full body measurement"""
        if not self._validate_pose(landmarks):
            return None
        
        scale_factor = self._compute_scale_factor(landmarks)
        if scale_factor is None:
            return None
        
        if not self._validate_head_to_shoulder_ratio(landmarks):
            return None
        
        # PHASE 1C: Detect pose state for correction
        pose_state = self._detect_pose_state(landmarks)
        
        shoulder_width = self._compute_shoulder_width(landmarks, scale_factor)
        chest_width = self._compute_chest_width(landmarks, scale_factor)
        torso_length = self._compute_torso_length(landmarks, scale_factor)
        
        # PHASE 1C: Apply pose-aware corrections
        shoulder_width = self._apply_pose_correction(shoulder_width, pose_state, 'shoulder')
        chest_width = self._apply_pose_correction(chest_width, pose_state, 'chest')
        torso_length = self._apply_pose_correction(torso_length, pose_state, 'torso')
        
        confidence = self._compute_confidence(landmarks)
        
        # Validate but don't reject - just reduce confidence for out-of-range values
        if not self._validate_measurements(shoulder_width, chest_width, torso_length):
            confidence *= 0.7  # Reduce confidence instead of rejecting
        
        # PHASE 1C: Reduce confidence for non-upright poses
        if pose_state != 'upright':
            confidence *= 0.9
        
        if confidence < 0.65:
            return None
        
        return BodyMeasurements(
            shoulder_width_cm=shoulder_width,
            chest_width_cm=chest_width,
            torso_length_cm=torso_length,
            confidence=confidence,
            timestamp=time.time(),
            detected_regions={BodyRegion.FULL_BODY}
        )
    
    def _detect_pose_state(self, landmarks: Dict) -> str:
        """PHASE 1C: Detect pose deviation for correction"""
        if not all(idx in landmarks for idx in [11, 12, 23, 24]):
            return 'unknown'
        
        # Check torso angle (slouch detection)
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]
        
        shoulder_hip_dx = abs(left_shoulder['x'] - left_hip['x'])
        shoulder_hip_dy = abs(left_shoulder['y'] - left_hip['y'])
        
        torso_angle = np.arctan2(shoulder_hip_dx, shoulder_hip_dy) if shoulder_hip_dy > 0 else 0
        torso_angle_deg = np.degrees(torso_angle)
        
        # Check shoulder tilt
        shoulder_tilt = abs(landmarks[11]['y'] - landmarks[12]['y'])
        
        # Classify pose
        if torso_angle_deg > 15:  # Significant lean
            return 'slouched'
        elif shoulder_tilt > 0.08:  # Tilted shoulders
            return 'tilted'
        else:
            return 'upright'
    
    def _apply_pose_correction(self, measurement: float, pose_state: str, measurement_type: str) -> float:
        """PHASE 1C: Apply correction factors based on pose state"""
        corrections = {
            'upright': {'shoulder': 1.0, 'chest': 1.0, 'torso': 1.0},
            'slouched': {'shoulder': 1.01, 'chest': 1.02, 'torso': 0.95},  # Slouch compresses torso, widens shoulder appearance
            'tilted': {'shoulder': 0.99, 'chest': 0.99, 'torso': 1.0},
            'unknown': {'shoulder': 1.0, 'chest': 1.0, 'torso': 1.0}
        }
        
        correction_factor = corrections.get(pose_state, {}).get(measurement_type, 1.0)
        return measurement * correction_factor
    
    def _validate_pose(self, landmarks: Dict) -> bool:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_tilt = abs(left_shoulder['y'] - right_shoulder['y'])
        return shoulder_tilt < self.max_shoulder_tilt
    
    def _validate_head_to_shoulder_ratio(self, landmarks: Dict) -> bool:
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        head_dist = abs(nose['y'] - shoulder_mid_y)
        
        shoulder_dist = abs(right_shoulder['x'] - left_shoulder['x'])
        
        ratio = head_dist / shoulder_dist if shoulder_dist > 0 else 0
        
        valid_ratio_range = (0.15, 0.45)
        return valid_ratio_range[0] < ratio < valid_ratio_range[1]
    
    def _compute_scale_factor_flexible(self, landmarks: Dict) -> Optional[float]:
        """Compute scale with fallback options"""
        # Try head-to-shoulder (best)
        if 0 in landmarks and 11 in landmarks and 12 in landmarks:
            primary_scale = self._compute_scale_factor(landmarks)
            
            # PHASE 1B: Validate with secondary anchor
            if primary_scale and self._validate_scale_with_shoulder(landmarks, primary_scale):
                return primary_scale
            else:
                # Scale failed validation, try fallback
                return self._compute_scale_factor_from_shoulder(landmarks)
        
        # Fallback: use shoulder width (assume 45cm average)
        return self._compute_scale_factor_from_shoulder(landmarks)
    
    def _compute_scale_factor_from_shoulder(self, landmarks: Dict) -> Optional[float]:
        """PHASE 1B: Fallback scale from shoulder width"""
        if 11 in landmarks and 12 in landmarks:
            left = landmarks[11]
            right = landmarks[12]
            pixel_dist = self._euclidean_distance(left, right) * self.frame_width
            return 45.0 / pixel_dist if pixel_dist > 0 else None
        return None
    
    def _validate_scale_with_shoulder(self, landmarks: Dict, scale: float) -> bool:
        """PHASE 1B: Cross-validate head-height scale using shoulder width"""
        if not (11 in landmarks and 12 in landmarks):
            return True  # Can't validate, accept primary
        
        # Compute shoulder width with primary scale
        shoulder_cm = self._compute_shoulder_width(landmarks, scale)
        
        # Expected range: 35-55cm for adults
        # If outside this range, head-height scale is likely wrong
        return 35.0 <= shoulder_cm <= 55.0
    
    def _compute_scale_factor(self, landmarks: Dict) -> Optional[float]:
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        head_height_pixels = abs(nose['y'] - shoulder_mid_y) * self.frame_height
        
        if not (self.valid_head_height_range[0] <= head_height_pixels <= self.valid_head_height_range[1]):
            return None
        
        return self.reference_head_height_cm / head_height_pixels
    
    def _compute_head_width(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Estimate head width from face landmarks"""
        if 7 in landmarks and 8 in landmarks:  # Left and right ear
            pixel_dist = self._euclidean_distance(landmarks[7], landmarks[8]) * self.frame_width
            return pixel_dist * scale
        return None
    
    def _compute_hand_length(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Estimate hand length from wrist to fingertip"""
        if 15 in landmarks and 19 in landmarks:  # Left wrist to left pinky
            pixel_dist = self._euclidean_distance(landmarks[15], landmarks[19]) * self.frame_height
            return pixel_dist * scale
        elif 16 in landmarks and 20 in landmarks:  # Right wrist to right pinky
            pixel_dist = self._euclidean_distance(landmarks[16], landmarks[20]) * self.frame_height
            return pixel_dist * scale
        return None
    
    def _compute_leg_length(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Estimate leg length from hip to ankle"""
        if 23 in landmarks and 27 in landmarks:  # Left hip to left ankle
            pixel_dist = self._euclidean_distance(landmarks[23], landmarks[27]) * self.frame_height
            return pixel_dist * scale
        elif 24 in landmarks and 28 in landmarks:  # Right hip to right ankle
            pixel_dist = self._euclidean_distance(landmarks[24], landmarks[28]) * self.frame_height
            return pixel_dist * scale
        return None
    
    def _compute_shoulder_width_safe(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Safe shoulder width calculation"""
        if 11 in landmarks and 12 in landmarks:
            return self._compute_shoulder_width(landmarks, scale)
        return None
    
    def _compute_chest_width_safe(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Safe chest width calculation"""
        if all(idx in landmarks for idx in [11, 12, 23, 24]):
            return self._compute_chest_width(landmarks, scale)
        return None
    
    def _compute_torso_length_safe(self, landmarks: Dict, scale: float) -> Optional[float]:
        """Safe torso length calculation"""
        if all(idx in landmarks for idx in [11, 12, 23, 24]):
            return self._compute_torso_length(landmarks, scale)
        return None
    
    def _compute_shoulder_width(self, landmarks: Dict, scale: float) -> float:
        left = landmarks[11]
        right = landmarks[12]
        pixel_dist = self._euclidean_distance(left, right) * self.frame_width
        return pixel_dist * scale
    
    def _compute_chest_width(self, landmarks: Dict, scale: float) -> float:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        left_chest_x = left_shoulder['x'] + 0.4 * (left_hip['x'] - left_shoulder['x'])
        left_chest_y = left_shoulder['y'] + 0.4 * (left_hip['y'] - left_shoulder['y'])
        
        right_chest_x = right_shoulder['x'] + 0.4 * (right_hip['x'] - right_shoulder['x'])
        right_chest_y = right_shoulder['y'] + 0.4 * (right_hip['y'] - right_shoulder['y'])
        
        chest_dist = np.sqrt((right_chest_x - left_chest_x)**2 + (right_chest_y - left_chest_y)**2)
        pixel_dist = chest_dist * self.frame_width
        return pixel_dist * scale
    
    def _compute_torso_length(self, landmarks: Dict, scale: float) -> float:
        shoulder_mid_y = (landmarks[11]['y'] + landmarks[12]['y']) / 2
        hip_mid_y = (landmarks[23]['y'] + landmarks[24]['y']) / 2
        
        pixel_dist = abs(shoulder_mid_y - hip_mid_y) * self.frame_height
        return pixel_dist * scale
    
    @staticmethod
    def _euclidean_distance(p1: Dict, p2: Dict) -> float:
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    @staticmethod
    def _validate_measurements(shoulder: float, chest: float, torso: float) -> bool:
        return (35 < shoulder < 55 and
                30 < chest < 60 and
                40 < torso < 80)  # Relaxed torso range from 45-75 to 40-80
    
    @staticmethod
    def _compute_confidence(landmarks: Dict) -> float:
        visibilities = [lm['visibility'] for lm in landmarks.values()]
        return min(visibilities)


class SizeMatcher:
    def __init__(self, garment_db_path: str):
        self.garment_db = self._load_garment_db(garment_db_path)
        self.ease_shoulder = 2.0
        self.ease_chest = 4.0
        self.ease_length = 3.0
        self.ease_tolerance = 4.0
        
    def _load_garment_db(self, path: str) -> Dict[str, GarmentSpec]:
        if not Path(path).exists():
            return {}
        with open(path, 'r') as f:
            data = json.load(f)
        return {
            item['sku']: GarmentSpec(
                sku=item['sku'],
                shoulder_cm=item['shoulder_cm'],
                chest_cm=item['chest_cm'],
                length_cm=item['length_cm'],
                size_label=item['size_label']
            )
            for item in data
        }
    
    def match(self, measurements: BodyMeasurements, sku: str) -> Optional[FitResult]:
        if sku not in self.garment_db:
            return None
        
        garment = self.garment_db[sku]
        
        fit_shoulder = self._categorize_fit(
            measurements.shoulder_width_cm,
            garment.shoulder_cm,
            self.ease_shoulder
        )
        
        fit_chest = self._categorize_fit(
            measurements.chest_width_cm,
            garment.chest_cm,
            self.ease_chest
        )
        
        fit_length = self._categorize_fit(
            measurements.torso_length_cm,
            garment.length_cm,
            self.ease_length
        )
        
        component_fits = {
            'shoulder': fit_shoulder,
            'chest': fit_chest,
            'length': fit_length
        }
        
        final_decision = self._aggregate_decision([fit_shoulder, fit_chest, fit_length])
        
        return FitResult(
            decision=final_decision,
            measurements=measurements,
            garment=garment,
            component_fits=component_fits,
            confidence=measurements.confidence,
            detected_regions=measurements.detected_regions
        )
    
    def _categorize_fit(self, body_measure: float, garment_measure: float, ease: float) -> FitDecision:
        diff = garment_measure - body_measure
        
        if diff < ease:
            return FitDecision.TIGHT
        elif ease <= diff <= (ease + self.ease_tolerance):
            return FitDecision.GOOD
        else:
            return FitDecision.LOOSE
    
    @staticmethod
    def _aggregate_decision(fits: List[FitDecision]) -> FitDecision:
        fit_counts = {fit: fits.count(fit) for fit in fits}
        return max(fit_counts, key=lambda x: fit_counts[x])


class AROverlayRenderer:
    def __init__(self):
        self.colors = {
            FitDecision.TIGHT: (0, 0, 255),
            FitDecision.GOOD: (0, 255, 0),
            FitDecision.LOOSE: (0, 255, 255),
            FitDecision.UNKNOWN: (128, 128, 128)
        }
        self.region_colors = {
            BodyRegion.FACE: (255, 255, 0),
            BodyRegion.HANDS: (255, 128, 0),
            BodyRegion.UPPER_BODY: (0, 255, 128),
            BodyRegion.LOWER_BODY: (128, 0, 255),
            BodyRegion.FULL_BODY: (0, 255, 0)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render_detection(self, frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """Render detected regions and available products"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw info panel on right side
        panel_x = w - 250
        y_offset = 30
        
        cv2.rectangle(overlay, (panel_x - 10, 10), (w - 10, min(h - 10, 400)), (0, 0, 0), -1)
        cv2.rectangle(overlay, (panel_x - 10, 10), (w - 10, min(h - 10, 400)), (255, 255, 255), 2)
        
        cv2.putText(overlay, "Detected Regions:", (panel_x, y_offset),
                   self.font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for region in detection.detected_regions:
            color = self.region_colors.get(region, (255, 255, 255))
            cv2.putText(overlay, f"  {region.value}", (panel_x, y_offset),
                       self.font, 0.5, color, 1)
            y_offset += 25
        
        y_offset += 10
        cv2.putText(overlay, "Try On:", (panel_x, y_offset),
                   self.font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for category in detection.available_categories[:6]:  # Limit to 6
            cv2.putText(overlay, f"  {category.value}", (panel_x, y_offset),
                       self.font, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        # Draw landmarks
        for idx, lm_data in detection.landmarks.items():
            if lm_data['visibility'] > 0.5:
                x = int(lm_data['x'] * w)
                y = int(lm_data['y'] * h)
                cv2.circle(overlay, (x, y), 3, (0, 255, 255), -1)
        
        return overlay
        
    def render(self, frame: np.ndarray, landmarks: Dict, result: FitResult, show_debug_shapes: bool = True) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # FEATURE 6: Camera auto-framing guide (silhouette overlay)
        if show_debug_shapes:
            self._draw_framing_guide(overlay)
        
        if landmarks and 11 in landmarks and 12 in landmarks:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            x1 = int(min(left_shoulder['x'], left_hip['x']) * w) - 20
            y1 = int(left_shoulder['y'] * h) - 20
            x2 = int(max(right_shoulder['x'], right_hip['x']) * w) + 20
            y2 = int(right_hip['y'] * h) + 20
            
            # FEATURE 2: Confidence indicator color based on stability
            if show_debug_shapes:
                color = self._get_confidence_color(result.measurements.stability_score)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                
                decision_text = result.decision.value
                cv2.putText(overlay, decision_text, (x1, y1 - 10),
                           self.font, 1.2, color, 2)
        
        if show_debug_shapes:
            # Enhanced measurement display with new features
            y_offset = 30
            
            # FEATURE 2 & 3: Show confidence level and stability
            stability_text = f"Stability: {result.measurements.stability_score or 0:.1%}" if result.measurements.stability_score else ""
            confidence_text = f"Confidence: {result.confidence_level or 'N/A'} ({result.confidence:.0%})"
            
            lines = [
                f"Shoulder: {result.measurements.shoulder_width_cm:.1f}cm ({result.component_fits['shoulder'].value})" if result.measurements.shoulder_width_cm > 0 else "",
                f"Chest: {result.measurements.chest_width_cm:.1f}cm ({result.component_fits['chest'].value})" if result.measurements.chest_width_cm > 0 else "",
                f"Torso: {result.measurements.torso_length_cm:.1f}cm ({result.component_fits['length'].value})" if result.measurements.torso_length_cm > 0 else "",
                f"Size: {result.garment.size_label}",
                confidence_text,
                stability_text
            ]
            
            # Add alternative recommendation if present
            if result.alt_recommendation:
                lines.append(f"Note: {result.alt_recommendation}")
            
            for line in lines:
                if line:  # Skip empty lines
                    cv2.putText(overlay, line, (10, y_offset),
                               self.font, 0.5, (255, 255, 255), 1)
                    y_offset += 25
        
        return overlay
    
    def _get_confidence_color(self, stability_score: Optional[float]) -> Tuple[int, int, int]:
        """FEATURE 2: Return color based on measurement stability (green=stable, yellow=ok, red=unstable)"""
        if stability_score is None:
            return (128, 128, 128)  # Gray for unknown
        
        if stability_score >= 0.75:
            return (0, 255, 0)  # Green - HIGH confidence
        elif stability_score >= 0.5:
            return (0, 255, 255)  # Yellow - MEDIUM confidence
        else:
            return (0, 0, 255)  # Red - LOW confidence
    
    def _draw_framing_guide(self, frame: np.ndarray):
        """FEATURE 6: Draw silhouette guide to help user position correctly"""
        h, w = frame.shape[:2]
        
        # Draw ideal framing zone (centered rectangle where user should fit)
        guide_w = int(w * 0.35)  # User should fill ~35% of width
        guide_h = int(h * 0.75)  # User should fill ~75% of height
        
        center_x = w // 2
        center_y = int(h * 0.55)  # Slightly below center
        
        x1 = center_x - guide_w // 2
        y1 = center_y - guide_h // 2
        x2 = center_x + guide_w // 2
        y2 = center_y + guide_h // 2
        
        # Draw dashed rectangle as guide
        dash_length = 20
        gap_length = 10
        
        # Top line
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), (100, 100, 255), 2)
        
        # Bottom line
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), (100, 100, 255), 2)
        
        # Left line
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), (100, 100, 255), 2)
        
        # Right line
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), (100, 100, 255), 2)
        
        # Add helper text
        cv2.putText(frame, "Fit body in guide", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)


class DataLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"sizing_log_{int(time.time())}.jsonl"
        self.session_id = str(time.time())
        
    def log_event(self, event_type: str, data: Dict):
        log_entry = {
            'timestamp': time.time(),
            'session_id': self.session_id,
            'event_type': event_type,
            'data': data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_fit_result(self, result: FitResult):
        data = {
            'decision': result.decision.value,
            'measurements': {
                'shoulder_cm': result.measurements.shoulder_width_cm,
                'chest_torso_breadth_cm': result.measurements.chest_width_cm,
                'torso_cm': result.measurements.torso_length_cm,
                'confidence': result.measurements.confidence
            },
            'garment': {
                'sku': result.garment.sku,
                'size_label': result.garment.size_label,
                'shoulder_cm': result.garment.shoulder_cm,
                'chest_cm': result.garment.chest_cm,
                'length_cm': result.garment.length_cm
            },
            'component_fits': {k: v.value for k, v in result.component_fits.items()}
        }
        self.log_event('fit_result', data)
    
    def log_failure(self, reason: str, frame_info: Dict):
        data = {
            'reason': reason,
            'frame_info': frame_info
        }
        self.log_event('failure', data)
    
    def log_manual_override(self, predicted_decision: str, staff_decision: str, sku: str, metadata: Optional[Dict] = None):
        data = {
            'predicted': predicted_decision,
            'staff_override': staff_decision,
            'sku': sku,
            'metadata': metadata or {}
        }
        self.log_event('manual_override', data)
    
    def log_purchase_outcome(self, sku: str, size_label: str, purchased: bool, return_reason: Optional[str] = None):
        data = {
            'sku': sku,
            'size_label': size_label,
            'purchased': purchased,
            'return_reason': return_reason
        }
        self.log_event('purchase_outcome', data)


class SizingPipeline:
    def __init__(self, garment_db_path: str, log_dir: str, adaptive_mode: bool = True, smoothing_window: int = 5, inventory_path: str = "garment_inventory.json"):
        self.preprocessor = FramePreprocessor()
        self.pose_detector = PoseDetector()
        self.measurement_estimator = MeasurementEstimator()
        self.size_matcher = SizeMatcher(garment_db_path)
        self.renderer = AROverlayRenderer()
        self.logger = DataLogger(log_dir)
        self.frame_times = []
        self.max_frame_times = 30
        self.adaptive_mode = adaptive_mode
        
        # FEATURE 1: Multi-frame smoothing
        self.smoothing_window = smoothing_window
        self.measurement_history = deque(maxlen=smoothing_window)
        
        # FEATURE 7: Measurement history export
        self.user_measurement_history = []  # Track all measurements per session
        
        # Load garment inventory
        self.inventory = self._load_inventory(inventory_path)
        self.current_garment_index = 0
        self.current_sku = None if not self.inventory else self.inventory[0]["sku"]
        
        # GARMENT VISUALIZATION: 2D overlay system
        self.garment_visualizer = GarmentVisualizer(GarmentAssetManager("garment_assets"))
        self.enable_garment_overlay = True
        self.current_size_override = None  # For manual size switching
        self.auto_matched_size = None  # Store the automatically matched size
        
    def set_garment(self, sku: str):
        self.current_sku = sku
    
    def _load_inventory(self, inventory_path: str) -> List[Dict]:
        """Load garment inventory from JSON file"""
        try:
            with open(inventory_path, 'r') as f:
                inventory = json.load(f)
            print(f"[+] Loaded {len(inventory)} garments from inventory")
            return inventory
        except FileNotFoundError:
            print(f"⚠ Inventory file not found: {inventory_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing inventory JSON: {e}")
            return []
    
    def _find_best_size(self, measurements: BodyMeasurements, garment: Dict) -> str:
        """Find best matching size for given measurements"""
        if not measurements or not garment:
            return "M"  # Default fallback
        
        sizes = garment.get("sizes", {})
        if not sizes:
            return "M"
        
        best_size = None
        best_score = float('inf')
        
        for size_label, size_spec in sizes.items():
            # Calculate fit score (lower is better)
            shoulder_diff = abs(measurements.shoulder_width_cm - size_spec["shoulder_cm"])
            chest_diff = abs(measurements.chest_width_cm - size_spec["chest_cm"])
            
            # Weighted score (shoulder is more important)
            score = (shoulder_diff * 1.5) + chest_diff
            
            if score < best_score:
                best_score = score
                best_size = size_label
        
        return best_size or "M"
    
    def cycle_garment(self, direction: int = 1):
        """Cycle through inventory (direction: 1=next, -1=previous)"""
        if not self.inventory:
            return
        
        self.current_garment_index = (self.current_garment_index + direction) % len(self.inventory)
        garment = self.inventory[self.current_garment_index]
        self.current_sku = garment["sku"]
        self.auto_matched_size = None  # Reset auto-match for new garment
        print(f"➜ Now showing: {garment['brand']} - {garment['name']} ({garment['color']}) - ₹{garment['price_inr']}")
    
    def _smooth_measurements(self, new_measurement: BodyMeasurements) -> BodyMeasurements:
        """FEATURE 1: Apply multi-frame smoothing to reduce jitter"""
        self.measurement_history.append({
            'shoulder': new_measurement.shoulder_width_cm,
            'chest': new_measurement.chest_width_cm,
            'torso': new_measurement.torso_length_cm,
            'confidence': new_measurement.confidence
        })
        
        if len(self.measurement_history) < 2:
            new_measurement.stability_score = 0.5
            return new_measurement
        
        # Calculate smoothed values
        shoulder_smooth = np.mean([m['shoulder'] for m in self.measurement_history])
        chest_smooth = np.mean([m['chest'] for m in self.measurement_history])
        torso_smooth = np.mean([m['torso'] for m in self.measurement_history])
        
        # Calculate stability (inverse of variance)
        shoulder_var = np.var([m['shoulder'] for m in self.measurement_history])
        chest_var = np.var([m['chest'] for m in self.measurement_history])
        stability_score = 1.0 / (1.0 + shoulder_var + chest_var)  # 0-1 range
        
        return BodyMeasurements(
            shoulder_width_cm=float(shoulder_smooth),
            chest_width_cm=float(chest_smooth),
            torso_length_cm=float(torso_smooth),
            confidence=new_measurement.confidence,
            timestamp=new_measurement.timestamp,
            detected_regions=new_measurement.detected_regions,
            head_width_cm=new_measurement.head_width_cm,
            hand_length_cm=new_measurement.hand_length_cm,
            leg_length_cm=new_measurement.leg_length_cm,
            stability_score=float(stability_score)
        )
    
    def _get_confidence_level(self, result: FitResult) -> str:
        """FEATURE 3: Determine confidence level based on measurements and stability"""
        stability = result.measurements.stability_score or 0.5
        pose_conf = result.measurements.confidence
        
        if pose_conf < 0.5:
            return "LOW" # Override: if pose is bad, confidence is low regardless of stability
            
        combined_confidence = (stability * 0.4) + (pose_conf * 0.6) # Give more weight to pose
        
        if combined_confidence >= 0.8:
            return "HIGH"
        elif combined_confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_alt_recommendation(self, result: FitResult) -> Optional[str]:
        """FEATURE 3: Check if measurement is on edge between sizes"""
        # Simple logic: if measurements are near boundaries, suggest trying both
        component_decisions = list(result.component_fits.values())
        
        # If mix of GOOD and adjacent sizes, suggest trying both
        has_good = FitDecision.GOOD in component_decisions
        has_tight = FitDecision.TIGHT in component_decisions
        has_loose = FitDecision.LOOSE in component_decisions
        
        if has_good and has_tight:
            return f"Between sizes - try {result.garment.size_label} and smaller"
        elif has_good and has_loose:
            return f"Between sizes - try {result.garment.size_label} and larger"
        elif result.confidence < 0.6:
            return "Low confidence - try on to confirm"
        
        return None
    
    def _get_posture_guidance(self, detection: Optional[DetectionResult], 
                             landmarks: Optional[Dict]) -> str:
        """FEATURE 4: Provide real-time posture coaching"""
        if detection is None and landmarks is None:
            return "Step back - full body not visible"
        
        if detection:
            if BodyRegion.LOWER_BODY not in detection.detected_regions:
                return "Move back - need to see legs"
            if BodyRegion.FACE not in detection.detected_regions:
                return "Move camera up - face not visible"
            if detection.confidence < 0.5:
                return "Stand still - pose unstable"
        
        if landmarks:
            # Check if shoulders are level (simple check)
            left_shoulder = landmarks.get(11)
            right_shoulder = landmarks.get(12)
            
            if left_shoulder and right_shoulder:
                shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
                if shoulder_diff > 0.1:  # More than 10% vertical difference
                    return "Stand straight - shoulders uneven"
        
        return "Good posture - hold position"
    
    def _check_lighting_quality(self, lighting: LightingQuality) -> str:
        """FEATURE 5: Analyze lighting quality and provide warnings"""
        if not lighting.is_acceptable:
            if lighting.mean_brightness < 80:
                return f"⚠ Too dark (brightness: {lighting.mean_brightness:.0f}) - add light"
            elif lighting.mean_brightness > 220:
                return f"⚠ Too bright (brightness: {lighting.mean_brightness:.0f}) - reduce glare"
            elif lighting.dark_ratio > 0.4:
                return f"⚠ Too many shadows ({lighting.dark_ratio*100:.0f}%) - improve lighting"
            else:
                return "⚠ Poor lighting - adjust position"
        
        if lighting.enhancement_applied:
            return "✓ Lighting enhanced"
        
        return "✓ Good lighting"
    
    def export_measurement_history(self, output_path: str, user_id: Optional[str] = None):
        """FEATURE 7: Export measurement history to CSV"""
        if not self.user_measurement_history:
            print("No measurements to export")
            return
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'user_id', 'shoulder_cm', 'chest_cm', 'torso_cm', 
                         'confidence', 'stability_score', 'decision', 'size_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.user_measurement_history:
                writer.writerow(entry)
        
        print(f"✓ Exported {len(self.user_measurement_history)} measurements to {output_path}")
    
    def _save_measurement_to_history(self, result: FitResult, user_id: Optional[str] = None):
        """FEATURE 7: Save measurement to history for export"""
        entry = {
            'timestamp': result.measurements.timestamp,
            'user_id': user_id or 'unknown',
            'shoulder_cm': result.measurements.shoulder_width_cm,
            'chest_cm': result.measurements.chest_width_cm,
            'torso_cm': result.measurements.torso_length_cm,
            'confidence': result.measurements.confidence,
            'stability_score': result.measurements.stability_score or 0,
            'decision': result.decision.value,
            'size_label': result.garment.size_label
        }
        self.user_measurement_history.append(entry)

    
    def process_frame_adaptive(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[DetectionResult]]:
        """Adaptive mode - show what's detected and available products"""
        frame_start = time.time()
        
        processed_frame, lighting_ok, lighting_quality = self.preprocessor.process(frame)
        
        # FEATURE 5: Show lighting quality feedback
        lighting_msg = self._check_lighting_quality(lighting_quality)
        
        if not lighting_ok:
            # Log detailed lighting failure
            self.logger.log_failure('poor_lighting', {
                'mean_brightness': lighting_quality.mean_brightness,
                'dark_ratio': lighting_quality.dark_ratio,
                'contrast': lighting_quality.contrast,
                'enhancement_applied': lighting_quality.enhancement_applied
            })
            error_frame = self._render_error(frame, lighting_msg)
            return error_frame, None
        
        detection = self.pose_detector.detect_regions(processed_frame)
        
        # FEATURE 4: Posture guidance
        posture_msg = self._get_posture_guidance(detection, detection.landmarks if detection else None)
        
        if detection is None:
            return self._render_error(frame, "No person detected"), None
        
        output_frame = self.renderer.render_detection(processed_frame, detection)
        
        # Overlay lighting and posture guidance
        self._add_guidance_overlay(output_frame, lighting_msg, posture_msg)
        
        return output_frame, detection
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[FitResult]]:
        """Original mode - full body measurement for garment fitting"""
        frame_start = time.time()
        
        processed_frame, lighting_ok, lighting_quality = self.preprocessor.process(frame)
        
        if not lighting_ok:
            self.logger.log_failure('poor_lighting', {
                'mean_brightness': lighting_quality.mean_brightness,
                'dark_ratio': lighting_quality.dark_ratio,
                'contrast': lighting_quality.contrast,
                'enhancement_applied': lighting_quality.enhancement_applied
            })
            return self._render_error(frame, "Poor lighting - adjust position"), None
        
        # Check if adaptive mode
        if self.adaptive_mode:
            detection = self.pose_detector.detect_regions(processed_frame)
            if detection is None:
                self.logger.log_failure('pose_detection_failed', {})
                return self._render_error(frame, "Stand in frame - need visible body parts"), None
            
            # If no garment selected, show adaptive view
            if self.current_sku is None:
                return self.renderer.render_detection(processed_frame, detection), None
            
            # Measure based on detection
            measurements = self.measurement_estimator.estimate_from_detection(detection)
            landmarks = detection.landmarks
        else:
            # Legacy full-body mode
            landmarks = self.pose_detector.detect(processed_frame)
            
            if landmarks is None:
                self.logger.log_failure('pose_detection_failed', {})
                return self._render_error(frame, "Stand straight, face camera"), None
            
            measurements = self.measurement_estimator.estimate(landmarks)
        
        if measurements is None:
            self.logger.log_failure('measurement_estimation_failed', {})
            return self._render_error(frame, "Move closer/farther or stand straight"), None
        
        # FEATURE 1: Apply multi-frame smoothing
        measurements = self._smooth_measurements(measurements)
        
        if self.current_sku is None:
            return self._render_error(frame, "No garment selected"), None
        
        # Auto-match best size if not already done
        if self.auto_matched_size is None and self.inventory:
            current_garment = next((g for g in self.inventory if g["sku"] == self.current_sku), None)
            if current_garment:
                self.auto_matched_size = self._find_best_size(measurements, current_garment)
                self.current_size_override = self.auto_matched_size
                print(f"[+] Auto-matched to size: {self.auto_matched_size}")
        
        result = self.size_matcher.match(measurements, self.current_sku)
        
        if result is None:
            self.logger.log_failure('garment_not_found', {'sku': self.current_sku})
            return self._render_error(frame, "Garment not in database"), None
        
        # Override size if manually selected or auto-matched
        if self.current_size_override:
            result.garment.size_label = self.current_size_override
        
        # FEATURE 3: Add confidence level and alternative recommendations
        result.confidence_level = self._get_confidence_level(result)
        result.alt_recommendation = self._get_alt_recommendation(result)
        
        # FEATURE 7: Save to measurement history
        self._save_measurement_to_history(result)
        
        self.logger.log_fit_result(result)
        
        output_frame = self.renderer.render(processed_frame, landmarks, result, show_debug_shapes=not self.enable_garment_overlay)
        
        # GARMENT VISUALIZATION: Overlay garment on body
        if self.enable_garment_overlay and landmarks:
            # Use override size if set, otherwise use matched size
            display_size = self.current_size_override or result.garment.size_label
            
            garment_image = self.garment_visualizer.asset_manager.get_garment(
                result.garment.sku,
                display_size.lower(),
                'front'
            )
            
            if garment_image is None:
                # Fallback if image missing
                print(f"[!] Missing asset for {result.garment.sku} {display_size}")
            else: 
                output_frame = self.garment_visualizer.overlay_on_body(
                    output_frame,
                    garment_image,
                    landmarks,
                    result.measurements.shoulder_width_cm,
                    result.measurements.chest_width_cm,
                    display_size
                )
        
        frame_end = time.time()
        frame_time_ms = (frame_end - frame_start) * 1000
        self.frame_times.append(frame_time_ms)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        return output_frame, result
    
    def _render_error(self, frame: np.ndarray, message: str) -> np.ndarray:
        output = frame.copy()
        cv2.putText(output, message, (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output
    
    def _add_guidance_overlay(self, frame: np.ndarray, lighting_msg: str, posture_msg: str):
        """FEATURE 4 & 5: Add lighting and posture guidance overlay"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent panel at bottom
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Lighting status (left side)
        light_color = (0, 255, 0) if "✓" in lighting_msg else (0, 165, 255)
        cv2.putText(frame, lighting_msg, (10, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 1)
        
        # Posture guidance (left side, below lighting)
        posture_color = (0, 255, 0) if "Good" in posture_msg else (0, 165, 255)
        cv2.putText(frame, f"Posture: {posture_msg}", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 1)


def main():
    garment_db_path = "garment_database.json"
    log_dir = "logs"
    
    pipeline = SizingPipeline(garment_db_path, log_dir)
    # First inventory item (TSH-001) is auto-set by __init__
    
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("AR Sizing System Started (ENHANCED)")
    print("=" * 60)
    print(f"Garment: SKU-001")
    print(f"Log file: {pipeline.logger.log_file}")
    print("NEW FEATURES:")
    print("  • Multi-frame smoothing (5-frame window)")
    print("  • Real-time confidence indicator (color-coded)")
    print("  • Enhanced size recommendations")
    print("  • Posture coaching")
    print("  • Lighting quality checker")
    print("  • Auto-framing guide")
    print("  • 2D Garment Try-On Overlay (NEW!)")
    print("Controls:")
    print("  'q' = quit  |  'e' = export measurements")
    print("  'n' = next size  |  'p' = previous size")
    print("  'g' = toggle garment overlay")
    print("=" * 60)
    
    frame_count = 0
    print("\n🎥 AR Sizing System Ready!")
    print("Controls:")
    print("  → (RIGHT ARROW) = next garment")
    print("  ← (LEFT ARROW) = previous garment")
    print("  n = next size (S→M→L→XL)")
    print("  p = previous size")
    print("  g = toggle garment overlay")
    print("  e = export measurements")
    print("  q = quit")
    print()
    
    # Size switching state
    sizes = ['S', 'M', 'L', 'XL']
    current_size_index = 1  # Start at M
    
    # Initialize tracking variables
    frame_count = 0
    detection_count = 0
    last_result = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        output_frame, result = pipeline.process_frame(frame)
        
        if result and result != last_result:
            detection_count += 1
            current_size = pipeline.current_size_override or result.garment.size_label
            print(f"\n[Frame {frame_count}] Measurement #{detection_count}")
            print(f"  Shoulder: {result.measurements.shoulder_width_cm:.1f}cm -> {result.component_fits['shoulder'].value}")
            print(f"  Chest:    {result.measurements.chest_width_cm:.1f}cm -> {result.component_fits['chest'].value}")
            print(f"  Torso:    {result.measurements.torso_length_cm:.1f}cm -> {result.component_fits['length'].value}")
            print(f"  Overall:  {result.decision.value} | Showing: Size {current_size}")
            print(f"  Confidence: {result.confidence_level} ({result.confidence:.0%}) | Stability: {result.measurements.stability_score or 0:.1%}")
            if result.alt_recommendation:
                print(f"  Note: {result.alt_recommendation}")
            last_result = result
        
        # Display current size on frame
        if pipeline.current_size_override:
            cv2.putText(output_frame, f"Viewing: Size {pipeline.current_size_override}", (10, output_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('AR Sizing', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow key
            pipeline.cycle_garment(1)
        elif key == 81:  # Left arrow key
            pipeline.cycle_garment(-1)
        elif key == ord('n'):  # Next size
            current_size_index = (current_size_index + 1) % len(sizes)
            pipeline.current_size_override = sizes[current_size_index]
            pipeline.auto_matched_size = None  # Clear auto-match when manually changing
            print(f"Size changed to: {sizes[current_size_index]}")
        elif key == ord('p'):  # Previous size
            current_size_index = (current_size_index - 1) % len(sizes)
            pipeline.current_size_override = sizes[current_size_index]
            pipeline.auto_matched_size = None  # Clear auto-match when manually changing
            print(f"Size changed to: {sizes[current_size_index]}")
        elif key == ord('g'):  # Toggle garment overlay
            pipeline.enable_garment_overlay = not pipeline.enable_garment_overlay
            status = "ON" if pipeline.enable_garment_overlay else "OFF"
            print(f"Garment overlay: {status}")
        elif key == ord('e'):
            # Export measurements to CSV
            export_path = f"measurement_history_{int(time.time())}.csv"
            pipeline.export_measurement_history(export_path)
            print(f"\n✓ Measurements exported to {export_path}")
        elif key == ord('n'):
            # Next size
            current_size_index = (current_size_index + 1) % len(sizes)
            pipeline.current_size_override = sizes[current_size_index]
            print(f"\n→ Switched to size: {pipeline.current_size_override}")
        elif key == ord('p'):
            # Previous size
            current_size_index = (current_size_index - 1) % len(sizes)
            pipeline.current_size_override = sizes[current_size_index]
            print(f"\n← Switched to size: {pipeline.current_size_override}")
        elif key == ord('g'):
            # Toggle garment overlay
            pipeline.enable_garment_overlay = not pipeline.enable_garment_overlay
            status = "ON" if pipeline.enable_garment_overlay else "OFF"
            print(f"\n🎽 Garment overlay: {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Auto-export at end of session
    if detection_count > 0:
        export_path = f"logs/measurement_history_{int(time.time())}.csv"
        pipeline.export_measurement_history(export_path)
    
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Successful measurements: {detection_count}")
    print(f"Success rate: {(detection_count/frame_count*100):.1f}%" if frame_count > 0 else "N/A")
    print(f"Log saved to: {pipeline.logger.log_file}")
    if detection_count > 0:
        print(f"Measurements exported to: {export_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

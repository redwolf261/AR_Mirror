"""
Lower-Body Measurement Estimation
Adds waist, hip, and inseam measurements for pants/skirt sizing
PHASE 1B implementation
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class LowerBodyMeasurements:
    hip_width_cm: float
    waist_estimate_cm: float
    inseam_cm: float
    confidence: float
    timestamp: float

class LowerBodyEstimator:
    """Estimates lower-body measurements from MediaPipe landmarks"""
    
    # Typical body proportion ratios
    WAIST_TO_HIP_RATIO = 0.72  # Empirical average
    INSEAM_CORRECTION_FACTOR = 1.15  # Account for knee-to-ankle vs true inseam
    
    # Hip width expected range (cm)
    HIP_MIN = 30.0
    HIP_MAX = 60.0
    
    # Confidence thresholds
    MIN_VISIBILITY = 0.5
    MIN_LANDMARK_CONFIDENCE = 0.3
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def estimate(self, landmarks: Dict, scale_factor: float) -> Optional[LowerBodyMeasurements]:
        """
        Estimate lower-body measurements from landmarks
        
        Required landmarks:
        - 23: Left hip
        - 24: Right hip
        - 25: Left knee
        - 26: Right knee
        - 27: Left ankle
        - 28: Right ankle
        """
        
        # Check required landmarks exist
        required = [23, 24, 25, 26, 27, 28]
        if not all(idx in landmarks for idx in required):
            return None
        
        # Check visibility
        if not self._check_visibility(landmarks, required):
            return None
        
        # Calculate hip width
        hip_left = landmarks[23]
        hip_right = landmarks[24]
        
        hip_dist_pixels = np.sqrt(
            (hip_left['x'] - hip_right['x']) ** 2 +
            (hip_left['y'] - hip_right['y']) ** 2
        ) * self.frame_width
        
        hip_width_cm = hip_dist_pixels * scale_factor
        
        # Validate hip width
        if not (self.HIP_MIN <= hip_width_cm <= self.HIP_MAX):
            return None
        
        # Estimate waist (typically 72% of hip width)
        waist_estimate_cm = hip_width_cm * self.WAIST_TO_HIP_RATIO
        
        # Calculate inseam (hip to ankle distance)
        knee_left = landmarks[25]
        ankle_left = landmarks[27]
        
        inseam_pixels = np.sqrt(
            (knee_left['x'] - ankle_left['x']) ** 2 +
            (knee_left['y'] - ankle_left['y']) ** 2
        ) * self.frame_height
        
        inseam_cm = inseam_pixels * scale_factor * self.INSEAM_CORRECTION_FACTOR
        
        # Calculate confidence
        confidence = self._calculate_confidence(landmarks, required)
        
        return LowerBodyMeasurements(
            hip_width_cm=hip_width_cm,
            waist_estimate_cm=waist_estimate_cm,
            inseam_cm=inseam_cm,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _check_visibility(self, landmarks: Dict, indices: list) -> bool:
        """Check if required landmarks are visible"""
        for idx in indices:
            if landmarks[idx].get('visibility', 0) < self.MIN_VISIBILITY:
                return False
        return True
    
    def _calculate_confidence(self, landmarks: Dict, indices: list) -> float:
        """Calculate measurement confidence based on landmark quality"""
        visibilities = [landmarks[idx].get('visibility', 0) for idx in indices]
        avg_visibility = np.mean(visibilities)
        
        # Check symmetry (left vs right hip/knee/ankle should be similar)
        hip_symmetry = self._check_symmetry(landmarks[23], landmarks[24])
        knee_symmetry = self._check_symmetry(landmarks[25], landmarks[26])
        ankle_symmetry = self._check_symmetry(landmarks[27], landmarks[28])
        
        symmetry_score = (hip_symmetry + knee_symmetry + ankle_symmetry) / 3
        
        # Combined confidence
        confidence = avg_visibility * 0.6 + symmetry_score * 0.4
        
        return float(confidence)
    
    def _check_symmetry(self, left_landmark: Dict, right_landmark: Dict) -> float:
        """Check if left and right landmarks are at similar heights (good posture)"""
        y_diff = abs(left_landmark['y'] - right_landmark['y'])
        
        # Perfect symmetry = 1.0, >10% difference = 0.0
        if y_diff < 0.05:
            return 1.0
        elif y_diff > 0.15:
            return 0.0
        else:
            return 1.0 - (y_diff - 0.05) / 0.10


@dataclass
class PantsSpec:
    sku: str
    waist_cm: float
    hip_cm: float
    inseam_cm: float
    size_label: str

class PantsSizeMatcher:
    """Match lower-body measurements to pants specifications"""
    
    # Fit tolerances (cm)
    WAIST_TIGHT_THRESHOLD = 1.5
    WAIST_LOOSE_THRESHOLD = 3.0
    HIP_TIGHT_THRESHOLD = 2.0
    HIP_LOOSE_THRESHOLD = 4.0
    INSEAM_TIGHT_THRESHOLD = 2.0
    INSEAM_LOOSE_THRESHOLD = 5.0
    
    def fit_pants(self, measurements: LowerBodyMeasurements, 
                  spec: PantsSpec) -> Dict:
        """
        Determine if pants will fit based on measurements
        
        Returns:
        {
            'decision': 'TIGHT' | 'GOOD' | 'LOOSE',
            'component_fits': {
                'waist': 'TIGHT' | 'GOOD' | 'LOOSE',
                'hip': 'TIGHT' | 'GOOD' | 'LOOSE',
                'inseam': 'TIGHT' | 'GOOD' | 'LOOSE'
            },
            'confidence': float,
            'explanation': str
        }
        """
        
        # Compare waist
        waist_diff = spec.waist_cm - measurements.waist_estimate_cm
        if waist_diff < -self.WAIST_TIGHT_THRESHOLD:
            waist_fit = 'TIGHT'
        elif waist_diff > self.WAIST_LOOSE_THRESHOLD:
            waist_fit = 'LOOSE'
        else:
            waist_fit = 'GOOD'
        
        # Compare hip
        hip_diff = spec.hip_cm - measurements.hip_width_cm
        if hip_diff < -self.HIP_TIGHT_THRESHOLD:
            hip_fit = 'TIGHT'
        elif hip_diff > self.HIP_LOOSE_THRESHOLD:
            hip_fit = 'LOOSE'
        else:
            hip_fit = 'GOOD'
        
        # Compare inseam
        inseam_diff = spec.inseam_cm - measurements.inseam_cm
        if inseam_diff < -self.INSEAM_TIGHT_THRESHOLD:
            inseam_fit = 'TIGHT'
        elif inseam_diff > self.INSEAM_LOOSE_THRESHOLD:
            inseam_fit = 'LOOSE'
        else:
            inseam_fit = 'GOOD'
        
        component_fits = {
            'waist': waist_fit,
            'hip': hip_fit,
            'inseam': inseam_fit
        }
        
        # Aggregate decision (most conservative fit)
        if 'TIGHT' in component_fits.values():
            decision = 'TIGHT'
            explanation = self._get_tight_explanation(component_fits)
        elif 'LOOSE' in component_fits.values():
            decision = 'LOOSE'
            explanation = self._get_loose_explanation(component_fits)
        else:
            decision = 'GOOD'
            explanation = "All measurements within good fit range"
        
        # Adjust confidence based on fit consistency
        confidence = measurements.confidence
        if len(set(component_fits.values())) > 1:
            # Mixed fits reduce confidence
            confidence *= 0.9
        
        return {
            'decision': decision,
            'component_fits': component_fits,
            'confidence': confidence,
            'explanation': explanation,
            'measurements': {
                'waist_diff_cm': waist_diff,
                'hip_diff_cm': hip_diff,
                'inseam_diff_cm': inseam_diff
            }
        }
    
    def _get_tight_explanation(self, fits: Dict) -> str:
        tight_parts = [k for k, v in fits.items() if v == 'TIGHT']
        return f"Too tight at: {', '.join(tight_parts)}. Consider sizing up."
    
    def _get_loose_explanation(self, fits: Dict) -> str:
        loose_parts = [k for k, v in fits.items() if v == 'LOOSE']
        return f"Too loose at: {', '.join(loose_parts)}. Consider sizing down."


def main():
    """Demo: Lower-body measurement estimation"""
    print("=" * 70)
    print("LOWER-BODY MEASUREMENT DEMO")
    print("=" * 70)
    
    estimator = LowerBodyEstimator(640, 480)
    
    # Simulate landmarks (typical adult proportions)
    print("\n1. Simulating pose landmarks...")
    landmarks = {
        0: {'x': 0.5, 'y': 0.15, 'visibility': 0.95},     # nose
        11: {'x': 0.45, 'y': 0.35, 'visibility': 0.95},   # left shoulder
        12: {'x': 0.55, 'y': 0.35, 'visibility': 0.95},   # right shoulder
        23: {'x': 0.46, 'y': 0.55, 'visibility': 0.90},   # left hip
        24: {'x': 0.54, 'y': 0.55, 'visibility': 0.90},   # right hip
        25: {'x': 0.46, 'y': 0.72, 'visibility': 0.85},   # left knee
        26: {'x': 0.54, 'y': 0.72, 'visibility': 0.85},   # right knee
        27: {'x': 0.46, 'y': 0.92, 'visibility': 0.80},   # left ankle
        28: {'x': 0.54, 'y': 0.92, 'visibility': 0.80},   # right ankle
    }
    
    # Assume scale factor from head-height (from existing system)
    scale_factor = 0.35  # pixels to cm
    
    print("\n2. Estimating lower-body measurements...")
    measurements = estimator.estimate(landmarks, scale_factor)
    
    if measurements:
        print(f"   ✓ Hip width: {measurements.hip_width_cm:.1f} cm")
        print(f"   ✓ Waist estimate: {measurements.waist_estimate_cm:.1f} cm")
        print(f"   ✓ Inseam: {measurements.inseam_cm:.1f} cm")
        print(f"   ✓ Confidence: {measurements.confidence:.2f}")
    else:
        print("   ✗ Failed to estimate measurements")
        return
    
    # Test pants fitting
    print("\n3. Testing pants fit...")
    
    # Test case 1: Good fit
    pants_good = PantsSpec(
        sku="pants_m_001",
        waist_cm=measurements.waist_estimate_cm + 1.0,
        hip_cm=measurements.hip_width_cm + 2.0,
        inseam_cm=measurements.inseam_cm + 1.0,
        size_label="M"
    )
    
    matcher = PantsSizeMatcher()
    result_good = matcher.fit_pants(measurements, pants_good)
    
    print(f"\n   Pants Size M (waist={pants_good.waist_cm:.1f}, hip={pants_good.hip_cm:.1f}):")
    print(f"   Decision: {result_good['decision']}")
    print(f"   Component fits: {result_good['component_fits']}")
    print(f"   Explanation: {result_good['explanation']}")
    print(f"   Confidence: {result_good['confidence']:.2f}")
    
    # Test case 2: Tight fit
    pants_tight = PantsSpec(
        sku="pants_s_001",
        waist_cm=measurements.waist_estimate_cm - 3.0,
        hip_cm=measurements.hip_width_cm - 2.0,
        inseam_cm=measurements.inseam_cm - 1.0,
        size_label="S"
    )
    
    result_tight = matcher.fit_pants(measurements, pants_tight)
    
    print(f"\n   Pants Size S (waist={pants_tight.waist_cm:.1f}, hip={pants_tight.hip_cm:.1f}):")
    print(f"   Decision: {result_tight['decision']}")
    print(f"   Component fits: {result_tight['component_fits']}")
    print(f"   Explanation: {result_tight['explanation']}")
    
    # Test case 3: Loose fit
    pants_loose = PantsSpec(
        sku="pants_l_001",
        waist_cm=measurements.waist_estimate_cm + 5.0,
        hip_cm=measurements.hip_width_cm + 6.0,
        inseam_cm=measurements.inseam_cm + 4.0,
        size_label="L"
    )
    
    result_loose = matcher.fit_pants(measurements, pants_loose)
    
    print(f"\n   Pants Size L (waist={pants_loose.waist_cm:.1f}, hip={pants_loose.hip_cm:.1f}):")
    print(f"   Decision: {result_loose['decision']}")
    print(f"   Component fits: {result_loose['component_fits']}")
    print(f"   Explanation: {result_loose['explanation']}")
    
    print("\n" + "=" * 70)
    print("Lower-body measurement system ready!")
    print("Enables: Pants, Skirts, Dresses with accurate waist/hip/inseam fit")
    print("=" * 70)


if __name__ == "__main__":
    main()

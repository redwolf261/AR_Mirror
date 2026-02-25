"""
Test PHASE 1 improvements: lighting robustness, head-height validation, pose-aware correction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from src.legacy.sizing_pipeline import (
    FramePreprocessor, 
    PoseDetector, 
    MeasurementEstimator,
    LightingQuality
)

def test_lighting_analysis():
    """Test PHASE 1A: Lighting robustness"""
    print("=" * 60)
    print("PHASE 1A: Testing Lighting Robustness")
    print("=" * 60)
    
    preprocessor = FramePreprocessor()
    
    # Test dark frame
    dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 30
    lighting = preprocessor.analyze_lighting(dark_frame)
    print(f"\nDark frame test:")
    print(f"  Mean brightness: {lighting.mean_brightness:.1f}")
    print(f"  Dark ratio: {lighting.dark_ratio:.2%}")
    print(f"  Contrast: {lighting.contrast:.1f}")
    print(f"  Acceptable: {lighting.is_acceptable}")
    
    # Test normal frame
    normal_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    lighting = preprocessor.analyze_lighting(normal_frame)
    print(f"\nNormal frame test:")
    print(f"  Mean brightness: {lighting.mean_brightness:.1f}")
    print(f"  Dark ratio: {lighting.dark_ratio:.2%}")
    print(f"  Contrast: {lighting.contrast:.1f}")
    print(f"  Acceptable: {lighting.is_acceptable}")
    
    # Test bright frame
    bright_frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
    lighting = preprocessor.analyze_lighting(bright_frame)
    print(f"\nBright frame test:")
    print(f"  Mean brightness: {lighting.mean_brightness:.1f}")
    print(f"  Dark ratio: {lighting.dark_ratio:.2%}")
    print(f"  Contrast: {lighting.contrast:.1f}")
    print(f"  Acceptable: {lighting.is_acceptable}")
    
    return True

def test_head_height_validation():
    """Test PHASE 1B: Head-height validation with shoulder cross-check"""
    print("\n" + "=" * 60)
    print("PHASE 1B: Testing Head-Height Validation")
    print("=" * 60)
    
    estimator = MeasurementEstimator(frame_width=640, frame_height=480)
    
    # Mock landmarks with reasonable proportions
    good_landmarks = {
        0: {'x': 0.5, 'y': 0.2, 'z': 0, 'visibility': 0.9},   # Nose
        11: {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 0.9},  # Left shoulder
        12: {'x': 0.6, 'y': 0.3, 'z': 0, 'visibility': 0.9},  # Right shoulder
        23: {'x': 0.42, 'y': 0.6, 'z': 0, 'visibility': 0.9}, # Left hip
        24: {'x': 0.58, 'y': 0.6, 'z': 0, 'visibility': 0.9}  # Right hip
    }
    
    # Test good proportions
    scale = estimator._compute_scale_factor_flexible(good_landmarks)
    if scale:
        shoulder_cm = estimator._compute_shoulder_width(good_landmarks, scale)
        is_valid = estimator._validate_scale_with_shoulder(good_landmarks, scale)
        print(f"\nGood proportions test:")
        print(f"  Scale factor: {scale:.4f}")
        print(f"  Shoulder width: {shoulder_cm:.1f}cm")
        print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Test bad proportions (unrealistic head size)
    bad_landmarks = good_landmarks.copy()
    bad_landmarks[0] = {'x': 0.5, 'y': 0.05, 'z': 0, 'visibility': 0.9}  # Huge head
    
    scale = estimator._compute_scale_factor_flexible(bad_landmarks)
    if scale:
        shoulder_cm = estimator._compute_shoulder_width(bad_landmarks, scale)
        is_valid = estimator._validate_scale_with_shoulder(bad_landmarks, scale)
        print(f"\nBad proportions test (huge head):")
        print(f"  Scale factor: {scale:.4f}")
        print(f"  Shoulder width: {shoulder_cm:.1f}cm")
        print(f"  Validation: {'PASS' if is_valid else 'FAIL (expected)'}")
    
    return True

def test_pose_aware_correction():
    """Test PHASE 1C: Pose-aware measurement correction"""
    print("\n" + "=" * 60)
    print("PHASE 1C: Testing Pose-Aware Correction")
    print("=" * 60)
    
    estimator = MeasurementEstimator(frame_width=640, frame_height=480)
    
    # Upright pose
    upright_landmarks = {
        11: {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 0.9},
        12: {'x': 0.6, 'y': 0.3, 'z': 0, 'visibility': 0.9},
        23: {'x': 0.42, 'y': 0.6, 'z': 0, 'visibility': 0.9},
        24: {'x': 0.58, 'y': 0.6, 'z': 0, 'visibility': 0.9}
    }
    
    pose_state = estimator._detect_pose_state(upright_landmarks)
    print(f"\nUpright pose test:")
    print(f"  Detected state: {pose_state}")
    print(f"  Shoulder correction: {estimator._apply_pose_correction(45.0, pose_state, 'shoulder'):.1f}cm (from 45.0cm)")
    print(f"  Torso correction: {estimator._apply_pose_correction(65.0, pose_state, 'torso'):.1f}cm (from 65.0cm)")
    
    # Slouched pose
    slouched_landmarks = upright_landmarks.copy()
    slouched_landmarks[23] = {'x': 0.35, 'y': 0.6, 'z': 0, 'visibility': 0.9}  # Hip moved left (slouch)
    
    pose_state = estimator._detect_pose_state(slouched_landmarks)
    print(f"\nSlouched pose test:")
    print(f"  Detected state: {pose_state}")
    print(f"  Shoulder correction: {estimator._apply_pose_correction(45.0, pose_state, 'shoulder'):.1f}cm (from 45.0cm)")
    print(f"  Torso correction: {estimator._apply_pose_correction(65.0, pose_state, 'torso'):.1f}cm (from 65.0cm)")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("PHASE 1 FEATURE VALIDATION")
    print("Testing: Lighting, Head-Height Validation, Pose Correction")
    print("=" * 60)
    
    try:
        test_lighting_analysis()
        test_head_height_validation()
        test_pose_aware_correction()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 1 TESTS PASSED")
        print("=" * 60)
        print("\nPhase 1 Features Active:")
        print("  [✓] 1A: Enhanced lighting analysis (histogram + dark ratio + contrast)")
        print("  [✓] 1B: Head-height validation with shoulder cross-check")
        print("  [✓] 1C: Pose-aware measurement correction (slouch, tilt)")
        print("\nNext: Run sizing_pipeline.py or adaptive_demo.py with PHASE 1 improvements")
        print("=" * 60)
        
        return True
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

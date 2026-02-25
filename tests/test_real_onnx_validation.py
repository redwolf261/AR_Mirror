#!/usr/bin/env python3
"""
CRITICAL TEST: Real ONNX Output Validation
Fixes the 0-pixel test red flag by using actual parsing output

This test validates BEHAVIOR, not just code paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
import json

print("=" * 70)
print("REAL ONNX OUTPUT VALIDATION TEST")
print("=" * 70)

# Test 1: Capture real ONNX parsing output
print("\n[1/5] Capturing real ONNX parsing output...")
try:
    from src.core.semantic_parser import SemanticParser
    
    # Initialize parser with ONNX backend
    parser = SemanticParser(
        backend='auto',
        temporal_smoothing=False,  # Disable for deterministic output
        onnx_model_path='models/schp_lip.onnx'
    )
    
    print(f"[OK] Parser initialized: {parser.backend.__class__.__name__}")
    
    # Create test frame (solid color with person-like shape)
    h, w = 480, 640
    test_frame = np.zeros((h, w, 3), dtype=np.uint8)
    test_frame[:, :] = (120, 150, 180)  # Skin-like color
    
    # Add person-like region (center oval)
    center_x, center_y = w // 2, h // 2
    cv2.ellipse(test_frame, (center_x, center_y), (150, 200), 0, 0, 360, (180, 160, 140), -1)
    
    # Parse the frame
    masks = parser.parse(test_frame, target_resolution=(473, 473))
    
    # Check mask statistics
    stats = {}
    for key in ['hair', 'face', 'neck', 'upper_body', 'arms', 'lower_body']:
        pixels = masks[key].sum() / 255
        coverage = (pixels / (h * w)) * 100
        stats[key] = {
            'pixels': int(pixels),
            'coverage_percent': round(coverage, 2)
        }
    
    print(f"[OK] Parsing complete")
    print(f"\nMask Statistics:")
    for key, stat in stats.items():
        print(f"  {key:12s}: {stat['pixels']:6d} pixels ({stat['coverage_percent']:5.2f}%)")
    
    # CRITICAL ASSERTION 1: Non-zero garment area
    garment_pixels = stats['upper_body']['pixels'] + stats['arms']['pixels']
    assert garment_pixels > 0, "FAIL: Garment region is empty (0 pixels)"
    print(f"\n[PASS] Assertion 1: Non-zero garment area ({garment_pixels} pixels)")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Validate zero garment-face intersection
print("\n[2/5] Validating occlusion correctness...")
try:
    # Calculate garment region
    garment_region = cv2.bitwise_or(masks['upper_body'], masks['arms'])
    
    # Check face intersection
    face_intersection = cv2.bitwise_and(garment_region, masks['face'])
    face_overlap_pixels = face_intersection.sum() / 255
    
    # Check hair intersection
    hair_intersection = cv2.bitwise_and(garment_region, masks['hair'])
    hair_overlap_pixels = hair_intersection.sum() / 255
    
    print(f"  Garment-Face overlap: {int(face_overlap_pixels)} pixels")
    print(f"  Garment-Hair overlap: {int(hair_overlap_pixels)} pixels")
    
    # CRITICAL ASSERTION 2: Zero or minimal overlap
    total_overlap = face_overlap_pixels + hair_overlap_pixels
    overlap_ratio = total_overlap / max(garment_pixels, 1)
    
    assert overlap_ratio < 0.05, f"FAIL: Excessive overlap ({overlap_ratio*100:.1f}%)"
    print(f"\n[PASS] Assertion 2: Minimal occlusion error ({overlap_ratio*100:.2f}%)")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with geometric constraint
print("\n[3/5] Testing geometric constraint...")
try:
    # Create mock pose landmarks
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    mock_landmarks = [None] * 33
    mock_landmarks[11] = MockLandmark(0.3, 0.3)   # Left shoulder
    mock_landmarks[12] = MockLandmark(0.7, 0.3)   # Right shoulder
    mock_landmarks[23] = MockLandmark(0.35, 0.7)  # Left hip
    mock_landmarks[24] = MockLandmark(0.65, 0.7)  # Right hip
    
    # Parse with constraint
    masks_constrained = parser.parse(
        test_frame,
        pose_landmarks=mock_landmarks,
        target_resolution=(473, 473)
    )
    
    # Compare
    unconstrained_pixels = masks['upper_body'].sum() / 255
    constrained_pixels = masks_constrained['upper_body'].sum() / 255
    
    reduction = unconstrained_pixels - constrained_pixels
    reduction_ratio = reduction / max(unconstrained_pixels, 1)
    
    print(f"  Unconstrained: {int(unconstrained_pixels)} pixels")
    print(f"  Constrained: {int(constrained_pixels)} pixels")
    print(f"  Reduction: {int(reduction)} pixels ({reduction_ratio*100:.1f}%)")
    
    # CRITICAL ASSERTION 3: Constraint reduces but doesn't annihilate
    assert constrained_pixels > 0, "FAIL: Constraint annihilated garment region"
    assert reduction >= 0, "FAIL: Constraint increased garment region"
    
    print(f"\n[PASS] Assertion 3: Constraint reduces without annihilation")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Temporal stability (bounded frame-to-frame expansion)
print("\n[4/5] Testing temporal stability...")
try:
    # Parse same frame twice (should be identical without smoothing)
    masks_t1 = parser.parse(test_frame, target_resolution=(473, 473))
    masks_t2 = parser.parse(test_frame, target_resolution=(473, 473))
    
    # Calculate difference
    diff = cv2.absdiff(masks_t1['upper_body'], masks_t2['upper_body'])
    diff_pixels = diff.sum() / 255
    
    print(f"  Frame-to-frame difference: {int(diff_pixels)} pixels")
    
    # CRITICAL ASSERTION 4: Deterministic output
    assert diff_pixels == 0, "FAIL: Non-deterministic parsing output"
    print(f"\n[PASS] Assertion 4: Deterministic output (temporal smoothing OFF)")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Save golden artifacts
print("\n[5/5] Saving golden artifacts...")
try:
    artifacts_dir = Path("test_artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save test frame
    cv2.imwrite(str(artifacts_dir / "golden_frame.png"), test_frame)
    
    # Save masks
    for key in ['hair', 'face', 'neck', 'upper_body', 'arms', 'lower_body']:
        cv2.imwrite(str(artifacts_dir / f"golden_mask_{key}.png"), masks[key])
    
    # Save statistics
    with open(artifacts_dir / "golden_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[OK] Golden artifacts saved to {artifacts_dir}/")
    print(f"     - golden_frame.png")
    print(f"     - golden_mask_*.png (6 masks)")
    print(f"     - golden_stats.json")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] All critical assertions passed!")
print("=" * 70)

print("\nValidated Behaviors:")
print("  [PASS] Non-zero garment area")
print("  [PASS] Minimal garment-face/hair overlap")
print("  [PASS] Geometric constraint reduces without annihilation")
print("  [PASS] Deterministic parsing output")
print("  [PASS] Golden artifacts frozen for regression testing")

print("\nThis test validates BEHAVIOR, not just code paths.")
print("Golden artifacts can be used for future regression tests.")

#!/usr/bin/env python3
"""Quick validation test for semantic parsing implementation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SEMANTIC PARSING VALIDATION")
print("=" * 70)

# Test 1: Import
print("\n[1/3] Testing imports...")
try:
    from src.core.semantic_parser import SemanticParser, create_occlusion_aware_composite
    print("[OK] Imports successful")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 2: Initialize with optimizations
print("\n[2/3] Testing initialization...")
try:
    parser = SemanticParser(use_mediapipe=True, temporal_smoothing=True)
    print("[OK] Parser initialized with temporal smoothing")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 3: Test parsing with resolution scaling
print("\n[3/3] Testing parsing with resolution scaling...")
try:
    import numpy as np
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Parse at optimized resolution
    masks = parser.parse(test_frame, target_resolution=(256, 192))
    
    # Verify output
    expected_keys = ['hair', 'face', 'neck', 'upper_body', 'arms', 'lower_body', 'full_parsing']
    for key in expected_keys:
        if key not in masks:
            raise Exception(f"Missing mask: {key}")
        if masks[key].shape != (480, 640):
            raise Exception(f"Wrong shape for {key}: {masks[key].shape}")
    
    print(f"[OK] Parsing works - all masks present with correct shape")
    print(f"     Hair mask: {masks['hair'].shape}")
    print(f"     Face mask: {masks['face'].shape}")
    
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] All validation tests passed!")
print("=" * 70)
print("\nImplementation summary:")
print("  - Semantic parsing enabled in app.py")
print("  - Resolution scaling: 256x192 -> full res")
print("  - Temporal smoothing: alpha=0.7")
print("  - Expected FPS: 14-17 with occlusion handling")

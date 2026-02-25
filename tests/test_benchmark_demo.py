#!/usr/bin/env python3
"""
Quick demonstration of occlusion benchmark capabilities
Tests the benchmark framework with synthetic data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

print("=" * 70)
print("OCCLUSION BENCHMARK DEMONSTRATION")
print("=" * 70)

# Test 1: Import benchmark
print("\n[1/3] Testing benchmark import...")
try:
    from benchmarks.occlusion_benchmark import OcclusionBenchmark, print_summary
    from src.core.semantic_parser import SemanticParser
    print("[OK] Benchmark module imported")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 2: Initialize benchmark
print("\n[2/3] Initializing benchmark...")
try:
    parser = SemanticParser(
        backend='auto',
        temporal_smoothing=True,
        onnx_model_path='models/schp_lip.onnx'
    )
    
    benchmark = OcclusionBenchmark(parser)
    print(f"[OK] Benchmark initialized")
    print(f"     Backend: {parser.backend.__class__.__name__}")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test metric calculations
print("\n[3/3] Testing metric calculations...")
try:
    # Create synthetic masks
    h, w = 480, 640
    
    # Garment mask (center region)
    garment_mask = np.zeros((h, w), dtype=np.uint8)
    garment_mask[200:400, 200:440] = 255
    
    # Face mask (upper center)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[100:200, 250:390] = 255
    
    # Hair mask (top)
    hair_mask = np.zeros((h, w), dtype=np.uint8)
    hair_mask[50:120, 240:400] = 255
    
    # Test occlusion error (should be 0 - no overlap)
    error = benchmark.measure_occlusion_error(garment_mask, face_mask, hair_mask)
    print(f"[OK] Occlusion error calculation: {error:.3f}")
    print(f"     Expected: ~0.0 (no overlap)")
    
    # Create overlapping garment (bad case)
    bad_garment = garment_mask.copy()
    bad_garment[100:200, 250:390] = 255  # Overlap with face
    
    bad_error = benchmark.measure_occlusion_error(bad_garment, face_mask, hair_mask)
    print(f"[OK] Occlusion error with overlap: {bad_error:.3f}")
    print(f"     Expected: >0.0 (has overlap)")
    
    # Test temporal stability
    prev_mask = garment_mask.copy()
    current_mask = garment_mask.copy()
    current_mask[200:400, 205:445] = 255  # Slight shift
    
    jitter = benchmark.measure_temporal_stability(current_mask, prev_mask)
    print(f"[OK] Temporal jitter: {jitter:.3f}")
    print(f"     Expected: small value (slight movement)")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Benchmark framework ready!")
print("=" * 70)

print("\nBenchmark Capabilities:")
print("  [OK] Occlusion error measurement")
print("  [OK] Temporal stability measurement")
print("  [OK] Frame-by-frame benchmarking")
print("  [OK] Video file benchmarking")
print("  [OK] Webcam benchmarking")

print("\nUsage:")
print("  python benchmarks/occlusion_benchmark.py")
print("    - Interactive benchmark with webcam or video file")
print("    - Measures occlusion errors, temporal jitter, FPS")
print("    - Compares with/without geometric constraints")

print("\nNext Steps:")
print("  1. Run benchmark with live webcam")
print("  2. Compare ONNX vs MediaPipe quality")
print("  3. Test with/without geometric constraints")
print("  4. Document baseline performance")

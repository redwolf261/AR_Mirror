#!/usr/bin/env python3
"""Test backend-agnostic semantic parser refactoring"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("BACKEND-AGNOSTIC SEMANTIC PARSER TEST")
print("=" * 70)

# Test 1: Import new backend system
print("\n[1/5] Testing backend imports...")
try:
    from src.core.parsing_backends import ParsingBackend, MediaPipeBackend, ONNXParsingBackend
    from src.core.semantic_parser import SemanticParser
    print("[OK] Backend abstraction imports successful")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize with MediaPipe backend
print("\n[2/5] Testing MediaPipe backend...")
try:
    parser_mp = SemanticParser(backend='mediapipe', temporal_smoothing=True)
    backend_name = parser_mp.backend.__class__.__name__
    print(f"[OK] MediaPipe backend initialized: {backend_name}")
    assert backend_name == 'MediaPipeBackend', f"Expected MediaPipeBackend, got {backend_name}"
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test auto-select (should fallback to MediaPipe since no ONNX model)
print("\n[3/5] Testing auto-select backend...")
try:
    parser_auto = SemanticParser(
        backend='auto',
        temporal_smoothing=True,
        onnx_model_path='models/schp_lip.onnx'  # Doesn't exist yet
    )
    backend_name = parser_auto.backend.__class__.__name__
    print(f"[OK] Auto-selected backend: {backend_name}")
    # Should fallback to MediaPipe since ONNX model doesn't exist
    assert backend_name == 'MediaPipeBackend', f"Expected MediaPipeBackend fallback, got {backend_name}"
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test parsing with backend
print("\n[4/5] Testing parsing with backend abstraction...")
try:
    import numpy as np
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Parse with MediaPipe backend
    masks = parser_mp.parse(test_frame, target_resolution=(256, 192))
    
    # Verify all masks present
    expected_keys = ['hair', 'face', 'neck', 'upper_body', 'arms', 'lower_body', 'full_parsing']
    for key in expected_keys:
        if key not in masks:
            raise Exception(f"Missing mask: {key}")
        if masks[key].shape != (480, 640):
            raise Exception(f"Wrong shape for {key}: {masks[key].shape}")
    
    print(f"[OK] Parsing works with backend abstraction")
    print(f"     All {len(expected_keys)} masks generated correctly")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify backward compatibility
print("\n[5/5] Testing backward compatibility...")
try:
    # Old API should still work (defaults to mediapipe)
    parser_legacy = SemanticParser()  # No arguments
    backend_name = parser_legacy.backend.__class__.__name__
    print(f"[OK] Backward compatible - defaults to {backend_name}")
    assert backend_name == 'MediaPipeBackend'
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Backend abstraction refactoring complete!")
print("=" * 70)
print("\nArchitecture summary:")
print("  - ParsingBackend: Abstract interface")
print("  - MediaPipeBackend: Current implementation (working)")
print("  - ONNXParsingBackend: Future implementation (stub ready)")
print("  - Auto-select: Tries ONNX first, falls back to MediaPipe")
print("\nNext steps:")
print("  1. Download/convert SCHP model to ONNX")
print("  2. Place in models/schp_lip.onnx")
print("  3. System will auto-select ONNX backend")
print("  4. A/B test performance vs MediaPipe")

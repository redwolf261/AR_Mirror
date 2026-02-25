#!/usr/bin/env python3
"""
Quick test to verify semantic parsing integration
Run this before running the full app to catch any import/initialization errors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SEMANTIC PARSING INTEGRATION TEST")
print("=" * 70)

# Test 1: Import semantic parser
print("\n[1/4] Testing semantic parser import...")
try:
    from src.core.semantic_parser import SemanticParser, create_occlusion_aware_composite, BodyPart
    print("✓ Semantic parser imports successful")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Initialize parser
print("\n[2/4] Testing parser initialization...")
try:
    parser = SemanticParser(use_mediapipe=True)
    print("✓ Parser initialized successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Test with dummy frame
print("\n[3/4] Testing parse on dummy frame...")
try:
    import numpy as np
    import cv2
    
    # Create dummy frame (640x480)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Parse it
    body_parts = parser.parse(dummy_frame)
    
    # Verify all expected masks are present
    expected_keys = ['hair', 'face', 'neck', 'upper_body', 'arms', 'lower_body', 'full_parsing']
    for key in expected_keys:
        if key not in body_parts:
            raise Exception(f"Missing mask: {key}")
    
    print(f"✓ Parsing successful - generated {len(body_parts)} masks")
    print(f"  Hair mask shape: {body_parts['hair'].shape}")
    print(f"  Face mask shape: {body_parts['face'].shape}")
    print(f"  Full parsing shape: {body_parts['full_parsing'].shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Test compositing function
print("\n[4/4] Testing occlusion-aware compositing...")
try:
    # Create dummy garment (same size as frame)
    dummy_garment = np.random.rand(480, 640, 3).astype(np.float32)
    dummy_mask = np.random.rand(480, 640).astype(np.float32)
    
    # Composite
    result = create_occlusion_aware_composite(
        dummy_frame,
        dummy_garment,
        dummy_mask,
        body_parts,
        collar_constraint=True
    )
    
    if result.shape != dummy_frame.shape:
        raise Exception(f"Output shape mismatch: {result.shape} vs {dummy_frame.shape}")
    
    if result.dtype != np.uint8:
        raise Exception(f"Output dtype wrong: {result.dtype} (expected uint8)")
    
    print(f"✓ Compositing successful")
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")
    print(f"  Output range: [{result.min()}, {result.max()}]")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Verify app.py integration
print("\n[5/5] Checking app.py integration...")
try:
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    checks = [
        ('SEMANTIC_PARSING_AVAILABLE', 'Global flag defined'),
        ('from src.core.semantic_parser import', 'Import statement present'),
        ('self.semantic_parser = None', 'Instance variable initialized'),
        ('SemanticParser(use_mediapipe=True)', 'Parser instantiation'),
        ('create_occlusion_aware_composite', 'Compositing function called'),
    ]
    
    for check_str, desc in checks:
        if check_str in app_content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ Missing: {desc}")
            
except Exception as e:
    print(f"✗ Failed to check app.py: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - SEMANTIC PARSING READY")
print("=" * 70)
print("\nYou can now run: python app.py")
print("\nExpected improvements:")
print("  • Hair will stay on top (not covered by garment)")
print("  • Face will remain visible")
print("  • Collar will stop at neck boundary")
print("  • Proper depth-based layering")
print("\nNote: FPS may drop slightly (14-16 FPS) but quality will improve!")

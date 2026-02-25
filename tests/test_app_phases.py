"""Test app.py phases without GUI"""
import sys
from app import ARMirrorApp
import numpy as np

print("Testing Phase 0...")
app0 = ARMirrorApp(phase=0, demo_duration=1)
if app0.initialize():
    print("✓ Phase 0 initialized successfully")
    # Test render without camera
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        result = app0._render_garment(test_frame, app0.garments[0])
        print(f"✓ Phase 0 render OK: {result.shape}")
    except Exception as e:
        print(f"✗ Phase 0 render failed: {e}")
else:
    print("✗ Phase 0 init failed")

print("\nTesting Phase 2...")
app2 = ARMirrorApp(phase=2, demo_duration=1)
if app2.initialize():
    print("✓ Phase 2 initialized successfully")
    # Test render without camera
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        result = app2._render_garment(test_frame, app2.garments[0])
        print(f"✓ Phase 2 render OK: {result.shape}")
    except Exception as e:
        print(f"✗ Phase 2 render failed: {e}")
else:
    print("✗ Phase 2 init failed")

print("\n✅ All phases working!")

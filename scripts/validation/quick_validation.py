#!/usr/bin/env python3
"""Quick validation of all AR Mirror phases"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("AR MIRROR - PHASE VALIDATION")
print("=" * 70)

# Test Phase 2
print("\n[1/3] Testing Phase 2 (Neural Warping)...")
try:
    from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
    pipeline = Phase2NeuralPipeline(device='auto', enable_tom=False)
    stats = pipeline.get_statistics()
    print(f"✓ Phase 2 Ready - Device: {stats['device']}")
except Exception as e:
    print(f"✗ Phase 2 Failed: {e}")

# Test Phase 2 components
print("\n[2/3] Testing Phase 2 Components...")
try:
    from src.core.live_pose_converter import LivePoseConverter, LiveBodySegmenter
    import numpy as np
    import cv2
    
    # Test pose converter
    converter = LivePoseConverter(heatmap_size=(256, 192))
    dummy_landmarks = np.random.rand(33, 3)  # 33 MediaPipe landmarks
    heatmaps = converter.landmarks_to_heatmaps(dummy_landmarks)  # type: ignore
    assert heatmaps.shape == (18, 256, 192), f"Wrong shape: {heatmaps.shape}"
    print(f"✓ Pose Converter: {heatmaps.shape}")
    
    # Test body segmenter
    segmenter = LiveBodySegmenter()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask, _ = segmenter.segment(dummy_frame)
    assert mask.shape == (256, 192), f"Wrong shape: {mask.shape}"
    print(f"✓ Body Segmenter: {mask.shape}")
    
except Exception as e:
    print(f"✗ Components Failed: {e}")

# Test App
print("\n[3/3] Testing App Initialization...")
try:
    from app import ARMirrorApp
    
    for phase in [2, 0]:
        app = ARMirrorApp(phase=phase)
        phase_name = {2: "Neural", 0: "Blending"}[phase]
        print(f"✓ Phase {phase} ({phase_name}): Created")
        
except Exception as e:
    print(f"✗ App Failed: {e}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print("\nReady to run:")
print("  python app.py --phase 2    # Neural warping (21+ FPS)")
print("  python app.py --phase 0    # Simple blending (30+ FPS)")
print("=" * 70)

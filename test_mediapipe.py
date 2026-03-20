#!/usr/bin/env python3
"""
Simple body detection test without camera
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def test_mediapipe_detection():
    print("Testing MediaPipe detection directly...")

    # Create a test frame (solid color with some noise)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Initialize MediaPipe detector like in body_aware_fitter.py
    model_path = 'pose_landmarker_lite.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)

    print("Detector created successfully")

    # Convert BGR to RGB (like in the body fitter)
    rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Use timestamp like in body_fitter
    start_time_ns = time.monotonic_ns()
    elapsed_ns = time.monotonic_ns() - start_time_ns
    timestamp_ms = elapsed_ns // 1_000_000

    print(f"Testing with timestamp: {timestamp_ms}")

    try:
        # Attempt detection
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        print(f"Detection successful!")
        print(f"Pose landmarks found: {len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0}")
        print(f"Segmentation masks: {len(detection_result.segmentation_masks) if detection_result.segmentation_masks else 0}")

        if not detection_result.pose_landmarks:
            print("No pose detected (expected for random noise frame)")
        else:
            print("Pose detected in noise frame (unexpected!)")

    except Exception as e:
        print(f"Detection FAILED with error: {e}")
        import traceback
        traceback.print_exc()

    # Test with incrementing timestamps
    print("\nTesting multiple frames with proper timestamps...")
    for i in range(5):
        elapsed_ns = time.monotonic_ns() - start_time_ns
        timestamp_ms = elapsed_ns // 1_000_000

        try:
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            print(f"Frame {i}: timestamp={timestamp_ms}, poses={len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0}")
        except Exception as e:
            print(f"Frame {i}: FAILED with error: {e}")
            break

        time.sleep(0.033)  # Simulate ~30fps

    detector.close()
    print("Test completed")

if __name__ == "__main__":
    test_mediapipe_detection()
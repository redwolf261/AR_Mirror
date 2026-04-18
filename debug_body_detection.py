#!/usr/bin/env python3
"""
Debug body detection to find out why measurements aren't working
"""

import cv2
import numpy as np
from src.core.body_aware_fitter import BodyAwareGarmentFitter

def test_body_detection():
    print("Testing body detection...")

    # Initialize body fitter
    try:
        body_fitter = BodyAwareGarmentFitter()
        print("OK: Body fitter initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize body fitter: {e}")
        import traceback
        traceback.print_exc()
        return

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("OK: Camera opened")
    print("Press 'q' to quit, any other key to test detection")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot read frame")
            break

        # Show frame
        cv2.imshow('Debug Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key != 255:  # Any key pressed
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            print(f"Frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")

            # Test body measurements
            try:
                print("Calling extract_body_measurements...")
                measurements = body_fitter.extract_body_measurements(frame)
                print(f"Result: {measurements is not None}")

                if measurements:
                    print(f"Measurements keys: {list(measurements.keys())}")
                    if 'shoulder_width' in measurements:
                        print(f"Shoulder width: {measurements['shoulder_width']:.1f}px")
                    if 'torso_height' in measurements:
                        print(f"Torso height: {measurements['torso_height']:.1f}px")
                    if 'size_recommendation' in measurements:
                        print(f"Size recommendation: {measurements['size_recommendation']}")
                else:
                    print("No measurements extracted")
                    # Get diagnostics
                    diag = body_fitter.get_diagnostics()
                    print(f"Detection status: {diag['status']}")
                    print(f"Confidence: {diag['confidence']:.2f}")
                    print(f"Consecutive failures: {diag['consecutive_failures']}")
                    print(f"Success rate: {diag['success_rate']:.1f}%")

            except Exception as e:
                print(f"Exception in extract_body_measurements: {e}")
                import traceback
                traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    test_body_detection()
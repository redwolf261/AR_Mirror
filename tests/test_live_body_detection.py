#!/usr/bin/env python3
"""
Test Live Body Detection with MediaPipe (New API)
Focus: Accurately detect body landmarks using MediaPipe tasks API
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on image"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def main():
    print("="*70)
    print("LIVE BODY DETECTION TEST (MediaPipe Tasks API)".center(70))
    print("="*70)
    print()
    
    # Initialize MediaPipe Pose Landmarker
    print("[1/3] Initializing MediaPipe Pose Landmarker...")
    
    # Download model if needed
    model_path = 'pose_landmarker_lite.task'
    import urllib.request
    import os
    
    if not os.path.exists(model_path):
        print(f"  Downloading pose model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)
        print(f"  [OK] Model downloaded")
    
    # Create PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    print("  [OK] MediaPipe Pose Landmarker initialized")
    
    # Open webcam
    print("[2/3] Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("  ✗ Cannot open camera")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [OK] Camera open at {width}x{height}")
    
    print("[3/3] Starting live detection...")
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to toggle segmentation mask")
    print()
    
    show_segmentation = True
    frame_count = 0
    fps_times = []
    timestamp_ms = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect pose
            timestamp_ms += 33  # Approximate 30 FPS
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            # Create output frame
            output = frame.copy()
            
            if detection_result.pose_landmarks:
                # Draw landmarks
                annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
                output = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw body segmentation
                if show_segmentation and detection_result.segmentation_masks:
                    mask = detection_result.segmentation_masks[0].numpy_view()
                    
                    # Create green overlay for body
                    body_overlay = np.zeros_like(frame)
                    body_overlay[:, :] = (0, 255, 0)  # Green
                    
                    # Blend
                    mask_threshold = 0.5
                    body_mask = (mask > mask_threshold).astype(np.uint8)
                    body_mask_3channel = np.stack([body_mask] * 3, axis=-1)
                    output = cv2.addWeighted(output, 1.0, body_overlay * body_mask_3channel, 0.3, 0)
                
                # Get key body points
                landmarks = detection_result.pose_landmarks[0]
                h, w = frame.shape[:2]
                
                # Shoulders (landmarks 11, 12)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                # Hips (landmarks 23, 24)
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                # Calculate body dimensions
                shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
                torso_height = abs(left_shoulder.y - left_hip.y) * h
                
                # Draw bounding box for torso
                torso_x1 = int(min(left_shoulder.x, left_hip.x) * w)
                torso_y1 = int(left_shoulder.y * h)
                torso_x2 = int(max(right_shoulder.x, right_hip.x) * w)
                torso_y2 = int(left_hip.y * h)
                
                cv2.rectangle(output, (torso_x1, torso_y1), (torso_x2, torso_y2), (255, 0, 0), 3)
                
                # Display measurements
                cv2.putText(output, f"Shoulder Width: {shoulder_width:.0f}px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(output, f"Torso Height: {torso_height:.0f}px", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(output, "BODY DETECTED", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(output, "NO BODY DETECTED - Step into frame", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps_times.append(frame_time)
            if len(fps_times) > 30:
                fps_times.pop(0)
            
            avg_fps = 1.0 / np.mean(fps_times) if fps_times else 0
            
            # Display FPS and status
            cv2.putText(output, f"FPS: {avg_fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output, f"Segmentation: {'ON' if show_segmentation else 'OFF'}", (w - 250, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Show frame
            cv2.imshow("Body Detection Test", output)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_segmentation = not show_segmentation
                print(f"Segmentation: {'ON' if show_segmentation else 'OFF'}")
            
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | FPS: {avg_fps:.1f} | Body: {'YES' if detection_result.pose_landmarks else 'NO'}")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        
        print()
        print("="*70)
        print("TEST COMPLETE".center(70))
        print("="*70)
        print(f"\nProcessed {frame_count} frames")
        if fps_times:
            print(f"Average FPS: {1.0/np.mean(fps_times):.1f}")
        print()

if __name__ == "__main__":
    main()

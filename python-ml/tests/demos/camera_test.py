"""
Quick camera test for MediaPipe pose detection
Tests if PoseDetector works with webcam
"""
import cv2
import numpy as np
from sizing_pipeline import PoseDetector

def test_camera():
    print("Initializing camera and pose detector...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = PoseDetector()
    print("Pose detector initialized successfully!")
    print(f"Using legacy API: {detector.legacy_mode}")
    print("\nPress 'q' to quit, 's' to save snapshot")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        frame_count += 1
        
        # Detect pose
        landmarks = detector.detect(frame)
        
        # Draw status
        if landmarks:
            detection_count += 1
            status = f"POSE DETECTED (Frame {frame_count})"
            color = (0, 255, 0)
            
            # Draw key landmarks
            h, w = frame.shape[:2]
            for idx, lm_data in landmarks.items():
                x = int(lm_data['x'] * w)
                y = int(lm_data['y'] * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            status = f"No pose detected (Frame {frame_count})"
            color = (0, 0, 255)
        
        # Display info
        success_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Success: {success_rate:.1f}% ({detection_count}/{frame_count})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Camera Test - Pose Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections: {detection_count}")
    print(f"  Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    test_camera()

"""
Body Detection Diagnostic Tool
Shows what MediaPipe is detecting and helps fix positioning issues
"""
import cv2
import numpy as np
import time
import sys
import os

# Import MediaPipe components carefully to avoid sounddevice issues
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
except Exception as e:
    print(f"MediaPipe import error: {e}")
    print("Trying alternative import method...")
    # Try importing just what we need
    from mediapipe import tasks
    python = tasks.python
    vision = tasks.python.vision
    import mediapipe as mp

def main():
    print("\n" + "="*70)
    print("BODY DETECTION DIAGNOSTIC".center(70))
    print("="*70)
    print()
    print("This tool shows what the system is detecting.")
    print()
    print("Common Issues:")
    print("  - Head/shoulders cut off → Stand further back")
    print("  - Camera pointed down → Angle camera at chest/face")
    print("  - Body at bottom of frame → Adjust camera height")
    print()
    print("Press 'q' to quit")
    print("="*70)
    print()
    
    # Initialize MediaPipe
    print("[1/2] Loading MediaPipe Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)
    print("  ✓ Loaded")
    
    # Open camera
    print("[2/2] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Could not open camera!")
        return
    print("  ✓ Camera ready")
    print()
    
    timestamp_ms = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp_ms += 33  # ~30 FPS
        
        h, w = frame.shape[:2]
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Detection error: {e}")
            continue
        
        # Create output frame
        output = frame.copy()
        
        # === Draw body segmentation ===
        if result.segmentation_masks:
            mask = result.segmentation_masks[0].numpy_view()
            # Resize mask to match frame dimensions if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Ensure 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Find mask bounds
            ys, xs = np.where(mask > 0)
            if len(ys) > 100:
                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())
                
                # Draw green tint over detected body
                green_overlay = np.zeros_like(output)
                green_overlay[:, :] = (0, 255, 0)
                mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
                output = (output * (1 - mask_3ch * 0.3) + green_overlay * mask_3ch * 0.3).astype(np.uint8)
                
                # Draw bounding box
                cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(output, f"Body: y={y_min}-{y_max} (height={y_max-y_min}px)", 
                           (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # === Draw landmarks ===
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            # Key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            nose = landmarks[0]
            
            # Convert to pixel coordinates
            def to_px(lm):
                return (int(lm.x * w), int(lm.y * h))
            
            # Draw landmarks
            for idx, lm in enumerate(landmarks):
                x, y = to_px(lm)
                if idx in [0, 11, 12, 23, 24]:  # Key points
                    cv2.circle(output, (x, y), 8, (255, 0, 255), -1)
                    labels = {
                        0: "NOSE",
                        11: "L_SHOULDER",
                        12: "R_SHOULDER", 
                        23: "L_HIP",
                        24: "R_HIP"
                    }
                    cv2.putText(output, labels[idx], (x + 10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                else:
                    cv2.circle(output, (x, y), 3, (0, 255, 255), -1)
            
            # Draw skeleton connections
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                (11, 23), (12, 24), (23, 24),  # Torso
                (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
            ]
            for start_idx, end_idx in connections:
                start = to_px(landmarks[start_idx])
                end = to_px(landmarks[end_idx])
                cv2.line(output, start, end, (0, 255, 255), 2)
            
            # === Calculate and show detected torso box ===
            sh_y = min(left_shoulder.y, right_shoulder.y) * h
            hip_y = max(left_hip.y, right_hip.y) * h
            left_x = min(left_shoulder.x, left_hip.x) * w
            right_x = max(right_shoulder.x, right_hip.x) * w
            
            torso_top = int(sh_y)
            torso_bottom = int(hip_y)
            torso_left = int(left_x)
            torso_right = int(right_x)
            
            # Draw detected torso box
            cv2.rectangle(output, (torso_left, torso_top), (torso_right, torso_bottom), 
                         (0, 180, 255), 3)
            cv2.putText(output, "GARMENT BOX", (torso_left, torso_top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            
            # === Diagnostic Messages ===
            # Check if head is in frame
            nose_y = nose.y * h
            if nose_y < 50:
                cv2.putText(output, "WARNING: HEAD CUT OFF! Stand further back!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if shoulders are too low (person too close)
            if sh_y > h * 0.6:
                cv2.putText(output, "WARNING: Shoulders too low! Step back from camera!",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if torso is too small
            torso_height = torso_bottom - torso_top
            if torso_height < h * 0.20:
                cv2.putText(output, "WARNING: Torso too small! Move closer or adjust camera!",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show measurements
            cv2.putText(output, f"Shoulder Y: {int(sh_y)} (should be < {int(h*0.4)})", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"Torso Height: {int(torso_height)}px (should be > {int(h*0.2)})", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"Frame Size: {w}x{h}", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Ideal positioning guide
            ideal_sh_y = int(h * 0.25)
            ideal_hip_y = int(h * 0.65)
            cv2.line(output, (0, ideal_sh_y), (w, ideal_sh_y), (255, 255, 0), 1)
            cv2.line(output, (0, ideal_hip_y), (w, ideal_hip_y), (255, 255, 0), 1)
            cv2.putText(output, "IDEAL SHOULDER HEIGHT", (w - 250, ideal_sh_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(output, "IDEAL HIP HEIGHT", (w - 250, ideal_hip_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        else:
            # No pose detected
            cv2.putText(output, "NO PERSON DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(output, "Step into frame and face camera", (w//2 - 180, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Body Detection Diagnostic', output)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    main()

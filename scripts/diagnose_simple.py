"""
Body Detection Diagnostic Tool - Simple Version
Shows what the system is detecting and helps fix positioning issues
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.body_aware_fitter import BodyAwareGarmentFitter

def main():
    print("\n" + "="*70)
    print("BODY DETECTION DIAGNOSTIC".center(70))
    print("="*70)
    print()
    print("This tool shows what the system is detecting.")
    print()
    print("Common Issues:")
    print("  - Head/shoulders cut off -> Stand further back")
    print("  - Camera pointed down -> Angle camera at chest/face")
    print("  - Body at bottom of frame -> Adjust camera height")
    print()
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*70)
    print()
    
    # Initialize body fitter (same as main app)
    print("[1/2] Loading body detection...")
    try:
        fitter = BodyAwareGarmentFitter()
        print("  [OK] Loaded")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    
    # Open camera
    print("[2/2] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Could not open camera!")
        return
    print("  [OK] Camera ready")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect body measurements
        try:
            measurements = fitter.extract_body_measurements(frame)
        except Exception as e:
            measurements = None
            if frame_count % 30 == 0:
                print(f"Detection error: {e}")
        
        # Create output frame
        output = frame.copy()
        
        if measurements is not None:
            # Draw debug overlay (shows skeleton, torso box, measurements)
            output = fitter.draw_debug_overlay(
                output, 
                measurements, 
                show_box=True,
                show_measurements=True,
                show_skeleton=True
            )
            
            # Get key values
            torso_x1, torso_y1, torso_x2, torso_y2 = measurements['torso_box']
            landmarks = measurements.get('landmarks')
            body_mask = measurements.get('body_mask')
            
            # === Additional diagnostic overlays ===
            
            # 1. Show body segmentation mask
            if body_mask is not None:
                # Convert to 2D if needed
                if len(body_mask.shape) == 3:
                    bm_2d = body_mask[:, :, 0]
                else:
                    bm_2d = body_mask
                
                # Find mask bounds
                if bm_2d.shape[:2] != (h, w):
                    bm = cv2.resize(bm_2d.astype(np.float32), (w, h), 
                                   interpolation=cv2.INTER_NEAREST)
                    bm = (bm > 0.5).astype(np.uint8)
                else:
                    bm = bm_2d
                
                ys, xs = np.where(bm > 0)
                if len(ys) > 100:
                    y_min, y_max = int(ys.min()), int(ys.max())
                    x_min, x_max = int(xs.min()), int(xs.max())
                    
                    # Draw body mask bounds
                    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    cv2.putText(output, f"Mask: {y_min}-{y_max} (h={y_max-y_min})", 
                               (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (0, 255, 0), 1)
            
            # 2. Ideal positioning guides
            ideal_sh_y = int(h * 0.25)  # Shoulders should be at 25% from top
            ideal_hip_y = int(h * 0.65)  # Hips should be at 65% from top
            
            cv2.line(output, (0, ideal_sh_y), (w, ideal_sh_y), (255, 255, 0), 1)
            cv2.line(output, (0, ideal_hip_y), (w, ideal_hip_y), (255, 255, 0), 1)
            
            cv2.putText(output, "IDEAL SHOULDER", (w - 180, ideal_sh_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(output, "IDEAL HIP", (w - 180, ideal_hip_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 3. Diagnostic warnings
            warnings = []
            
            # Check if shoulders are too low
            shoulder_y = torso_y1
            if shoulder_y > h * 0.5:
                warnings.append("! SHOULDERS TOO LOW! Step back from camera")
            elif shoulder_y > h * 0.4:
                warnings.append("! Could be better - step back a bit")
            
            # Check torso height
            torso_height = torso_y2 - torso_y1
            if torso_height < h * 0.15:
                warnings.append("! TORSO TOO SMALL! Move closer or adjust camera")
            
            # Check if nose is visible
            if landmarks is not None and len(landmarks) > 0:
                nose = landmarks[0]
                nose_y = nose.y * h
                if nose_y < 30:
                    warnings.append("! HEAD CUT OFF! Stand further back")
                elif nose_y > h * 0.5:
                    warnings.append("! Camera too low! Raise camera to chest level")
            
            # Display warnings
            for i, warning in enumerate(warnings):
                y_pos = 30 + (i * 30)
                # Red background for visibility
                cv2.rectangle(output, (5, y_pos - 20), (w - 5, y_pos + 5), 
                             (0, 0, 255), -1)
                cv2.putText(output, warning, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 4. Status info at bottom
            status_y = h - 100
            cv2.rectangle(output, (0, status_y), (w, h), (0, 0, 0), -1)
            
            cv2.putText(output, f"Frame: {w}x{h}", 
                       (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"Shoulder Y: {shoulder_y}px (ideal: {ideal_sh_y})", 
                       (10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"Torso Height: {torso_height}px (need > {int(h*0.15)})", 
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if len(warnings) == 0:
                cv2.putText(output, "✓ POSITIONING GOOD! Ready for try-on", 
                           (10, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(output, f"✗ {len(warnings)} issue(s) - see warnings above", 
                           (10, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        else:
            # No detection
            cv2.putText(output, "NO PERSON DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(output, "Step into frame and face camera", (w//2 - 180, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show ideal guides anyway
            ideal_sh_y = int(h * 0.25)
            ideal_hip_y = int(h * 0.65)
            cv2.line(output, (0, ideal_sh_y), (w, ideal_sh_y), (255, 255, 0), 1)
            cv2.line(output, (0, ideal_hip_y), (w, ideal_hip_y), (255, 255, 0), 1)
            cv2.putText(output, "← Shoulders should be here", (10, ideal_sh_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(output, "← Hips should be here", (10, ideal_hip_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show frame
        cv2.imshow('Body Detection Diagnostic (Press Q to quit, S to save)', output)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"diagnostic_{frame_count}.png"
            cv2.imwrite(filename, output)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDiagnostic complete!")
    
    if measurements is not None:
        print("\nFinal Measurements:")
        print(f"  Shoulder Width: {measurements['shoulder_width']:.0f}px")
        print(f"  Torso Height: {measurements['torso_height']:.0f}px")
        print(f"  Torso Box: {measurements['torso_box']}")

if __name__ == "__main__":
    main()

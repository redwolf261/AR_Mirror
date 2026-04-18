"""
Adaptive AR Try-On Demo
Shows detected body regions and available product categories in real-time
"""
import cv2
from sizing_pipeline import SizingPipeline, BodyRegion, ProductCategory

def main():
    log_dir = "logs"
    
    # Start with adaptive mode (no garment database needed)
    pipeline = SizingPipeline("garment_database.json", log_dir, adaptive_mode=True)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=" * 70)
    print("ADAPTIVE AR TRY-ON SYSTEM")
    print("=" * 70)
    print("\nHow it works:")
    print("  • Show your FACE → Try hats, sunglasses, earrings")
    print("  • Show your HANDS → Try watches, bracelets, rings")
    print("  • Show UPPER BODY → Try shirts, jackets, tops")
    print("  • Show LOWER BODY → Try pants, shorts, skirts")
    print("  • Show FULL BODY → Try dresses, suits, jumpsuits")
    print("\nThe system adapts to whatever body parts are visible!")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save snapshot")
    print("=" * 70)
    
    frame_count = 0
    last_detected = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break
        
        frame_count += 1
        
        # Process frame adaptively
        output_frame, detection = pipeline.process_frame_adaptive(frame)
        
        # Print when detected regions change
        if detection and detection.detected_regions != last_detected:
            print(f"\n[Frame {frame_count}]")
            print(f"  Detected: {', '.join(r.value for r in detection.detected_regions)}")
            print(f"  Can try: {', '.join(c.value for c in detection.available_categories[:5])}")
            if len(detection.available_categories) > 5:
                print(f"           ...and {len(detection.available_categories) - 5} more")
            last_detected = detection.detected_regions.copy() if detection else set()
        
        cv2.imshow('Adaptive AR Try-On', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and detection:
            filename = f"snapshot_{frame_count}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"  Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("Session ended")
    print(f"Total frames processed: {frame_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Quick diagnostic: Check torso measurement computation
"""
import cv2
from sizing_pipeline import SizingPipeline

pipeline = SizingPipeline("garment_database.json", "logs")
pipeline.set_garment("SKU-001")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Stand in front of camera, showing full body (head to hips)")
print("Press SPACE to check measurements, Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output_frame, result = pipeline.process_frame(frame)
    
    if result:
        # Debug info
        print("\n" + "="*60)
        print(f"Shoulder: {result.measurements.shoulder_width_cm:.1f}cm")
        print(f"Chest: {result.measurements.chest_width_cm:.1f}cm")
        print(f"Torso: {result.measurements.torso_length_cm:.1f}cm")
        print(f"Confidence: {result.measurements.confidence:.2f}")
        
        # Note: Internal detector attributes not accessible - this is expected
        cv2.putText(output_frame, f"Torso: {result.measurements.torso_length_cm:.1f}cm", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Torso Diagnostic', output_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and result:
        print("\n[Captured measurement - see above]")

cap.release()
cv2.destroyAllWindows()

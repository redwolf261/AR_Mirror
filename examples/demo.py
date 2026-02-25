#!/usr/bin/env python3
"""
Quick demonstration of the AR sizing system using webcam
Tests the pipeline with sample garment data
"""

import cv2
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.legacy.sizing_pipeline import SizingPipeline
except ImportError:
    print("Error: Required modules not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    print("=" * 60)
    print("AR SIZING SYSTEM - QUICK DEMO")
    print("=" * 60)
    print()
    print("This demo will:")
    print("1. Open your webcam")
    print("2. Detect your pose")
    print("3. Estimate body measurements")
    print("4. Compare against sample garment (SKU-001, Size M)")
    print("5. Show fit recommendation")
    print()
    print("Instructions:")
    print("- Stand 1.5-2m from camera")
    print("- Face camera directly")
    print("- Keep full torso visible")
    print("- Good lighting recommended")
    print()
    print("Press 'q' to quit, 's' to switch garment")
    print("=" * 60)
    print()
    
    garment_db_path = "data/garments/garment_database.json"
    log_dir = "data/logs"
    
    if not Path(garment_db_path).exists():
        print(f"Error: {garment_db_path} not found")
        print("Run from the project root directory")
        sys.exit(1)
    
    Path(log_dir).mkdir(exist_ok=True)
    
    pipeline = SizingPipeline(garment_db_path, log_dir)
    
    available_skus = ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]
    current_sku_idx = 0
    pipeline.set_garment(available_skus[current_sku_idx])
    
    print(f"Testing with garment: {available_skus[current_sku_idx]}")
    print("Opening camera...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        frame_count += 1
        
        output_frame, result = pipeline.process_frame(frame)
        
        info_text = f"Frame: {frame_count} | SKU: {available_skus[current_sku_idx]}"
        cv2.putText(output_frame, info_text, (10, output_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('AR Sizing Demo', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            current_sku_idx = (current_sku_idx + 1) % len(available_skus)
            pipeline.set_garment(available_skus[current_sku_idx])
            print(f"Switched to garment: {available_skus[current_sku_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("Demo ended")
    print(f"Logs saved to: {log_dir}/")
    print()
    print("Next steps:")
    print("1. Review logs for measurement data")
    print("2. Test with different body types")
    print("3. Validate against manual measurements")
    print("4. Adjust thresholds if needed")


if __name__ == "__main__":
    main()

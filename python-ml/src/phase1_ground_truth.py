"""
PHASE 1: Ground Truth Data Collection
Captures manual measurements alongside system predictions
"""
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from .sizing_pipeline import SizingPipeline

class GroundTruthCollector:
    """Collects paired data: system predictions + manual measurements"""
    
    def __init__(self, output_dir='validation_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pipeline = SizingPipeline("garment_database.json", "logs")
        self.pipeline.set_garment("SKU-001")
        
    def collect_session(self, subject_id, manual_measurements):
        """
        Collect one validation session
        
        Args:
            subject_id: Unique identifier (e.g., "sub001")
            manual_measurements: Dict with tape measurements
                {
                    'shoulder_width_cm': float,
                    'chest_width_cm': float,
                    'torso_length_cm': float,
                    'waist_circumference_cm': float,
                    'hip_width_cm': float,
                    'inseam_cm': float
                }
        
        Returns:
            Dict with results and file paths
        """
        
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH COLLECTION: {subject_id}")
        print(f"{'='*60}")
        
        # Validate manual measurements
        if not self._validate_manual_measurements(manual_measurements):
            return {'error': 'Invalid manual measurements'}
        
        # Camera capture
        print("\n📹 Starting camera...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        system_results = None
        frame_saved = None
        
        print("\nInstructions:")
        print("  - Stand 2-3m from camera")
        print("  - Face camera, arms at sides")
        print("  - Good lighting, plain background")
        print("  - Press SPACE when pose detected")
        print("  - Press Q to abort")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame - returns (output_frame, fit_result or None)
            output_frame, fit_result = self.pipeline.process_frame(frame)
            
            # Display
            display = output_frame.copy()
            if fit_result:
                # Show measurements
                y = 30
                measurements = {
                    'shoulder_width_cm': fit_result.measurements.shoulder_width_cm,
                    'chest_width_cm': fit_result.measurements.chest_width_cm,
                    'torso_length_cm': fit_result.measurements.torso_length_cm
                }
                for key, value in measurements.items():
                    text = f"{key}: {value:.1f}cm"
                    cv2.putText(display, text, (10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 25
                
                cv2.putText(display, "POSE GOOD - Press SPACE", (10, frame.shape[0]-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Waiting: No valid pose detected",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Ground Truth Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and fit_result:
                system_results = fit_result
                frame_saved = frame.copy()
                break
            elif key == ord('q'):
                print("❌ Aborted by user")
                cap.release()
                cv2.destroyAllWindows()
                return {'error': 'Aborted'}
        
        cap.release()
        cv2.destroyAllWindows()
        
        if system_results is None:
            return {'error': 'No valid measurement captured'}
        
        # Save data
        print("\n💾 Saving data...")
        session_data = {
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'manual_measurements': manual_measurements,
            'system_measurements': {
                'shoulder_width_cm': system_results.measurements.shoulder_width_cm,
                'chest_width_cm': system_results.measurements.chest_width_cm,
                'torso_length_cm': system_results.measurements.torso_length_cm
            },
            'system_metadata': {
                'confidence': system_results.measurements.confidence,
                'fit_decision': system_results.decision.value
            }
        }
        
        # Save JSON
        json_path = self.output_dir / f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save frame
        frame_path = self.output_dir / f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if frame_saved is not None:
            cv2.imwrite(str(frame_path), frame_saved)
        
        print(f"  ✓ Data: {json_path}")
        print(f"  ✓ Frame: {frame_path}")
        
        # Compute errors
        print("\n📊 Measurement Comparison:")
        print(f"{'Dimension':<20} {'Manual':<10} {'System':<10} {'Error':<10}")
        print("-" * 50)
        
        errors = {}
        for key in ['shoulder_width_cm', 'chest_width_cm', 'torso_length_cm']:
            manual = manual_measurements.get(key)
            system = session_data['system_measurements'].get(key)
            
            if manual and system:
                error = abs(system - manual)
                errors[key] = error
                print(f"{key:<20} {manual:<10.1f} {system:<10.1f} {error:<10.2f}")
        
        mae = np.mean(list(errors.values())) if errors else None
        if mae:
            print(f"\n{'Mean Absolute Error:':<20} {mae:.2f} cm")
            
            if mae <= 2.5:
                print("  ✅ PASS (MAE ≤ 2.5cm)")
            elif mae <= 3.0:
                print("  ⚠️  MARGINAL (2.5 < MAE ≤ 3.0cm)")
            else:
                print("  ❌ FAIL (MAE > 3.0cm)")
        
        return {
            'success': True,
            'subject_id': subject_id,
            'json_path': str(json_path),
            'frame_path': str(frame_path),
            'mae': mae,
            'errors': errors
        }
    
    def _validate_manual_measurements(self, measurements):
        """Validate manual measurements are in reasonable range"""
        
        ranges = {
            'shoulder_width_cm': (35, 55),
            'chest_width_cm': (30, 50),
            'torso_length_cm': (45, 75),
            'waist_circumference_cm': (60, 120),
            'hip_width_cm': (30, 60),
            'inseam_cm': (65, 95)
        }
        
        for key, (min_val, max_val) in ranges.items():
            if key in measurements:
                value = measurements[key]
                if not (min_val <= value <= max_val):
                    print(f"  ❌ {key} = {value}cm out of range [{min_val}, {max_val}]")
                    return False
        
        return True
    
    def _draw_landmarks(self, frame, landmarks):
        """Draw MediaPipe landmarks"""
        h, w = frame.shape[:2]
        
        connections = [
            (11, 12),  # Shoulders
            (11, 23),  # Left torso
            (12, 24),  # Right torso
            (23, 24),  # Hips
        ]
        
        for lm in landmarks:
            x, y = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                x1, y1 = int(landmarks[start]['x'] * w), int(landmarks[start]['y'] * h)
                x2, y2 = int(landmarks[end]['x'] * w), int(landmarks[end]['y'] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def interactive_collection():
    """Interactive data collection session"""
    
    collector = GroundTruthCollector()
    
    print("\n" + "="*70)
    print("PHASE 1: GROUND TRUTH DATA COLLECTION")
    print("="*70)
    print("\nThis tool collects paired measurements:")
    print("  1. You provide manual tape measurements")
    print("  2. System captures AR measurements")
    print("  3. Both saved for validation analysis")
    
    while True:
        print("\n" + "-"*70)
        subject_id = input("\nSubject ID (e.g., sub001) or 'quit': ").strip()
        
        if subject_id.lower() == 'quit':
            break
        
        if not subject_id:
            print("❌ Subject ID required")
            continue
        
        print("\nEnter manual measurements (from MEASUREMENT_PROTOCOL.md):")
        print("(Press Enter to skip optional measurements)")
        
        manual = {}
        
        # Required measurements
        try:
            shoulder = float(input("  Shoulder width (cm): "))
            chest = float(input("  Chest width (cm): "))
            torso = float(input("  Torso length (cm): "))
            
            manual['shoulder_width_cm'] = shoulder
            manual['chest_width_cm'] = chest
            manual['torso_length_cm'] = torso
            
            # Optional measurements
            waist_str = input("  Waist circumference (cm) [optional]: ").strip()
            if waist_str:
                manual['waist_circumference_cm'] = float(waist_str)
            
            hip_str = input("  Hip width (cm) [optional]: ").strip()
            if hip_str:
                manual['hip_width_cm'] = float(hip_str)
            
            inseam_str = input("  Inseam (cm) [optional]: ").strip()
            if inseam_str:
                manual['inseam_cm'] = float(inseam_str)
        
        except ValueError:
            print("❌ Invalid number format")
            continue
        
        # Collect
        result = collector.collect_session(subject_id, manual)
        
        if result.get('success'):
            print(f"\n✅ Session saved: {result['subject_id']}")
        else:
            print(f"\n❌ Collection failed: {result.get('error')}")

if __name__ == "__main__":
    interactive_collection()

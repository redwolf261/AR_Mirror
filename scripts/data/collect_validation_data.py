"""
Data Collection Tool for Days 7-30 Human Validation
Records system measurements + manual tape measurements for MAE calculation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import json
from datetime import datetime
from src.legacy.sizing_pipeline import SizingPipeline

class ValidationDataCollector:
    def __init__(self, output_dir="validation_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_file = self.output_dir / f"validation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.subjects = []
        
    def collect_subject(self):
        """Collect data for one subject"""
        print("=" * 70)
        print("VALIDATION DATA COLLECTION")
        print("=" * 70)
        
        # Get subject metadata
        subject_id = len(self.subjects) + 1
        print(f"\nSubject #{subject_id}")
        
        # Demographics (optional but useful for PHASE 2)
        age = input("Age (optional, press Enter to skip): ").strip()
        gender = input("Gender (M/F/Other, optional): ").strip()
        notes = input("Notes (clothing, lighting, etc.): ").strip()
        
        print("\n1. System Measurement")
        print("   Stand in front of camera, press SPACE when ready")
        
        pipeline = SizingPipeline("garment_database.json", "logs", adaptive_mode=False)
        pipeline.set_garment("SKU-001")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        system_measurement = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output_frame, result = pipeline.process_frame(frame)
            
            cv2.putText(output_frame, "Press SPACE to capture measurement", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, "Press Q to skip subject", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Validation - System Measurement', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and result:
                system_measurement = {
                    'shoulder_cm': result.measurements.shoulder_width_cm,
                    'chest_cm': result.measurements.chest_width_cm,
                    'torso_cm': result.measurements.torso_length_cm,
                    'confidence': result.measurements.confidence,
                    'fit_decision': result.decision.value,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"\n   Captured: Shoulder={system_measurement['shoulder_cm']:.1f}cm, "
                      f"Chest={system_measurement['chest_cm']:.1f}cm, "
                      f"Torso={system_measurement['torso_cm']:.1f}cm")
                break
            
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not system_measurement:
            print("\n   SKIP: No measurement captured")
            return False
        
        # Get manual tape measurements
        print("\n2. Manual Tape Measurement (Ground Truth)")
        print("   Measure with tape and enter values")
        
        try:
            manual_shoulder = float(input("   Shoulder width (cm): "))
            manual_chest = float(input("   Chest breadth (cm): "))
            manual_torso = float(input("   Torso length (cm): "))
        except ValueError:
            print("   SKIP: Invalid input")
            return False
        
        manual_measurement = {
            'shoulder_cm': manual_shoulder,
            'chest_cm': manual_chest,
            'torso_cm': manual_torso,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate errors
        errors = {
            'shoulder': system_measurement['shoulder_cm'] - manual_measurement['shoulder_cm'],
            'chest': system_measurement['chest_cm'] - manual_measurement['chest_cm'],
            'torso': system_measurement['torso_cm'] - manual_measurement['torso_cm']
        }
        
        mae = sum(abs(e) for e in errors.values()) / 3
        
        print(f"\n   Errors: Shoulder={errors['shoulder']:+.1f}cm, "
              f"Chest={errors['chest']:+.1f}cm, "
              f"Torso={errors['torso']:+.1f}cm")
        print(f"   MAE: {mae:.2f}cm")
        
        # Store subject data
        subject_data = {
            'subject_id': subject_id,
            'demographics': {
                'age': age or None,
                'gender': gender or None
            },
            'notes': notes,
            'system_measurement': system_measurement,
            'manual_measurement': manual_measurement,
            'errors': errors,
            'mae': mae,
            'timestamp': datetime.now().isoformat()
        }
        
        self.subjects.append(subject_data)
        self._save_session()
        
        return True
    
    def run_session(self, target_subjects=5):
        """Collect data from multiple subjects"""
        print("\n" + "=" * 70)
        print(f"TARGET: Collect {target_subjects} subjects")
        print("=" * 70)
        
        while len(self.subjects) < target_subjects:
            print(f"\nProgress: {len(self.subjects)}/{target_subjects} subjects collected")
            
            cont = input("Collect next subject? (y/n): ").strip().lower()
            if cont != 'y':
                break
            
            success = self.collect_subject()
            if success:
                print(f"✓ Subject #{len(self.subjects)} saved")
        
        self._print_summary()
    
    def _save_session(self):
        """Save session data"""
        session_data = {
            'session_date': datetime.now().isoformat(),
            'num_subjects': len(self.subjects),
            'subjects': self.subjects
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _print_summary(self):
        """Print session summary"""
        if not self.subjects:
            print("\nNo subjects collected")
            return
        
        import numpy as np
        
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        
        maes = [s['mae'] for s in self.subjects]
        shoulder_errors = [s['errors']['shoulder'] for s in self.subjects]
        chest_errors = [s['errors']['chest'] for s in self.subjects]
        torso_errors = [s['errors']['torso'] for s in self.subjects]
        
        print(f"\nSubjects collected: {len(self.subjects)}")
        print(f"\nMean Absolute Error (MAE):")
        print(f"  Overall:  {np.mean(maes):.2f} ± {np.std(maes):.2f} cm")
        print(f"  Shoulder: {np.mean(np.abs(shoulder_errors)):.2f} ± {np.std(np.abs(shoulder_errors)):.2f} cm")
        print(f"  Chest:    {np.mean(np.abs(chest_errors)):.2f} ± {np.std(np.abs(chest_errors)):.2f} cm")
        print(f"  Torso:    {np.mean(np.abs(torso_errors)):.2f} ± {np.std(np.abs(torso_errors)):.2f} cm")
        
        print(f"\nBias (systematic error):")
        print(f"  Shoulder: {np.mean(shoulder_errors):+.2f} cm")
        print(f"  Chest:    {np.mean(chest_errors):+.2f} cm")
        print(f"  Torso:    {np.mean(torso_errors):+.2f} cm")
        
        print("\n" + "=" * 70)
        print("DAY 30 GATE: MAE < 2.5cm")
        print("=" * 70)
        
        overall_mae = np.mean(maes)
        if overall_mae < 2.5:
            print(f"✓ PASS: {overall_mae:.2f}cm < 2.5cm")
            print("  Ready for retailer pilot discussion")
        else:
            print(f"✗ FAIL: {overall_mae:.2f}cm ≥ 2.5cm")
            print("  Need to reduce error before pilot")
            print("\n  Review biases above - can PHASE 2 corrections help?")
        
        print("=" * 70)
        print(f"\nData saved to: {self.session_file}")
        
        if len(self.subjects) < 30:
            print(f"\nNote: Only {len(self.subjects)} subjects collected")
            print(f"      Target is 30-50 for robust validation")

def main():
    collector = ValidationDataCollector()
    collector.run_session(target_subjects=5)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

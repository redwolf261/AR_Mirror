"""
SELF-TEST Section 1: Repeatability Validation
Tests if system produces consistent measurements under identical conditions
CRITICAL: Must pass ≤5% variance before proceeding to human validation
"""
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.sizing_pipeline import SizingPipeline

class RepeatabilityTest:
    def __init__(self, num_runs=10, output_dir="validation_data"):
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_test(self):
        """Run repeatability test"""
        print("=" * 70)
        print("SELF-TEST SECTION 1: REPEATABILITY VALIDATION")
        print("=" * 70)
        print(f"\nInstructions:")
        print("  1. Stand in SAME position for all {0} measurements".format(self.num_runs))
        print("  2. Keep lighting CONSTANT")
        print("  3. Wear SAME clothing")
        print("  4. Stand STILL during each measurement")
        print("  5. Press SPACE to capture each measurement")
        print("  6. Press 'q' to quit\n")
        print("Gate: Variance must be ≤5% on shoulder, chest")
        print("Note: Torso measurement optional (requires full body visible)")
        print("=" * 70)
        
        input("\nPress ENTER when ready to start...")
        
        pipeline = SizingPipeline("garment_database.json", "logs", adaptive_mode=True)
        pipeline.set_garment("SKU-001")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Let camera auto-expose initially, then try to lock
        print("✓ Warming up camera (auto-exposure stabilizing)...")
        for _ in range(90):  # 3 seconds warm-up
            cap.read()
        
        # Try to lock exposure at current level (may not work on all cameras)
        try:
            # Some cameras support locking exposure after warm-up
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Try manual mode
            print("✓ Attempted to lock camera exposure")
        except:
            print("⚠ Camera exposure lock not supported - using auto mode")
        
        print("✓ Camera ready\n")
        
        measurement_count = 0
        
        print(f"\n[0/{self.num_runs}] Position yourself and press SPACE when ready")
        
        while measurement_count < self.num_runs:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use regular mode for repeatability test (needs full measurements)
            output_frame, result = pipeline.process_frame(frame)
            
            # Add instruction overlay
            cv2.putText(output_frame, 
                       f"Measurement {measurement_count}/{self.num_runs}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(output_frame, 
                       "Press SPACE to capture, Q to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show pose detection status
            if result:
                cv2.putText(output_frame, 
                           "Pose detected - Ready to capture", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(output_frame, 
                           "Waiting for pose detection...", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Repeatability Test', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar
                if result:
                    # Capture measurement
                    measurement = {
                        'run': measurement_count + 1,
                        'timestamp': datetime.now().isoformat(),
                        'shoulder_cm': result.measurements.shoulder_width_cm,
                        'chest_cm': result.measurements.chest_width_cm,
                        'torso_cm': result.measurements.torso_length_cm,
                        'confidence': result.measurements.confidence,
                        'fit_decision': result.decision.value
                    }
                    self.results.append(measurement)
                    measurement_count += 1
                    
                    print(f"[{measurement_count}/{self.num_runs}] Captured: "
                          f"Shoulder={measurement['shoulder_cm']:.1f}cm, "
                          f"Chest={measurement['chest_cm']:.1f}cm, "
                          f"Torso={measurement['torso_cm']:.1f}cm, "
                          f"Confidence={measurement['confidence']:.2f}")
                    
                    if measurement_count < self.num_runs:
                        print(f"      Stand still, press SPACE for next measurement")
                else:
                    print("      ⚠ Cannot capture - no pose detected (step back or adjust camera)")
                    continue
                
                if measurement_count < self.num_runs:
                    print(f"      Stand still, press SPACE for next measurement")
            
            elif key == ord('q'):
                print("\nTest aborted by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if measurement_count < self.num_runs:
            print(f"\nWARNING: Only {measurement_count}/{self.num_runs} measurements captured")
            if measurement_count < 5:
                print("FAIL: Need at least 5 measurements for variance analysis")
                return False
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze variance in measurements"""
        if len(self.results) < 2:
            print("\nFAIL: Need at least 2 measurements")
            return False
        
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        # Extract measurements
        shoulders = [r['shoulder_cm'] for r in self.results]
        chests = [r['chest_cm'] for r in self.results]
        torsos = [r['torso_cm'] for r in self.results]
        confidences = [r['confidence'] for r in self.results]
        
        # Calculate statistics
        stats = {
            'shoulder': self._calculate_stats(shoulders),
            'chest': self._calculate_stats(chests),
            'torso': self._calculate_stats(torsos),
            'confidence': self._calculate_stats(confidences)
        }
        
        # Print results
        print(f"\nMeasurement Statistics (n={len(self.results)}):\n")
        
        for measurement, data in stats.items():
            print(f"{measurement.upper()}")
            print(f"  Mean:   {data['mean']:.2f} {'cm' if measurement != 'confidence' else ''}")
            print(f"  Std:    {data['std']:.2f}")
            print(f"  CV:     {data['cv']:.1f}%")
            print(f"  Range:  {data['min']:.2f} - {data['max']:.2f}")
            print()
        
        # Check gate (only shoulder and chest - torso requires full body)
        print("=" * 70)
        print("GATE: Coefficient of Variation ≤5% (shoulder, chest)")
        print("=" * 70)
        
        passed = True
        for measurement in ['shoulder', 'chest']:
            cv = stats[measurement]['cv']
            status = "PASS" if cv <= 5.0 else "FAIL"
            print(f"{measurement.upper()}: {cv:.1f}% - {status}")
            if cv > 5.0:
                passed = False
        
        # Report torso for informational purposes only
        torso_cv = stats['torso']['cv']
        torso_detected = stats['torso']['mean'] > 0
        if torso_detected:
            print(f"TORSO: {torso_cv:.1f}% - INFO ONLY (requires full body)")
        else:
            print(f"TORSO: Not detected - INFO ONLY (requires hips visible)")
        
        print("=" * 70)
        
        if passed:
            print("✓ REPEATABILITY TEST PASSED")
            print("  System is stable enough for human validation")
        else:
            print("✗ REPEATABILITY TEST FAILED")
            print("  System variance too high - debug before recruiting others")
            print("\nPossible causes:")
            print("  • Lighting changed between measurements")
            print("  • You moved position between captures")
            print("  • Camera auto-exposure/white balance adjusting")
            print("  • Algorithm instability (needs fixing)")
        
        print("=" * 70)
        
        # Save results
        self._save_results(stats, passed)
        
        return passed
    
    def _calculate_stats(self, values):
        """Calculate statistical measures"""
        mean = np.mean(values)
        std = np.std(values)
        cv = (std / mean * 100) if mean > 0 else 0
        
        return {
            'mean': mean,
            'std': std,
            'cv': cv,
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    def _save_results(self, stats, passed):
        """Save test results to file"""
        output = {
            'test_type': 'repeatability',
            'date': datetime.now().isoformat(),
            'num_runs': len(self.results),
            'passed': passed,
            'gate_threshold': 5.0,
            'statistics': {k: {sk: (sv if sk != 'values' else None) 
                              for sk, sv in v.items()} 
                          for k, v in stats.items()},
            'raw_measurements': self.results
        }
        
        filename = self.output_dir / f"repeatability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    test = RepeatabilityTest(num_runs=10)
    passed = test.run_test()
    
    print("\n" + "=" * 70)
    if passed:
        print("Next Step: Invite 1 external tester and repeat this test")
        print("If they also pass, proceed to 30-50 person validation")
    else:
        print("Next Step: Debug variance sources before external testing")
        print("Review FRAGILITIES.md and check:")
        print("  • Lighting stability")
        print("  • Head-height scaling sensitivity")
        print("  • Pose detection robustness")
    print("=" * 70)
    
    return 0 if passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

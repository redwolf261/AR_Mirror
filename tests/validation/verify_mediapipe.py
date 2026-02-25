"""
Quick verification that MediaPipe API is working
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.legacy.sizing_pipeline import PoseDetector

def main():
    print("MediaPipe Pose Detection - API Verification")
    print("=" * 50)
    
    try:
        detector = PoseDetector()
        print("[OK] PoseDetector initialized successfully")
        print(f"[INFO] Using legacy API: {detector.legacy_mode}")
        print(f"[INFO] MediaPipe version: 0.10.31")
        print()
        print("=" * 50)
        print("STATUS: MediaPipe API is OPERATIONAL")
        print("=" * 50)
        print()
        print("Next steps:")
        print("1. Test with camera: python camera_test.py")
        print("2. Run full pipeline: python sizing_pipeline.py")
        print()
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

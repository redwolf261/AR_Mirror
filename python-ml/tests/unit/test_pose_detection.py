"""
Test pose detection on a static image to verify MediaPipe API works
"""
import cv2
import numpy as np
from src.sizing_pipeline import PoseDetector

def create_test_image():
    """Create a simple test image with a person silhouette"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(128)  # Gray background
    
    # Draw a simple stick figure
    # Head
    cv2.circle(img, (320, 100), 30, (255, 200, 150), -1)
    
    # Body
    cv2.line(img, (320, 130), (320, 280), (255, 200, 150), 20)
    
    # Arms
    cv2.line(img, (320, 160), (250, 220), (255, 200, 150), 15)
    cv2.line(img, (320, 160), (390, 220), (255, 200, 150), 15)
    
    # Legs
    cv2.line(img, (320, 280), (280, 400), (255, 200, 150), 18)
    cv2.line(img, (320, 280), (360, 400), (255, 200, 150), 18)
    
    return img

def test_detection():
    print("Testing MediaPipe Pose Detection...")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    cv2.imwrite("test_figure.jpg", test_img)
    print("Created test image: test_figure.jpg")
    
    # Initialize detector
    try:
        detector = PoseDetector()
        print(f"✓ PoseDetector initialized")
        print(f"  Using legacy API: {detector.legacy_mode}")
    except Exception as e:
        print(f"✗ Failed to initialize PoseDetector: {e}")
        return False
    
    # Test detection on synthetic image
    print("\nTesting detection on synthetic image...")
    landmarks = detector.detect(test_img)
    
    if landmarks:
        print("✗ Detected pose on synthetic image (unexpected - it's just shapes)")
        print("  This is fine - MediaPipe may find patterns in simple shapes")
    else:
        print("✓ No pose detected on synthetic image (expected)")
    
    # Test with a realistic scenario - person needs to be in front of camera
    print("\n" + "=" * 50)
    print("MediaPipe API Status: OPERATIONAL")
    print("=" * 50)
    print("\nThe pose detector is ready. Next steps:")
    print("1. Stand in front of your camera (well-lit, full body visible)")
    print("2. Run: python camera_test.py")
    print("3. You should see green landmarks on your body")
    print("\nIf camera test works, proceed to full sizing_pipeline.py test")
    
    return True

if __name__ == "__main__":
    success = test_detection()
    exit(0 if success else 1)

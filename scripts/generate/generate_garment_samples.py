"""
Generate sample garment images for AR try-on demo
Creates realistic-looking clothing overlays
"""
import cv2
import numpy as np
from pathlib import Path

def create_shirt_sample():
    """Create a simple shirt overlay with sleeves"""
    h, w = 300, 200
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Main body (light blue)
    cv2.rectangle(img, (40, 30), (160, 180), (200, 150, 100, 200), -1)
    
    # Left sleeve
    cv2.ellipse(img, (30, 70), (35, 50), 0, 0, 360, (200, 150, 100, 180), -1)
    
    # Right sleeve
    cv2.ellipse(img, (170, 70), (35, 50), 0, 0, 360, (200, 150, 100, 180), -1)
    
    # Neckline
    cv2.circle(img, (100, 40), 15, (200, 150, 100, 200), -1)
    
    # Buttons (decorative)
    cv2.circle(img, (100, 80), 3, (100, 100, 100, 255), -1)
    cv2.circle(img, (100, 110), 3, (100, 100, 100, 255), -1)
    cv2.circle(img, (100, 140), 3, (100, 100, 100, 255), -1)
    
    return img

def create_pants_sample():
    """Create simple pants overlay"""
    h, w = 350, 180
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Waistband
    cv2.rectangle(img, (30, 10), (150, 40), (60, 60, 80, 200), -1)
    
    # Left leg
    cv2.rectangle(img, (30, 40), (85, 320), (60, 60, 120, 190), -1)
    
    # Right leg
    cv2.rectangle(img, (95, 40), (150, 320), (60, 60, 120, 190), -1)
    
    # Seam details
    cv2.line(img, (57, 40), (57, 320), (40, 40, 60, 150), 2)
    cv2.line(img, (123, 40), (123, 320), (40, 40, 60, 150), 2)
    
    # Belt (optional)
    cv2.rectangle(img, (30, 25), (150, 35), (80, 60, 40, 200), -1)
    cv2.circle(img, (90, 30), 5, (100, 80, 60, 255), -1)
    
    return img

def create_dress_sample():
    """Create a simple dress overlay"""
    h, w = 380, 220
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Main body (pink/purple)
    points = np.array([
        [50, 30],   # Left shoulder
        [170, 30],  # Right shoulder
        [180, 100], # Right waist
        [190, 380], # Right hem
        [30, 380],  # Left hem
        [40, 100]   # Left waist
    ], dtype=np.int32)
    
    cv2.fillPoly(img, [points], (150, 100, 150, 200))
    
    # Waist definition
    cv2.rectangle(img, (45, 95), (175, 110), (120, 70, 120, 255), -1)
    
    # Neckline
    cv2.circle(img, (110, 35), 20, (150, 100, 150, 200), -1)
    
    # Details - simple stripes
    for x in range(60, 160, 15):
        cv2.line(img, (x, 120), (x-15, 280), (130, 80, 130, 150), 1)
    
    return img

def create_jacket_sample():
    """Create a simple jacket overlay"""
    h, w = 320, 240
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Main body (dark gray)
    cv2.rectangle(img, (50, 20), (190, 200), (80, 80, 80, 210), -1)
    
    # Left sleeve
    cv2.ellipse(img, (30, 80), (40, 60), 0, 0, 360, (80, 80, 80, 190), -1)
    
    # Right sleeve
    cv2.ellipse(img, (210, 80), (40, 60), 0, 0, 360, (80, 80, 80, 190), -1)
    
    # Lapels
    points = np.array([
        [120, 40],
        [150, 120],
        [110, 120]
    ], dtype=np.int32)
    cv2.fillPoly(img, [points], (60, 60, 60, 200))
    
    # Buttons
    for y in [70, 110, 150]:
        cv2.circle(img, (120, y), 4, (40, 40, 40, 255), -1)
    
    return img

def create_samples():
    """Create all garment samples"""
    output_dir = Path("garment_samples")
    output_dir.mkdir(exist_ok=True)
    
    samples = {
        'shirt_sample.png': create_shirt_sample(),
        'pants_sample.png': create_pants_sample(),
        'dress_sample.png': create_dress_sample(),
        'jacket_sample.png': create_jacket_sample()
    }
    
    for filename, img in samples.items():
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), img)
        print(f"✓ Created {filepath}")
    
    return output_dir

if __name__ == "__main__":
    print("Generating garment samples...\n")
    output_dir = create_samples()
    print(f"\n✓ All samples created in {output_dir}/")
    print("\nYou can now use these in the AR demo!")

"""
Generate Placeholder Garment Images
Creates simple shirt-shaped images for each SKU and size
"""

import cv2
import numpy as np
from pathlib import Path


def create_placeholder_garment(sku: str, size_name: str, output_path: str, color_index: int = 0):
    """Create a simple shirt-shaped garment placeholder"""
    h, w = 400, 250
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Different colors for different SKUs
    colors = [
        (180, 100, 100),  # Blue
        (100, 180, 100),  # Green
        (100, 100, 180),  # Red
        (150, 150, 100),  # Cyan
        (150, 100, 150),  # Purple
    ]
    
    color = colors[color_index % len(colors)]
    
    # Size-specific width adjustments
    size_widths = {
        'small': 0.65,
        'medium': 0.75,
        'large': 0.85,
        'xlarge': 0.95
    }
    
    width_factor = size_widths.get(size_name, 0.75)
    
    # Draw shirt shape
    shoulder_w = int(w * width_factor)
    waist_w = int(w * (width_factor - 0.1))
    
    # Shirt polygon (shoulders → waist)
    pts = np.array([
        [(w - shoulder_w) // 2, 50],      # Left shoulder
        [(w + shoulder_w) // 2, 50],      # Right shoulder
        [(w + waist_w) // 2, 350],        # Right waist
        [(w - waist_w) // 2, 350]         # Left waist
    ], np.int32)
    
    # Fill with semi-transparent color
    cv2.fillPoly(img, [pts], (*color, 180))  # BGRA
    
    # Add outline
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255, 200), thickness=2)
    
    # Add size label on garment
    label_text = size_name[0].upper()  # S, M, L, X
    cv2.putText(
        img, label_text, (w // 2 - 20, 200),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255, 255), 3
    )
    
    # Save
    cv2.imwrite(output_path, img)
    return output_path


def generate_all_placeholders():
    """Generate placeholder images for all SKUs and sizes"""
    sizes = ['small', 'medium', 'large', 'xlarge']
    skus = ['SKU-001', 'SKU-002', 'SKU-003', 'SKU-004', 'SKU-005']
    
    print("=" * 60)
    print("GENERATING PLACEHOLDER GARMENT IMAGES")
    print("=" * 60)
    
    total = 0
    for idx, sku in enumerate(skus):
        sku_dir = Path("garment_assets") / sku
        sku_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{sku}:")
        for size in sizes:
            output_path = sku_dir / f"front_{size}.png"
            create_placeholder_garment(sku, size, str(output_path), color_index=idx)
            print(f"  ✓ {output_path}")
            total += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {total} placeholder images")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace placeholders with real product photos (optional)")
    print("2. Run: python sizing_pipeline.py")
    print("3. Press 'n'/'p' keys to switch between sizes")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_placeholders()

"""Generate placeholder garment images for the full inventory"""
import cv2
import numpy as np
import json
from pathlib import Path

def create_placeholder_garment(size_label: str, sku: str, color_name: str, width=250, height=400):
    """Create a simple shirt-shaped placeholder image with transparency"""
    # Create RGBA image
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Color mapping for different garment types
    colors = {
        'Navy Blue': (139, 69, 19),
        'Black': (30, 30, 30),
        'White': (240, 240, 240),
        'Light Blue': (173, 216, 230),
        'Olive Green': (85, 107, 47),
        'Gray': (128, 128, 128),
        'Maroon': (128, 0, 0),
        'Navy': (0, 0, 128),
        'Forest Green': (34, 139, 34),
        'Burgundy': (128, 0, 32),
        'Charcoal': (54, 69, 79),
        'Beige': (245, 245, 220)
    }
    
    base_color = colors.get(color_name, (100, 149, 237))  # Default blue
    
    # Shirt shape (polygon points)
    center_x = width // 2
    shoulder_y = 80
    bottom_y = height - 40
    
    # Simple shirt silhouette
    points = np.array([
        [center_x - 80, shoulder_y],          # Left shoulder
        [center_x - 50, shoulder_y + 120],    # Left armpit
        [center_x - 50, bottom_y],            # Left bottom
        [center_x + 50, bottom_y],            # Right bottom
        [center_x + 50, shoulder_y + 120],    # Right armpit
        [center_x + 80, shoulder_y],          # Right shoulder
        [center_x, shoulder_y - 20],          # Neck
    ], dtype=np.int32)
    
    # Fill shirt shape
    cv2.fillPoly(img, [points], (*base_color, 200))
    
    # Add outline
    cv2.polylines(img, [points], True, (*base_color, 255), 2)
    
    # Add size label
    label_text = f"Size {size_label}"
    cv2.putText(img, label_text, (center_x - 40, center_x + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 2)
    
    # Add SKU at bottom
    cv2.putText(img, sku, (center_x - 50, bottom_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)
    
    return img

def main():
    # Load inventory
    with open('garment_inventory.json', 'r') as f:
        inventory = json.load(f)
    
    sizes = ['small', 'medium', 'large', 'xlarge']
    size_labels = ['S', 'M', 'L', 'XL']
    
    count = 0
    for garment in inventory:
        sku = garment['sku']
        color = garment['color']
        image_path = Path(garment['image_path'])
        
        # Create directory if needed
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Generate placeholder for each size
        for size, label in zip(sizes, size_labels):
            if label in garment['available_sizes']:
                img = create_placeholder_garment(label, sku, color)
                output_path = image_path / f"front_{size}.png"
                cv2.imwrite(str(output_path), img)
                count += 1
    
    print(f"✅ Generated {count} placeholder images for {len(inventory)} garments")

if __name__ == "__main__":
    main()

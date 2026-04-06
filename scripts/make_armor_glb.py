#!/usr/bin/env python3
"""
Generate realistic armor GLB model for testing full-body wrapping
Creates a metallic armor texture with proper UV mapping
"""

import numpy as np
import trimesh
import json
from pathlib import Path

VARIANTS = [
    {
        "sku": "ARMOR-001",
        "name": "Winter Warrior Armor",
        "folder": "ARMOR-TEST",
        "base": (182, 184, 190),
        "accent": (80, 82, 88),
        "shine": (255, 255, 255),
        "grid_spacing": 32,
        "scale_x": 1.00,
        "scale_y": 1.00,
        "depth": 0.10,
    },
    {
        "sku": "ARMOR-HEAVY-001",
        "name": "Heavy Plate Armor",
        "folder": "ARMOR-HEAVY",
        "base": (96, 100, 108),
        "accent": (56, 60, 68),
        "shine": (210, 214, 220),
        "grid_spacing": 24,
        "scale_x": 1.12,
        "scale_y": 1.08,
        "depth": 0.14,
    },
    {
        "sku": "ARMOR-LIGHT-001",
        "name": "Light Vanguard Armor",
        "folder": "ARMOR-LIGHT",
        "base": (214, 216, 220),
        "accent": (150, 154, 160),
        "shine": (255, 255, 255),
        "grid_spacing": 40,
        "scale_x": 0.96,
        "scale_y": 0.96,
        "depth": 0.08,
    },
    {
        "sku": "ARMOR-FANTASY-001",
        "name": "Fantasy Royal Armor",
        "folder": "ARMOR-FANTASY",
        "base": (120, 134, 172),
        "accent": (212, 180, 72),
        "shine": (248, 244, 220),
        "grid_spacing": 28,
        "scale_x": 1.00,
        "scale_y": 1.02,
        "depth": 0.12,
    },
]


# Create a simple armor mesh (metallic grid pattern)
def create_armor_mesh(scale_x=1.0, scale_y=1.0, depth=0.10):
    """Create a realistic metallic armor mesh"""
    
    # Create a basic torso-like quad mesh
    vertices = np.array([
        # Front face
        [-0.3, 0.0, 0.0],   # 0: left shoulder
        [0.3, 0.0, 0.0],    # 1: right shoulder
        [0.3, -0.8, 0.0],   # 2: right hip
        [-0.3, -0.8, 0.0],  # 3: left hip
        
        # Back face (for 3D depth)
        [-0.3, 0.0, 0.1],
        [0.3, 0.0, 0.1],
        [0.3, -0.8, 0.1],
        [-0.3, -0.8, 0.1],
    ], dtype=np.float32)

    vertices[:, 0] *= scale_x
    vertices[:, 1] *= scale_y
    vertices[4:, 2] = depth
    
    # Faces (triangles)
    faces = np.array([
        # Front
        [0, 1, 2],
        [0, 2, 3],
        # Back
        [5, 4, 6],
        [4, 7, 6],
        # Sides
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ], dtype=np.uint32)
    
    # UV coordinates for armor texture
    uv = np.array([
        [0.0, 1.0],  # 0
        [1.0, 1.0],  # 1
        [1.0, 0.0],  # 2
        [0.0, 0.0],  # 3
        [0.0, 1.0],  # 4
        [1.0, 1.0],  # 5
        [1.0, 0.0],  # 6
        [0.0, 0.0],  # 7
    ], dtype=np.float32)
    
    # Create metallic armor texture
    texture = create_armor_texture()
    
    # Create mesh with texture
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)
    
    return mesh

def create_armor_texture(width=512, height=512, base=(182, 184, 190), accent=(80, 82, 88), shine=(255, 255, 255), grid_spacing=32):
    """Create a realistic metallic armor texture with grid pattern"""
    from PIL import Image, ImageDraw
    
    # Create gradient metallic base (silver-gray)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        mix = y / max(1, height - 1)
        row = (np.array(base, dtype=np.float32) * (1.0 - mix) + np.array(accent, dtype=np.float32) * mix).astype(np.uint8)
        gradient[y, :] = row
    
    # Convert to PIL for drawing grid pattern
    img = Image.fromarray(gradient)
    draw = ImageDraw.Draw(img)
    
    # Draw armor grid lines (darker metal)
    line_color = accent
    
    # Horizontal lines
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=line_color, width=2)
    
    # Vertical lines
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=line_color, width=2)
    
    # Add metallic shine spots (brighter reflections)
    shine_color = shine
    for _ in range(20):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        draw.ellipse([x-15, y-15, x+15, y+15], fill=shine_color)
    
    return np.array(img)

inventory_path = Path("assets/garments/garment_inventory.json")
with open(inventory_path, 'r') as f:
    inventory = json.load(f)

inventory_by_sku = {item.get('sku'): item for item in inventory if isinstance(item, dict)}

for variant in VARIANTS:
    print(f"Generating {variant['name']}...")
    armor_mesh = create_armor_mesh(
        scale_x=variant['scale_x'],
        scale_y=variant['scale_y'],
        depth=variant['depth'],
    )
    armor_mesh.visual = trimesh.visual.TextureVisuals(
        uv=armor_mesh.visual.uv,
        image=create_armor_texture(
            base=variant['base'],
            accent=variant['accent'],
            shine=variant['shine'],
            grid_spacing=variant['grid_spacing'],
        ),
    )

    output_dir = Path("garment_assets") / variant['folder']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.glb"
    armor_mesh.export(str(output_path))
    print(f"✅ Exported {variant['sku']} to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    new_entry = {
        "sku": variant['sku'],
        "name": variant['name'],
        "category": "armor",
        "image_path": f"garment_assets/{variant['folder']}",
        "default_size": "M",
        "sizes": {
            "XS": {"shoulder_cm": 36, "chest_cm": 80, "length_cm": 60},
            "S": {"shoulder_cm": 40, "chest_cm": 90, "length_cm": 65},
            "M": {"shoulder_cm": 44, "chest_cm": 100, "length_cm": 70},
            "L": {"shoulder_cm": 48, "chest_cm": 110, "length_cm": 75},
            "XL": {"shoulder_cm": 52, "chest_cm": 120, "length_cm": 80}
        }
    }

    if variant['sku'] not in inventory_by_sku:
        inventory.append(new_entry)
        inventory_by_sku[variant['sku']] = new_entry
        print(f"✅ Added {variant['sku']} to inventory")
    else:
        print(f"ℹ️  {variant['sku']} already in inventory")

with open(inventory_path, 'w') as f:
    json.dump(inventory, f, indent=2)

print("\n✅ Done! Restart backend to load the armor variants.")

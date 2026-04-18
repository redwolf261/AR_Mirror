#!/usr/bin/env python3
"""Generate sample garments for VITON dataset"""
import cv2
import numpy as np
from pathlib import Path

# Create dataset directories
Path('dataset/train/cloth').mkdir(parents=True, exist_ok=True)
Path('dataset/train/cloth-mask').mkdir(parents=True, exist_ok=True)

print('Generating sample garments for VITON dataset...\n')

# Create 10 sample garments with different colors
colors = [
    ('red', (50, 50, 200)),
    ('blue', (200, 50, 50)),
    ('green', (50, 200, 50)),
    ('yellow', (50, 200, 200)),
    ('purple', (200, 50, 150)),
    ('cyan', (200, 200, 50)),
    ('orange', (50, 120, 255)),
    ('pink', (180, 120, 255)),
    ('white', (230, 230, 230)),
    ('gray', (120, 120, 120))
]

for i, (name, color) in enumerate(colors, 1):
    # Create garment image (shirt shape)
    h, w = 512, 384
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light background
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw shirt shape
    pts = np.array([
        [w//2-100, 100],  # Top left
        [w//2+100, 100],  # Top right
        [w//2+120, 200],  # Mid right
        [w//2+100, 400],  # Bottom right
        [w//2-100, 400],  # Bottom left
        [w//2-120, 200],  # Mid left
    ], np.int32)

    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], 255)

    # Add sleeves
    cv2.ellipse(img, (w//2-115, 150), (40, 60), -30, 0, 360, color, -1)
    cv2.ellipse(img, (w//2+115, 150), (40, 60), 30, 0, 360, color, -1)
    cv2.ellipse(mask, (w//2-115, 150), (40, 60), -30, 0, 360, 255, -1)
    cv2.ellipse(mask, (w//2+115, 150), (40, 60), 30, 0, 360, 255, -1)

    # Add some texture (buttons)
    cv2.circle(img, (w//2, 200), 8, (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)), -1)
    cv2.circle(img, (w//2, 250), 8, (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)), -1)
    cv2.circle(img, (w//2, 300), 8, (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)), -1)

    # Save files
    cloth_path = f'dataset/train/cloth/{i:05d}_00.jpg'
    mask_path = f'dataset/train/cloth-mask/{i:05d}_00.jpg'

    cv2.imwrite(cloth_path, img)
    cv2.imwrite(mask_path, mask)
    print(f'[OK] Created: {name:8s} -> {cloth_path}')

print(f'\nSuccess! Generated 10 sample garments in dataset/train/cloth/')
print('You can now run the AR Mirror application!')

"""
Garment Visualization System for AR Try-On
Phase 1: 2D Image Overlay with Warping
Phase 2: 3D Model Rendering (future)
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class GarmentAsset:
    """Represents a loaded garment image"""
    sku: str
    size_label: str
    view: str  # 'front', 'back', 'side'
    image: np.ndarray  # RGBA image with transparency
    width: int
    height: int
    metadata: Dict


class GarmentAssetManager:
    """Manages loading, caching, and retrieval of garment images"""
    
    def __init__(self, asset_dir: str = "garment_assets", max_cache_size: int = 20):
        """
        Initialize the garment asset manager.
        
        Args:
            asset_dir: Root directory containing garment assets
            max_cache_size: Max images to keep in memory (LRU)
        """
        self.asset_dir = Path(asset_dir)
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # LRU cache
        self.fallback_image = self._create_fallback_image()
        
        # Create asset directory if missing
        self.asset_dir.mkdir(parents=True, exist_ok=True)
    
    def get_garment(self, sku: str, size: str, view: str = 'front') -> Optional[np.ndarray]:
        """
        Get garment image for a specific SKU, size, and view.
        
        Returns:
            RGBA numpy array or fallback image if not found
        """
        cache_key = f"{sku}_{size}_{view}"
        
        # Check cache first
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)  # Move to end (LRU)
            return self.cache[cache_key].image
        
        # Try to load from disk
        asset = self._load_from_disk(sku, size, view)
        
        if asset is None:
            print(f"⚠ Garment not found: {sku}/{size}/{view}, using fallback")
            return self.fallback_image
        
        # Add to cache (with eviction if needed)
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        self.cache[cache_key] = asset
        return asset.image
    
    def _load_from_disk(self, sku: str, size: str, view: str) -> Optional[GarmentAsset]:
        """Load garment image from disk"""
        sku_dir = self.asset_dir / sku
        
        if not sku_dir.exists():
            return None
        
        # Map size labels to file names
        size_map = {
            's': 'small',
            'm': 'medium',
            'l': 'large',
            'xl': 'xlarge'
        }
        size_filename = size_map.get(size.lower(), size.lower())
        
        # Try common image formats
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = sku_dir / f"{view}_{size_filename}{ext}"
            
            if image_path.exists():
                # Load with alpha channel
                img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    continue
                
                # Ensure 4 channels (RGBA)
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:  # BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                
                # Load metadata if available
                metadata_path = sku_dir / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                return GarmentAsset(
                    sku=sku,
                    size_label=size,
                    view=view,
                    image=img,
                    width=img.shape[1],
                    height=img.shape[0],
                    metadata=metadata
                )
        
        return None
    
    def _create_fallback_image(self) -> np.ndarray:
        """Create a placeholder image when garment is missing"""
        placeholder = np.zeros((400, 250, 4), dtype=np.uint8)
        placeholder[:, :, 3] = 100  # Semi-transparent
        
        # Add text
        cv2.putText(
            placeholder, "Garment", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200, 255), 2
        )
        cv2.putText(
            placeholder, "Not Found", (40, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200, 255), 2
        )
        
        return placeholder
    
    def preload(self, sku_list: list) -> None:
        """Pre-load garments into cache"""
        for sku in sku_list:
            for size in ['small', 'medium', 'large', 'xlarge']:
                self.get_garment(sku, size, 'front')
    
    def clear_cache(self) -> None:
        """Clear all cached images"""
        self.cache.clear()


class GarmentVisualizer:
    """2D garment overlay visualization using image warping"""
    
    def __init__(self, asset_manager: GarmentAssetManager):
        self.asset_manager = asset_manager
        self.alpha_default = 0.65  # Transparency level
    
    def overlay_on_body(
        self,
        frame: np.ndarray,
        garment_image: np.ndarray,
        landmarks: Dict,
        body_shoulder_cm: float,
        body_chest_cm: float,
        size_label: str
    ) -> np.ndarray:
        """
        Overlay garment image on body in frame.
        
        Args:
            frame: Input video frame (BGR)
            garment_image: Garment image (RGBA)
            landmarks: MediaPipe landmarks dict
            body_shoulder_cm: Detected shoulder width in cm
            body_chest_cm: Detected chest width in cm
            size_label: Size label (S/M/L/XL) for reference
        
        Returns:
            Frame with garment overlaid
        """
        if landmarks is None or 11 not in landmarks or 12 not in landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Get shoulder coordinates
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        shoulder_x1 = int(left_shoulder['x'] * w)
        shoulder_y_left = int(left_shoulder['y'] * h)
        shoulder_y_right = int(right_shoulder['y'] * h)
        shoulder_y = (shoulder_y_left + shoulder_y_right) // 2
        shoulder_x2 = int(right_shoulder['x'] * w)
        
        # Scale garment based on body measurements
        garment_scaled = self._scale_garment(
            garment_image,
            body_shoulder_cm,
            body_chest_cm,
            size_label
        )
        
        if garment_scaled is None:
            return frame
        
        # Position garment on body
        g_h, g_w = garment_scaled.shape[:2]
        
        # Center horizontally on shoulders
        center_x = (shoulder_x1 + shoulder_x2) // 2
        garment_x = max(0, center_x - g_w // 2)
        
        # Fix Alignment: Lift garment so collar matches neck
        # Assume shoulders are ~20-25% down from top of garment image
        vertical_offset = int(g_h * 0.22) 
        garment_y = shoulder_y - vertical_offset
        
        # Blend onto frame
        output = self._blend_garment_on_frame(
            frame,
            garment_scaled,
            garment_x,
            garment_y,
            alpha=self.alpha_default
        )
        
        return output
    
    def _scale_garment(
        self,
        garment_img: np.ndarray,
        body_shoulder_cm: float,
        body_chest_cm: float,
        size_label: str
    ) -> Optional[np.ndarray]:
        """
        Scale garment image to match body proportions.
        Uses chest width as primary scaling metric.
        """
        if garment_img is None or garment_img.size == 0:
            return None
        
        # Reference chest widths for each size (cm)
        reference_widths = {
            'S': 46.0,
            'M': 50.0,
            'L': 54.0,
            'XL': 58.0
        }
        
        ref_width = reference_widths.get(size_label, 50.0)
        
        # Scale factor based on how garment fits
        scale_factor = body_chest_cm / ref_width
        
        # Clamp to reasonable range (0.7 - 1.5x)
        scale_factor = np.clip(scale_factor, 0.7, 1.5)
        
        # Apply scaling
        new_width = int(garment_img.shape[1] * scale_factor)
        new_height = int(garment_img.shape[0] * scale_factor)
        
        scaled = cv2.resize(garment_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return scaled
    
    def _blend_garment_on_frame(
        self,
        frame: np.ndarray,
        garment: np.ndarray,
        x: int,
        y: int,
        alpha: float = 0.65
    ) -> np.ndarray:
        """
        Blend garment image (with alpha channel) onto frame.
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        g_h, g_w = garment.shape[:2]
        
        # Calculate overlap region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + g_w)
        y2 = min(h, y + g_h)
        
        # Calculate garment region (with clipping)
        gx1 = max(0, -x)
        gy1 = max(0, -y)
        gx2 = gx1 + (x2 - x1)
        gy2 = gy1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return output  # No overlap
        
        # Extract regions
        frame_region = output[y1:y2, x1:x2]
        garment_region = garment[gy1:gy2, gx1:gx2]
        
        # Get alpha channel (assume RGBA)
        if garment_region.shape[2] == 4:
            garment_alpha = garment_region[:, :, 3].astype(float) / 255.0
        else:
            garment_alpha = np.ones((garment_region.shape[0], garment_region.shape[1]))
        
        # Apply global alpha multiplier
        garment_alpha = garment_alpha * alpha
        
        # Blend using alpha
        garment_bgr = garment_region[:, :, :3]
        
        for c in range(3):
            frame_region[:, :, c] = (
                (1 - garment_alpha) * frame_region[:, :, c] +
                garment_alpha * garment_bgr[:, :, c]
            ).astype(np.uint8)
        
        output[y1:y2, x1:x2] = frame_region
        
        return output


def create_garment_visualizer() -> GarmentVisualizer:
    """Factory function to create visualizer with asset manager"""
    asset_manager = GarmentAssetManager("garment_assets")
    visualizer = GarmentVisualizer(asset_manager)
    return visualizer


if __name__ == "__main__":
    # Test garment asset loading
    print("Testing Garment Asset Manager...")
    manager = GarmentAssetManager()
    
    test_image = manager.get_garment("SKU-001", "medium", "front")
    print(f"✓ Loaded test image: shape={test_image.shape}")
    
    visualizer = create_garment_visualizer()
    print("✓ Garment visualizer created successfully")

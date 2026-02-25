"""
Multi-Garment Layer Management System
Supports 2-5 simultaneous garments with z-order occlusion handling
PHASE 1A implementation
"""
import numpy as np
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import cv2

class LayerType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    OUTER = "outer"
    ACCESSORY = "accessory"

class ProductCategory(Enum):
    SHIRT = "shirt"
    TSHIRT = "tshirt"
    PANTS = "pants"
    SKIRT = "skirt"
    DRESS = "dress"
    JACKET = "jacket"
    HAT = "hat"
    WATCH = "watch"
    GLASSES = "glasses"

@dataclass
class GarmentLayer:
    garment_id: str
    category: ProductCategory
    layer_type: LayerType
    z_index: int
    overlay_image: Optional[np.ndarray] = None
    scale_factor: float = 1.0
    position_offset: tuple = (0, 0)
    opacity: float = 1.0

@dataclass
class MeasurementRequirements:
    shoulder: bool = False
    chest: bool = False
    torso: bool = False
    waist: bool = False
    hip: bool = False
    inseam: bool = False
    head_width: bool = False
    hand_length: bool = False

class LayerManager:
    """Manages multiple garment overlays with proper occlusion"""
    
    def __init__(self):
        self.layers: Dict[str, GarmentLayer] = {}
        self.z_index_map = {
            LayerType.LOWER: 1,
            LayerType.UPPER: 2,
            LayerType.OUTER: 3,
            LayerType.ACCESSORY: 4
        }
    
    def add_garment(self, garment_id: str, category: ProductCategory, 
                   layer_type: LayerType, overlay_image: np.ndarray) -> bool:
        """Add a garment to the layer stack"""
        if len(self.layers) >= 5:
            print(f"Warning: Maximum 5 garments supported. Remove one first.")
            return False
        
        z_index = self.z_index_map[layer_type]
        
        layer = GarmentLayer(
            garment_id=garment_id,
            category=category,
            layer_type=layer_type,
            z_index=z_index,
            overlay_image=overlay_image
        )
        
        self.layers[garment_id] = layer
        return True
    
    def remove_garment(self, garment_id: str) -> bool:
        """Remove a garment from the layer stack"""
        if garment_id in self.layers:
            del self.layers[garment_id]
            return True
        return False
    
    def get_sorted_layers(self) -> List[GarmentLayer]:
        """Get layers sorted by z-index for proper rendering order"""
        return sorted(self.layers.values(), key=lambda l: l.z_index)
    
    def get_measurements_needed(self) -> MeasurementRequirements:
        """Determine which measurements are needed for current garments"""
        requirements = MeasurementRequirements()
        
        for layer in self.layers.values():
            if layer.layer_type == LayerType.UPPER:
                requirements.shoulder = True
                requirements.chest = True
                requirements.torso = True
            elif layer.layer_type == LayerType.LOWER:
                requirements.waist = True
                requirements.hip = True
                requirements.inseam = True
            elif layer.layer_type == LayerType.OUTER:
                requirements.shoulder = True
                requirements.chest = True
                requirements.torso = True
            elif layer.layer_type == LayerType.ACCESSORY:
                if layer.category == ProductCategory.HAT:
                    requirements.head_width = True
                elif layer.category == ProductCategory.WATCH:
                    requirements.hand_length = True
        
        return requirements
    
    def render_layers(self, base_frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """Composite all layers onto base frame in z-order"""
        output = base_frame.copy()
        
        sorted_layers = self.get_sorted_layers()
        
        for layer in sorted_layers:
            if layer.overlay_image is not None:
                try:
                    output = self._composite_layer(output, layer, landmarks)
                except Exception as e:
                    # If positioning fails, still show the garment as a semi-transparent overlay
                    output = self._composite_layer_simple(output, layer)
        
        return output
    
    def _composite_layer_simple(self, base: np.ndarray, layer: GarmentLayer) -> np.ndarray:
        """Simple compositing - overlay garment color on upper/lower portion of frame"""
        overlay = layer.overlay_image
        if overlay is None:
            return base
        
        h, w = base.shape[:2]
        
        # Resize overlay to fit frame
        overlay_h, overlay_w = overlay.shape[:2]
        
        # Calculate scaling to fit
        scale = min(w / overlay_w, h / (overlay_h * 1.5))  # Fit within portion of frame
        new_w = int(overlay_w * scale)
        new_h = int(overlay_h * scale)
        
        overlay_resized = cv2.resize(overlay, (new_w, new_h))
        
        # Position based on layer type
        if layer.layer_type == LayerType.UPPER:
            # Upper portion (shoulders/chest)
            y_start = int(h * 0.15)
            x_start = int((w - new_w) / 2)
        elif layer.layer_type == LayerType.LOWER:
            # Lower portion (waist/hips/legs)
            y_start = int(h * 0.45)
            x_start = int((w - new_w) / 2)
        elif layer.layer_type == LayerType.OUTER:
            # Full body
            y_start = int(h * 0.10)
            x_start = int((w - new_w) / 2)
        else:
            y_start = int(h * 0.25)
            x_start = int((w - new_w) / 2)
        
        # Clamp to frame bounds
        y_end = min(y_start + new_h, h)
        x_end = min(x_start + new_w, w)
        
        if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
            return base
        
        # Alpha blend if available, otherwise just overlay with transparency
        if overlay_resized.shape[2] == 4:  # Has alpha channel
            alpha = overlay_resized[:, :, 3] / 255.0
            for c in range(3):
                base[y_start:y_end, x_start:x_end, c] = \
                    (1 - alpha) * base[y_start:y_end, x_start:x_end, c] + \
                    alpha * overlay_resized[:, :, c]
        else:
            # Simple overlay with 40% transparency
            base[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                base[y_start:y_end, x_start:x_end], 0.6,
                overlay_resized[:, :, :3], 0.4, 0
            )
        
        return base
    
    def _composite_layer(self, base: np.ndarray, layer: GarmentLayer, 
                        landmarks: Dict) -> np.ndarray:
        """Composite a single garment layer with proper positioning"""
        overlay = layer.overlay_image
        if overlay is None:
            return base
        
        # Calculate position based on layer type and landmarks
        if layer.layer_type == LayerType.UPPER:
            # Position based on shoulder landmarks
            if 11 in landmarks and 12 in landmarks:
                shoulder_center_x = int((landmarks[11]['x'] + landmarks[12]['x']) / 2 * base.shape[1])
                shoulder_center_y = int((landmarks[11]['y'] + landmarks[12]['y']) / 2 * base.shape[0])
                position = (shoulder_center_x, shoulder_center_y)
            else:
                return base
        
        elif layer.layer_type == LayerType.LOWER:
            # Position based on hip landmarks
            if 23 in landmarks and 24 in landmarks:
                hip_center_x = int((landmarks[23]['x'] + landmarks[24]['x']) / 2 * base.shape[1])
                hip_center_y = int((landmarks[23]['y'] + landmarks[24]['y']) / 2 * base.shape[0])
                position = (hip_center_x, hip_center_y)
            else:
                return base
        
        elif layer.layer_type == LayerType.ACCESSORY:
            # Position based on category
            if layer.category == ProductCategory.HAT and 0 in landmarks:
                head_x = int(landmarks[0]['x'] * base.shape[1])
                head_y = int(landmarks[0]['y'] * base.shape[0])
                position = (head_x, head_y - 50)
            elif layer.category == ProductCategory.WATCH and 15 in landmarks:
                wrist_x = int(landmarks[15]['x'] * base.shape[1])
                wrist_y = int(landmarks[15]['y'] * base.shape[0])
                position = (wrist_x, wrist_y)
            else:
                return base
        else:
            position = (base.shape[1] // 2, base.shape[0] // 2)
        
        # Apply alpha blending
        return self._alpha_blend(base, overlay, position, layer.opacity)
    
    def _alpha_blend(self, base: np.ndarray, overlay: np.ndarray, 
                    position: tuple, opacity: float) -> np.ndarray:
        """Alpha blend overlay onto base at specified position"""
        x, y = position
        h, w = overlay.shape[:2]
        
        # Calculate bounds
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(base.shape[1], x1 + w)
        y2 = min(base.shape[0], y1 + h)
        
        # Adjust overlay size if needed
        overlay_h = y2 - y1
        overlay_w = x2 - x1
        
        if overlay_h <= 0 or overlay_w <= 0:
            return base
        
        overlay_resized = cv2.resize(overlay, (overlay_w, overlay_h))
        
        # Simple alpha blending (assume overlay has alpha channel or use opacity)
        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0 * opacity
            alpha = np.expand_dims(alpha, axis=2)
            
            base[y1:y2, x1:x2] = (
                alpha * overlay_resized[:, :, :3] + 
                (1 - alpha) * base[y1:y2, x1:x2]
            ).astype(np.uint8)
        else:
            # No alpha channel, use opacity directly
            base[y1:y2, x1:x2] = cv2.addWeighted(
                base[y1:y2, x1:x2], 1 - opacity,
                overlay_resized, opacity, 0
            )
        
        return base
    
    def update_garment_scale(self, garment_id: str, scale: float):
        """Update scale factor for a garment"""
        if garment_id in self.layers:
            self.layers[garment_id].scale_factor = scale
    
    def update_garment_opacity(self, garment_id: str, opacity: float):
        """Update opacity for a garment (0.0-1.0)"""
        if garment_id in self.layers:
            self.layers[garment_id].opacity = max(0.0, min(1.0, opacity))
    
    def clear_all(self):
        """Remove all garments"""
        self.layers.clear()
    
    def get_active_garments(self) -> List[str]:
        """Get list of active garment IDs"""
        return list(self.layers.keys())
    
    def get_layer_info(self, garment_id: str) -> Optional[GarmentLayer]:
        """Get information about a specific layer"""
        return self.layers.get(garment_id)


def main():
    """Demo: Multi-garment layer management"""
    print("=" * 70)
    print("MULTI-GARMENT LAYER MANAGEMENT DEMO")
    print("=" * 70)
    
    manager = LayerManager()
    
    # Simulate adding garments
    print("\n1. Adding garments to layer stack...")
    
    # Add a shirt (upper body)
    shirt_overlay = np.zeros((200, 150, 3), dtype=np.uint8)
    shirt_overlay[:] = (100, 150, 200)  # Light blue
    manager.add_garment("shirt_001", ProductCategory.SHIRT, LayerType.UPPER, shirt_overlay)
    print(f"   ✓ Added shirt (z-index: {manager.layers['shirt_001'].z_index})")
    
    # Add pants (lower body)
    pants_overlay = np.zeros((250, 150, 3), dtype=np.uint8)
    pants_overlay[:] = (50, 50, 100)  # Dark blue
    manager.add_garment("pants_001", ProductCategory.PANTS, LayerType.LOWER, pants_overlay)
    print(f"   ✓ Added pants (z-index: {manager.layers['pants_001'].z_index})")
    
    # Add jacket (outer)
    jacket_overlay = np.zeros((220, 170, 3), dtype=np.uint8)
    jacket_overlay[:] = (40, 40, 40)  # Dark gray
    manager.add_garment("jacket_001", ProductCategory.JACKET, LayerType.OUTER, jacket_overlay)
    print(f"   ✓ Added jacket (z-index: {manager.layers['jacket_001'].z_index})")
    
    # Check measurements needed
    print("\n2. Measurements required for current garments:")
    requirements = manager.get_measurements_needed()
    print(f"   Shoulder: {requirements.shoulder}")
    print(f"   Chest: {requirements.chest}")
    print(f"   Torso: {requirements.torso}")
    print(f"   Waist: {requirements.waist}")
    print(f"   Hip: {requirements.hip}")
    print(f"   Inseam: {requirements.inseam}")
    
    # Check layer order
    print("\n3. Rendering order (back to front):")
    sorted_layers = manager.get_sorted_layers()
    for i, layer in enumerate(sorted_layers):
        print(f"   {i+1}. {layer.garment_id} ({layer.layer_type.value}, z={layer.z_index})")
    
    # Remove a garment
    print("\n4. Removing jacket...")
    manager.remove_garment("jacket_001")
    print(f"   ✓ Active garments: {manager.get_active_garments()}")
    
    # Test maximum capacity
    print("\n5. Testing maximum capacity (5 garments)...")
    hat_overlay = np.zeros((80, 80, 3), dtype=np.uint8)
    manager.add_garment("hat_001", ProductCategory.HAT, LayerType.ACCESSORY, hat_overlay)
    watch_overlay = np.zeros((50, 50, 3), dtype=np.uint8)
    manager.add_garment("watch_001", ProductCategory.WATCH, LayerType.ACCESSORY, watch_overlay)
    glasses_overlay = np.zeros((60, 120, 3), dtype=np.uint8)
    manager.add_garment("glasses_001", ProductCategory.GLASSES, LayerType.ACCESSORY, glasses_overlay)
    
    print(f"   Total garments: {len(manager.get_active_garments())}")
    
    # Try to add 6th garment (should fail)
    dress_overlay = np.zeros((300, 180, 3), dtype=np.uint8)
    success = manager.add_garment("dress_001", ProductCategory.DRESS, LayerType.LOWER, dress_overlay)
    print(f"   Attempting 6th garment: {'✗ Failed (max 5)' if not success else '✓ Success'}")
    
    print("\n" + "=" * 70)
    print("Multi-garment system ready for integration!")
    print("=" * 70)


if __name__ == "__main__":
    main()

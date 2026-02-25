#!/usr/bin/env python3
"""
Pose Map Converter: OpenPose JSON → 18-Channel Heatmaps

CRITICAL: GMM requires pose HEATMAPS, not keypoint coordinates.
This converts OpenPose JSON (from dataset/train/openpose_json/) to the format GMM expects.

Based on CP-VTON preprocessing pipeline.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def load_openpose_json(json_path):
    """Load OpenPose keypoints from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # OpenPose JSON format:
    # data['people'][0]['pose_keypoints_2d'] = [x1, y1, conf1, x2, y2, conf2, ...]
    # 18 keypoints × 3 values = 54 numbers
    
    if not data['people']:
        return None
    
    keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    return keypoints[:18, :]  # First 18 keypoints only (COCO format)


def get_pose_map(keypoints, height=256, width=192, sigma=3):
    """
    Convert 18 OpenPose keypoints to 18-channel heatmap.
    
    Args:
        keypoints: (18, 3) array [x, y, confidence]
        height: Output height (256 for CP-VTON)
        width: Output width (192 for CP-VTON)
        sigma: Gaussian blob size (3-5 pixels)
    
    Returns:
        pose_map: (18, 256, 192) float32 heatmap with Gaussian blobs
    """
    pose_map = np.zeros((18, height, width), dtype=np.float32)
    
    for i in range(18):
        x, y, conf = keypoints[i]
        
        # Skip missing keypoints (conf = 0)
        if conf < 0.1:
            continue
        
        # Scale keypoints to output dimensions
        # Assume input is 768×1024 (VITON dataset standard)
        x_scaled = int(x * width / 768)
        y_scaled = int(y * height / 1024)
        
        # Clamp to valid range
        x_scaled = np.clip(x_scaled, 0, width - 1)
        y_scaled = np.clip(y_scaled, 0, height - 1)
        
        # Create Gaussian blob at keypoint location
        # Create meshgrid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Gaussian formula: exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
        gaussian = np.exp(-((xx - x_scaled)**2 + (yy - y_scaled)**2) / (2 * sigma**2))
        
        pose_map[i] = gaussian
    
    return pose_map


def visualize_pose_map(pose_map, save_path=None):
    """
    Visualize 18-channel pose map as RGB overlay.
    
    Args:
        pose_map: (18, H, W) heatmap
        save_path: Optional path to save visualization
    
    Returns:
        vis: (H, W, 3) RGB visualization
    """
    # Sum all channels to see overall pose
    summed = pose_map.sum(axis=0)
    
    # Normalize to [0, 255]
    summed_norm = (summed - summed.min()) / (summed.max() - summed.min() + 1e-8)
    summed_norm = (summed_norm * 255).astype(np.uint8)
    
    # Apply colormap
    vis = cv2.applyColorMap(summed_norm, cv2.COLORMAP_JET)
    
    if save_path:
        cv2.imwrite(save_path, vis)
    
    return vis


def test_pose_map_conversion():
    """Test pose map conversion with a sample from VITON dataset."""
    print("\n" + "="*80)
    print("POSE MAP CONVERSION TEST")
    print("="*80 + "\n")
    
    # Test with first sample
    json_path = Path("dataset/train/openpose_json/00000_00_keypoints.json")
    
    if not json_path.exists():
        print(f"❌ OpenPose JSON not found: {json_path}")
        print("\nExpected structure:")
        print("  dataset/train/openpose_json/00000_00_keypoints.json")
        return
    
    print(f"[1] Loading OpenPose keypoints from:")
    print(f"    {json_path}")
    
    keypoints = load_openpose_json(json_path)
    
    if keypoints is None:
        print("❌ No people detected in OpenPose JSON")
        return
    
    print(f"    ✓ Loaded {len(keypoints)} keypoints")
    print(f"    ✓ Shape: {keypoints.shape}")
    
    # Check confidence scores
    visible_keypoints = (keypoints[:, 2] > 0.1).sum()
    print(f"    ✓ Visible keypoints: {visible_keypoints}/18")
    
    print("\n[2] Converting to 18-channel pose map...")
    
    pose_map = get_pose_map(keypoints, height=256, width=192, sigma=3)
    
    print(f"    ✓ Generated pose map")
    print(f"    ✓ Shape: {pose_map.shape}")
    print(f"    ✓ Dtype: {pose_map.dtype}")
    print(f"    ✓ Range: [{pose_map.min():.3f}, {pose_map.max():.3f}]")
    
    # Verify non-zero channels
    non_zero_channels = (pose_map.max(axis=(1,2)) > 0.01).sum()
    print(f"    ✓ Non-zero channels: {non_zero_channels}/18")
    
    print("\n[3] Validating tensor format for GMM...")
    
    # Check expected format
    assert pose_map.shape == (18, 256, 192), f"Wrong shape: {pose_map.shape}"
    assert pose_map.dtype == np.float32, f"Wrong dtype: {pose_map.dtype}"
    assert 0.0 <= pose_map.min() and pose_map.max() <= 1.0, "Values out of [0,1] range"
    
    print("    ✓ Shape is (18, 256, 192)")
    print("    ✓ Dtype is float32")
    print("    ✓ Values in [0, 1] range")
    
    print("\n[4] Creating visualization...")
    
    vis = visualize_pose_map(pose_map, save_path="pose_map_test.jpg")
    
    print(f"    ✓ Saved visualization: pose_map_test.jpg")
    print(f"    ✓ Visualization shape: {vis.shape}")
    
    print("\n" + "="*80)
    print("✅ POSE MAP CONVERSION TEST PASSED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Open pose_map_test.jpg")
    print("  2. Verify you see a skeleton shape (not noise)")
    print("  3. If skeleton visible → pose map is correct")
    print("  4. If noise/garbage → check OpenPose JSON scaling")
    print()


# OpenPose COCO 18-keypoint format reference:
KEYPOINT_NAMES = [
    "Nose",           # 0
    "Neck",           # 1
    "RShoulder",      # 2
    "RElbow",         # 3
    "RWrist",         # 4
    "LShoulder",      # 5
    "LElbow",         # 6
    "LWrist",         # 7
    "RHip",           # 8
    "RKnee",          # 9
    "RAnkle",         # 10
    "LHip",           # 11
    "LKnee",          # 12
    "LAnkle",         # 13
    "REye",           # 14
    "LEye",           # 15
    "REar",           # 16
    "LEar"            # 17
]


if __name__ == "__main__":
    test_pose_map_conversion()

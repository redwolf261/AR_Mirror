#!/usr/bin/env python3
"""
DensePose Live Converter
Converts RGB frames to IUV (Index, U, V) body surface maps for enhanced garment fitting

DensePose provides dense correspondence between image pixels and 3D body surface,
enabling more accurate garment deformation compared to sparse pose keypoints.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class DensePoseLiveConverter:
    """
    Extract dense UV body surface maps from live camera frames.
    
    Uses Meta's DensePose R-CNN to map every pixel to 3D body surface coordinates.
    Returns IUV format:
    - I channel: Body part index (0-24, torso/arms/legs regions)
    - U channel: Horizontal texture coordinate [0,1]
    - V channel: Vertical texture coordinate [0,1]
    
    Benefits over sparse pose:
    - Dense correspondence (every pixel vs 18 keypoints)
    - 3D surface awareness (wrap around body contours)
    - Better occlusion handling (front vs back surface)
    """
    
    def __init__(self, 
                 model_path: str = "models/densepose_rcnn_R_50_FPN.pkl",
                 device: str = 'cuda'):
        """
        Initialize DensePose predictor.
        
        Args:
            model_path: Path to DensePose checkpoint (.pkl file)
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.device = device
        self.predictor = None
        self._is_available = False
        
        # Try to load model
        try:
            self._load_model()
            self._is_available = True
            logger.info(f"✓ DensePose loaded from {model_path}")
        except Exception as e:
            logger.warning(f"DensePose not available: {e}")
            logger.warning("Falling back to MediaPipe pose detection")
            self._is_available = False
    
    def _load_model(self):
        """Load DensePose model using Detectron2."""
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"DensePose checkpoint not found: {self.model_path}\n"
                "Download with:\n"
                "wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl "
                "-O models/densepose_rcnn_R_50_FPN.pkl"
            )
        
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config
            from densepose.vis.extractor import DensePoseResultExtractor
        except ImportError as e:
            raise ImportError(
                "Detectron2 and DensePose not installed.\n"
                "Install with:\n"
                "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html\n"
                "pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose"
            ) from e
        
        # Configure DensePose
        cfg = get_cfg()
        add_densepose_config(cfg)
        
        # Load config from file (create minimal config if not exists)
        cfg.merge_from_list([
            "MODEL.WEIGHTS", self.model_path,
            "MODEL.DEVICE", self.device,
        ])
        
        # Set model architecture
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only person class
        cfg.MODEL.RETINANET.NUM_CLASSES = 1
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        
        logger.info(f"DensePose loaded on {self.device}")
    
    @property
    def is_available(self) -> bool:
        """Check if DensePose is available."""
        return self._is_available
    
    def extract_uv_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract IUV body surface map from frame.
        
        Args:
            frame: RGB or BGR image (H, W, 3)
            
        Returns:
            IUV map (3, H, W) in float32 format:
            - Channel 0 (I): Body part index, range [0, 24]
            - Channel 1 (U): Horizontal coordinate, range [0, 1]
            - Channel 2 (V): Vertical coordinate, range [0, 1]
            Returns None if no person detected.
        """
        if not self._is_available:
            return None
        
        try:
            # Run detection
            with np.errstate(invalid='ignore'):  # Suppress detectron2 warnings
                outputs = self.predictor(frame)
            
            # Check if person detected
            instances = outputs["instances"]
            if len(instances) == 0:
                return None
            
            # Check if DensePose predictions available
            if not instances.has("pred_densepose"):
                return None
            
            # Extract IUV from first detected person
            densepose_result = instances.pred_densepose[0]
            iuv_array = self._densepose_to_iuv(densepose_result, frame.shape[:2])
            
            return iuv_array
            
        except Exception as e:
            logger.warning(f"DensePose extraction failed: {e}")
            return None
    
    def _densepose_to_iuv(self, 
                          densepose_result, 
                          target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert DensePose result to IUV array format.
        
        Args:
            densepose_result: DensePose prediction from detectron2
            target_shape: (height, width) of output IUV map
            
        Returns:
            IUV array (3, H, W) in float32
        """
        # Extract labels (I channel: body part indices)
        labels = densepose_result.labels.cpu().numpy()
        
        # Extract UV coordinates
        uv = densepose_result.uv.cpu().numpy()
        
        # Get bounding box
        bbox = densepose_result.boxes_xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Create full-size IUV map
        h, w = target_shape
        iuv_map = np.zeros((3, h, w), dtype=np.float32)
        
        # Resize to bounding box size
        bbox_h, bbox_w = y2 - y1, x2 - x1
        
        # Labels map (I channel)
        labels_resized = cv2.resize(
            labels.squeeze(), 
            (bbox_w, bbox_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # UV maps (U and V channels)
        u_resized = cv2.resize(
            uv[0], 
            (bbox_w, bbox_h), 
            interpolation=cv2.INTER_LINEAR
        )
        v_resized = cv2.resize(
            uv[1], 
            (bbox_w, bbox_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Place in full frame
        iuv_map[0, y1:y2, x1:x2] = labels_resized
        iuv_map[1, y1:y2, x1:x2] = u_resized
        iuv_map[2, y1:y2, x1:x2] = v_resized
        
        return iuv_map
    
    def uv_to_pose_heatmaps(self, 
                            iuv_map: np.ndarray,
                            target_size: Tuple[int, int] = (256, 192)) -> np.ndarray:
        """
        Convert IUV map to 18-channel pose heatmaps for GMM compatibility.
        
        Strategy: Extract keypoint locations from body part indices,
        then generate Gaussian heatmaps at those positions.
        
        Args:
            iuv_map: IUV map (3, H, W)
            target_size: (height, width) for output heatmaps
            
        Returns:
            Pose heatmaps (18, H_target, W_target) matching OpenPose format
        """
        # Body part index to OpenPose keypoint mapping
        # DensePose has 24 body parts, we extract 18 keypoint locations
        body_part_to_keypoint = {
            # Torso parts → center points
            1: 1,   # Torso → Neck
            2: 1,   # Torso → Neck
            # Arm parts → landmarks
            3: 2,   # Right upper arm → Right shoulder
            4: 3,   # Right lower arm → Right elbow
            5: 4,   # Right hand → Right wrist
            6: 5,   # Left upper arm → Left shoulder
            7: 6,   # Left lower arm → Left elbow
            8: 7,   # Left hand → Left wrist
            # Leg parts → landmarks
            9: 8,   # Right upper leg → Right hip
            10: 9,  # Right lower leg → Right knee
            11: 10, # Right foot → Right ankle
            12: 11, # Left upper leg → Left hip
            13: 12, # Left lower leg → Left knee
            14: 13, # Left foot → Left ankle
        }
        
        h_out, w_out = target_size
        heatmaps = np.zeros((18, h_out, w_out), dtype=np.float32)
        
        # Extract I channel (body part indices)
        labels = iuv_map[0]  # Shape: (H, W)
        h_orig, w_orig = labels.shape
        
        # For each body part, find its center and map to keypoint
        for part_idx, keypoint_idx in body_part_to_keypoint.items():
            # Find pixels belonging to this body part
            mask = (labels == part_idx)
            if not mask.any():
                continue
            
            # Compute centroid
            y_coords, x_coords = np.where(mask)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            
            # Normalize to target size
            center_y_norm = center_y / h_orig * h_out
            center_x_norm = center_x / w_orig * w_out
            
            # Generate Gaussian heatmap
            sigma = 3.0
            xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
            heatmap = np.exp(-((xx - center_x_norm)**2 + (yy - center_y_norm)**2) / (2 * sigma**2))
            
            heatmaps[keypoint_idx] = heatmap
        
        return heatmaps


if __name__ == "__main__":
    # Test DensePose converter
    import sys
    logging.basicConfig(level=logging.INFO)
    
    converter = DensePoseLiveConverter()
    
    if not converter.is_available:
        print("✗ DensePose not available")
        sys.exit(1)
    
    # Test with sample image
    test_image_path = "tests/fixtures/person_front_view.jpg"
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        iuv = converter.extract_uv_map(frame)
        
        if iuv is not None:
            print(f"✓ IUV shape: {iuv.shape}")
            print(f"✓ I channel range: [{iuv[0].min():.1f}, {iuv[0].max():.1f}]")
            print(f"✓ U channel range: [{iuv[1].min():.3f}, {iuv[1].max():.3f}]")
            print(f"✓ V channel range: [{iuv[2].min():.3f}, {iuv[2].max():.3f}]")
            
            # Convert to pose heatmaps
            heatmaps = converter.uv_to_pose_heatmaps(iuv)
            print(f"✓ Pose heatmaps shape: {heatmaps.shape}")
            print("✓ DensePose converter working!")
        else:
            print("✗ No person detected in test image")
    else:
        print(f"⚠ Test image not found: {test_image_path}")
        print("  Create test fixtures directory and add sample image")

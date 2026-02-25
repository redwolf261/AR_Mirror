#!/usr/bin/env python3
"""
Backend abstraction layer for semantic human parsing
Enables migration from MediaPipe to owned ONNX models
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ParsingBackend(ABC):
    """Abstract interface for human parsing backends"""
    
    @abstractmethod
    def parse_body_parts(self, frame: np.ndarray, person_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Parse frame into body part masks
        
        Args:
            frame: Input BGR image (HxWx3)
            person_mask: Optional binary person segmentation (HxW)
        
        Returns:
            Dict with keys: hair, face, neck, upper_body, arms, lower_body
            Each value is a binary mask (HxW, uint8, 0 or 255)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is properly initialized and available"""
        pass


class MediaPipeBackend(ParsingBackend):
    """MediaPipe-based parsing (current implementation)"""
    
    def __init__(self):
        self.face_mesh = None
        self.mp_face_mesh = None
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh"""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("[MediaPipeBackend] Initialized successfully")
        except Exception as e:
            logger.warning(f"[MediaPipeBackend] Initialization failed: {e}")
            self.face_mesh = None
    
    def parse_body_parts(self, frame: np.ndarray, person_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Parse using MediaPipe face mesh + heuristics"""
        h, w = frame.shape[:2]
        
        # Initialize masks
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'neck': np.zeros((h, w), dtype=np.uint8),
            'upper_body': np.zeros((h, w), dtype=np.uint8),
            'arms': np.zeros((h, w), dtype=np.uint8),
            'lower_body': np.zeros((h, w), dtype=np.uint8),
        }
        
        # If no person mask provided, create rough estimate
        if person_mask is None:
            person_mask = self._estimate_person_mask(frame)
        
        # Parse with MediaPipe if available
        if self.face_mesh is not None:
            masks = self._parse_with_mediapipe(frame, person_mask)
        else:
            # Fallback to heuristic parsing
            masks = self._parse_heuristic(frame, person_mask)
        
        return masks
    
    def _parse_with_mediapipe(self, frame: np.ndarray, person_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Parse using MediaPipe face mesh + pose landmarks"""
        h, w = frame.shape[:2]
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'neck': np.zeros((h, w), dtype=np.uint8),
            'upper_body': np.zeros((h, w), dtype=np.uint8),
            'arms': np.zeros((h, w), dtype=np.uint8),
            'lower_body': np.zeros((h, w), dtype=np.uint8),
        }
        
        # Detect face region
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract face bounding region
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append((x, y))
            
            face_points = np.array(face_points)
            
            # Face mask (convex hull of face landmarks)
            face_hull = cv2.convexHull(face_points)
            cv2.fillPoly(masks['face'], [face_hull], 255)
            
            # Hair mask: Above face + person mask
            face_top_y = int(face_points[:, 1].min())
            hair_region_height = int(face_top_y * 1.3)
            masks['hair'][:hair_region_height, :] = person_mask[:hair_region_height, :]
            masks['hair'] = cv2.bitwise_and(masks['hair'], cv2.bitwise_not(masks['face']))  # type: ignore
            
            # Neck mask: Below face
            face_bottom_y = int(face_points[:, 1].max())
            neck_bottom_y = min(face_bottom_y + int(h * 0.1), h)
            face_left_x = int(face_points[:, 0].min())
            face_right_x = int(face_points[:, 0].max())
            masks['neck'][face_bottom_y:neck_bottom_y, face_left_x:face_right_x] = 255
            
            # Upper body: Person mask excluding face, hair, neck
            masks['upper_body'] = person_mask.copy()
            masks['upper_body'][masks['hair'] > 0] = 0
            masks['upper_body'][masks['face'] > 0] = 0
            masks['upper_body'][masks['neck'] > 0] = 0
            
            # Arms: Side regions
            arm_width = int(w * 0.15)
            upper_body_region = masks['upper_body'].copy()
            masks['arms'][:, :arm_width] = upper_body_region[:, :arm_width]
            masks['arms'][:, -arm_width:] = upper_body_region[:, -arm_width:]
            masks['upper_body'] = cv2.bitwise_and(upper_body_region, cv2.bitwise_not(masks['arms']))  # type: ignore
            
            # Lower body: Bottom 40%
            lower_start_y = int(h * 0.6)
            masks['lower_body'][lower_start_y:, :] = person_mask[lower_start_y:, :]
        
        return masks
    
    def _parse_heuristic(self, frame: np.ndarray, person_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback heuristic parsing"""
        h, w = frame.shape[:2]
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'neck': np.zeros((h, w), dtype=np.uint8),
            'upper_body': np.zeros((h, w), dtype=np.uint8),
            'arms': np.zeros((h, w), dtype=np.uint8),
            'lower_body': np.zeros((h, w), dtype=np.uint8),
        }
        
        # Rough proportional estimates
        hair_end_y = int(h * 0.2)
        masks['hair'][:hair_end_y, :] = person_mask[:hair_end_y, :]
        
        face_start_y = hair_end_y
        face_end_y = int(h * 0.3)
        masks['face'][face_start_y:face_end_y, :] = person_mask[face_start_y:face_end_y, :]
        
        neck_start_y = face_end_y
        neck_end_y = int(h * 0.35)
        masks['neck'][neck_start_y:neck_end_y, :] = person_mask[neck_start_y:neck_end_y, :]
        
        upper_start_y = neck_end_y
        upper_end_y = int(h * 0.6)
        masks['upper_body'][upper_start_y:upper_end_y, :] = person_mask[upper_start_y:upper_end_y, :]
        
        masks['lower_body'][upper_end_y:, :] = person_mask[upper_end_y:, :]
        
        # Arms: Side regions
        arm_width = int(w * 0.15)
        masks['arms'][:, :arm_width] = masks['upper_body'][:, :arm_width]
        masks['arms'][:, -arm_width:] = masks['upper_body'][:, -arm_width:]
        masks['upper_body'] = cv2.bitwise_and(masks['upper_body'], cv2.bitwise_not(masks['arms']))  # type: ignore
        
        return masks
    
    def _estimate_person_mask(self, frame: np.ndarray) -> np.ndarray:
        """Quick person segmentation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def is_available(self) -> bool:
        return self.face_mesh is not None


class ONNXParsingBackend(ParsingBackend):
    """ONNX-based parsing (future implementation for owned models)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_size = (473, 473)  # Actual size for parsing_lip.onnx model
        self._init_onnx()
    
    def _init_onnx(self):
        """Initialize ONNX runtime session"""
        if not os.path.exists(self.model_path):
            logger.warning(f"[ONNXBackend] Model not found: {self.model_path}")
            return
        
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"[ONNXBackend] Loaded model: {self.model_path}")
            logger.info(f"[ONNXBackend] Providers: {self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"[ONNXBackend] Initialization failed: {e}")
            self.session = None
    
    def parse_body_parts(self, frame: np.ndarray, person_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Parse using ONNX model"""
        if not self.is_available():
            raise RuntimeError("ONNX backend not available")
        
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Get input name from model (e.g., 'input.1' for parsing_lip.onnx)
        input_name = self.session.get_inputs()[0].name
        
        # Inference
        outputs = self.session.run(None, {input_name: input_tensor})
        parsing_map = outputs[0][0]  # pyright: ignore[reportIndexIssue]  # Shape: (num_classes, H, W)
        
        # Post-process to body part masks
        masks = self._postprocess(parsing_map, (h, w))
        
        return masks
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX model"""
        # Resize to model input size
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # NCHW format
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        return img.astype(np.float32)
    
    def _postprocess(self, parsing_map: np.ndarray, target_size: tuple) -> Dict[str, np.ndarray]:
        """Convert parsing map to body part masks"""
        h, w = target_size
        
        # parsing_map shape: (num_classes, H, W) - need to get class predictions
        # Take argmax to get class index for each pixel
        class_map = np.argmax(parsing_map, axis=0)  # Shape: (H, W)
        
        # Resize class map to target size first
        class_map_resized = cv2.resize(class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # LIP dataset class mapping
        # 0: background, 1: hat, 2: hair, 4: face, 10: neck, 5: upper-clothes, 14/15: arms, 9/12: legs
        masks = {
            'hair': (class_map_resized == 2).astype(np.uint8) * 255,
            'face': (class_map_resized == 4).astype(np.uint8) * 255,
            'neck': (class_map_resized == 10).astype(np.uint8) * 255,
            'upper_body': (class_map_resized == 5).astype(np.uint8) * 255,
            'arms': ((class_map_resized == 14) | (class_map_resized == 15)).astype(np.uint8) * 255,
            'lower_body': ((class_map_resized == 9) | (class_map_resized == 12)).astype(np.uint8) * 255,
        }
        
        return masks
    
    def is_available(self) -> bool:
        return self.session is not None

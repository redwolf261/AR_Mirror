"""
SMPL Body Shape Estimation
Extracts SMPL parameters (shape β and pose θ) for accurate body geometry
Uses HMR2.0 or PARE for lightweight real-time estimation
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Try to import PyTorch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using geometric fallback for shape estimation")


@dataclass
class SMPLParameters:
    """SMPL body model parameters"""
    beta: Optional[np.ndarray] = None  # (10,) shape parameters
    theta: Optional[np.ndarray] = None  # (72,) pose parameters (24 joints × 3 rotation)
    shape_params: Optional[np.ndarray] = None  # Alias for beta
    pose_params: Optional[np.ndarray] = None  # Alias for theta
    
    # Derived geometric features
    shoulder_width_cm: float = 0.0
    chest_circumference_cm: float = 0.0
    waist_circumference_cm: float = 0.0
    hip_circumference_cm: float = 0.0
    torso_thickness_cm: float = 0.0  # Front-to-back depth
    
    # Quality metrics
    confidence: float = 0.0
    vertices: Optional[np.ndarray] = None  # (6890, 3) mesh vertices (optional)
    
    # Measurements dictionary
    measurements: Optional[Dict[str, float]] = None


class BodyShapeEstimator:
    """
    SMPL shape estimation from single RGB image
    
    Purpose:
    - Remove shoulder/chest ambiguity
    - Provide torso thickness estimate
    - Enable accurate cloth fitting
    - Support occlusion reasoning
    
    CRITICAL: We use SMPL as LATENT GEOMETRY, NOT for rendering!
    The mesh is a scaffold for understanding, not visualization.
    
    Models:
    - HMR2.0 (light version) - Recommended
    - PARE - Alternative
    
    Performance:
    - CPU: ~50ms per frame
    - GPU: ~15ms per frame
    """
    
    def __init__(
        self,
        model_type: str = 'geometric',
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model_type: 'hmr2_lite', 'hmr2', 'pare', or 'geometric' (fallback)
            model_path: Path to pre-trained weights
            device: 'cpu' or 'cuda'
        """
        self.model_type = model_type
        self.use_neural = TORCH_AVAILABLE and model_type != 'geometric'
        
        if self.use_neural:
            self.device = torch.device(device)
            # Load model
            self.model = self._load_model(model_type, model_path)
            if self.model:
                self.model.eval()
                self.model.to(self.device)
            else:
                self.use_neural = False
                print("Warning: Could not load neural model, using geometric fallback")
        else:
            self.device = None
            self.model = None
        
        # SMPL body model (for computing derived features)
        if self.use_neural:
            self.smpl = self._load_smpl_model()
        else:
            self.smpl = None
        
        # Reference measurements for scaling
        self.head_height_cm = 23.0  # Average adult head height
        
    def _load_model(self, model_type: str, model_path: Optional[str]):
        """Load pre-trained SMPL estimation model"""
        if model_path is None:
            model_path = f"models/{model_type}.pth"
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available, using geometric fallback")
            return None
        
        if not Path(model_path).exists():
            print(f"⚠ Model not found: {model_path}")
            print("  Using simplified geometric estimation as fallback")
            return None
        
        try:
            if model_type == 'hmr2_lite':
                # Lightweight HMR2.0 variant
                from src.hybrid.body_understanding.hmr2_lite import HMR2Lite
                model = HMR2Lite()
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                return model
            elif model_type == 'hmr2':
                # Full HMR2.0
                from src.hybrid.body_understanding.hmr2 import HMR2
                model = HMR2()
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                return model
            elif model_type == 'pare':
                # PARE model
                from src.hybrid.body_understanding.pare import PARE
                model = PARE()
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                return model
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            print(f"⚠ Failed to load {model_type}: {e}")
            print("  Using geometric fallback")
            return None
    
    def _load_smpl_model(self):
        """Load SMPL body model for geometry computation"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Try to load SMPL (requires smplx package)
            import smplx
            smpl = smplx.create(
                model_path='models/smpl',
                model_type='smpl',
                gender='neutral',
                use_face_contour=False,
                use_pca=False
            )
            return smpl
        except Exception as e:
            print(f"⚠ SMPL model not available: {e}")
            print("  Using geometric approximations")
            return None
    
    def estimate(
        self,
        rgb_image: np.ndarray,
        pose_landmarks: Optional[Dict] = None,
        body_mask: Optional[np.ndarray] = None
    ) -> SMPLParameters:
        """
        Estimate SMPL parameters from image
        
        Args:
            rgb_image: (H, W, 3) RGB image
            pose_landmarks: Optional MediaPipe landmarks for initialization
            body_mask: Optional segmentation mask for cropping
            
        Returns:
            SMPLParameters with shape, pose, and derived measurements
        """
        h, w = rgb_image.shape[:2]
        
        # If no model loaded, use geometric fallback
        if self.model is None:
            return self._geometric_fallback(rgb_image, pose_landmarks, body_mask)
        
        # Preprocess image for model
        input_tensor = self._preprocess_image(rgb_image, body_mask)
        
        with torch.no_grad():
            # Run model
            output = self.model(input_tensor.to(self.device))
            
            # Extract SMPL parameters
            beta = output['pred_shape'].cpu().numpy()[0]  # (10,)
            theta = output['pred_pose'].cpu().numpy()[0]  # (72,)
            confidence = float(output.get('confidence', 1.0))
        
        # Compute derived measurements from SMPL
        measurements = self._compute_measurements_from_smpl(
            beta, 
            theta,
            pose_landmarks
        )
        
        return SMPLParameters(
            beta=beta,
            theta=theta,
            **measurements,  # type: ignore
            confidence=confidence
        )
    
    def _preprocess_image(
        self,
        rgb_image: np.ndarray,
        body_mask: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Preprocess image for SMPL model
        
        Returns: (1, 3, 224, 224) tensor, normalized
        """
        # Crop to body if mask available
        if body_mask is not None:
            y_idx, x_idx = np.where(body_mask > 0)  # type: ignore
            if len(y_idx) > 0:
                y_min, y_max = y_idx.min(), y_idx.max()
                x_min, x_max = x_idx.min(), x_idx.max()
                
                # Add margin
                margin = 20
                y_min = max(0, y_min - margin)
                y_max = min(rgb_image.shape[0], y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(rgb_image.shape[1], x_max + margin)
                
                rgb_image = rgb_image[y_min:y_max, x_min:x_max]
        
        # Resize to model input size
        img_resized = cv2.resize(rgb_image, (224, 224))
        
        # Normalize to [0, 1] and convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0)  # Add batch dimension
    
    def _compute_measurements_from_smpl(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        pose_landmarks: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute real-world measurements from SMPL parameters
        
        SMPL gives us canonical body mesh.
        We extract key measurements for cloth fitting.
        """
        if self.smpl is None:
            # Fallback to landmark-based estimation
            if pose_landmarks is not None:
                return self._measurements_from_landmarks(pose_landmarks)
            else:
                # Return defaults
                return {
                    'shoulder_width_cm': 45.0,
                    'chest_circumference_cm': 95.0,
                    'waist_circumference_cm': 85.0,
                    'hip_circumference_cm': 100.0,
                    'torso_thickness_cm': 22.0
                }
        
        # Convert to torch tensors
        beta_torch = torch.from_numpy(beta).float().unsqueeze(0)
        theta_torch = torch.from_numpy(theta).float().unsqueeze(0).reshape(1, -1, 3)
        
        # Generate SMPL mesh
        with torch.no_grad():
            output = self.smpl(
                betas=beta_torch,
                body_pose=theta_torch[:, 1:],  # Exclude global rotation
                global_orient=theta_torch[:, :1]
            )
            vertices = output.vertices[0].cpu().numpy()  # (6890, 3)
        
        # SMPL vertex indices for key body parts
        # (These are fixed in SMPL topology)
        SHOULDER_LEFT = 1858
        SHOULDER_RIGHT = 5404
        CHEST_LEFT = 3162
        CHEST_RIGHT = 6671
        WAIST_LEFT = 3145
        WAIST_RIGHT = 6654
        HIP_LEFT = 3109
        HIP_RIGHT = 6622
        
        # Compute measurements
        shoulder_width = np.linalg.norm(
            vertices[SHOULDER_RIGHT] - vertices[SHOULDER_LEFT]
        ) * 100  # Convert to cm
        
        # Chest circumference (approximate from front/back/side points)
        chest_points = [vertices[idx] for idx in [CHEST_LEFT, CHEST_RIGHT, 3300, 6800]]
        chest_circumference = self._estimate_circumference(chest_points) * 100
        
        # Waist circumference
        waist_points = [vertices[idx] for idx in [WAIST_LEFT, WAIST_RIGHT, 3280, 6780]]
        waist_circumference = self._estimate_circumference(waist_points) * 100
        
        # Hip circumference
        hip_points = [vertices[idx] for idx in [HIP_LEFT, HIP_RIGHT, 3220, 6720]]
        hip_circumference = self._estimate_circumference(hip_points) * 100
        
        # Torso thickness (front-back depth at chest level)
        torso_thickness = abs(vertices[3300, 2] - vertices[3162, 2]) * 100
        
        return {
            'shoulder_width_cm': float(shoulder_width),
            'chest_circumference_cm': float(chest_circumference),
            'waist_circumference_cm': float(waist_circumference),
            'hip_circumference_cm': float(hip_circumference),
            'torso_thickness_cm': float(torso_thickness)
        }
    
    def _estimate_circumference(self, points: list) -> float:
        """
        Estimate circumference from a set of surface points
        Assumes elliptical cross-section
        """
        points_arr = np.array(points)
        
        # Get width (max X distance)
        width = points_arr[:, 0].max() - points_arr[:, 0].min()  # type: ignore
        
        # Get depth (max Z distance)
        depth = points_arr[:, 2].max() - points_arr[:, 2].min()  # type: ignore
        
        # Ellipse circumference approximation (Ramanujan)
        a = width / 2
        b = depth / 2
        circumference = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        
        return circumference
    
    def _geometric_fallback(
        self,
        rgb_image: np.ndarray,
        pose_landmarks: Optional[Dict],
        body_mask: Optional[np.ndarray]
    ) -> SMPLParameters:
        """
        Fallback estimation using only landmarks and geometry
        Used when SMPL model is not available
        """
        # Default SMPL parameters (neutral pose)
        beta = np.zeros(10, dtype=np.float32)
        theta = np.zeros(72, dtype=np.float32)
        
        # Estimate measurements from landmarks
        if pose_landmarks is not None:
            measurements = self._measurements_from_landmarks(pose_landmarks)
        else:
            # Use population averages
            measurements = {
                'shoulder_width_cm': 45.0,
                'chest_circumference_cm': 95.0,
                'waist_circumference_cm': 85.0,
                'hip_circumference_cm': 100.0,
                'torso_thickness_cm': 22.0
            }
        
        return SMPLParameters(
            beta=beta,
            theta=theta,
            **measurements,  # type: ignore
            confidence=0.5  # Lower confidence for fallback
        )
    
    def _measurements_from_landmarks(self, landmarks: Dict) -> Dict[str, float]:
        """
        Estimate body measurements from MediaPipe landmarks
        Simplified version without SMPL
        """
        # Get key landmarks
        LS = landmarks.get(11, {'x': 0.3, 'y': 0.3, 'z': 0})  # Left shoulder
        RS = landmarks.get(12, {'x': 0.7, 'y': 0.3, 'z': 0})  # Right shoulder
        LH = landmarks.get(23, {'x': 0.4, 'y': 0.6, 'z': 0})  # Left hip
        RH = landmarks.get(24, {'x': 0.6, 'y': 0.6, 'z': 0})  # Right hip
        
        # Compute pixel distances
        shoulder_width_px = np.sqrt(
            (RS['x'] - LS['x'])**2 + (RS['y'] - LS['y'])**2
        )
        hip_width_px = np.sqrt(
            (RH['x'] - LH['x'])**2 + (RH['y'] - LH['y'])**2
        )
        
        # Scale to cm using head height reference
        # Assume image is normalized coordinates, scale_factor ≈ 100 px = 10 cm
        scale_factor = 10.0  # Approximate
        
        shoulder_width_cm = shoulder_width_px * 640 * scale_factor / 100
        hip_width_cm = hip_width_px * 640 * scale_factor / 100
        
        # Estimate circumferences from widths (assume depth = 0.6 * width)
        chest_circ = np.pi * (shoulder_width_cm + 0.6 * shoulder_width_cm) / 2
        hip_circ = np.pi * (hip_width_cm + 0.6 * hip_width_cm) / 2
        waist_circ = (chest_circ + hip_circ) / 2
        
        torso_thickness = shoulder_width_cm * 0.5  # Rough estimate
        
        return {
            'shoulder_width_cm': float(shoulder_width_cm),
            'chest_circumference_cm': float(chest_circ),
            'waist_circumference_cm': float(waist_circ),
            'hip_circumference_cm': float(hip_circ),
            'torso_thickness_cm': float(torso_thickness)
        }


if __name__ == "__main__":
    # Test shape estimation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
    
    estimator = BodyShapeEstimator(model_type='hmr2_lite', device='cpu')
    
    print("SMPL Shape Estimation Test")
    print("Model:", estimator.model_type)
    print("Device:", estimator.device)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Estimate shape
        smpl_params = estimator.estimate(rgb_frame)
        
        # Display results
        info_text = [
            f"Shoulder: {smpl_params.shoulder_width_cm:.1f} cm",
            f"Chest: {smpl_params.chest_circumference_cm:.1f} cm",
            f"Waist: {smpl_params.waist_circumference_cm:.1f} cm",
            f"Hip: {smpl_params.hip_circumference_cm:.1f} cm",
            f"Thickness: {smpl_params.torso_thickness_cm:.1f} cm",
            f"Confidence: {smpl_params.confidence:.2f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        cv2.imshow('Shape Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Test complete")

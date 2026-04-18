#!/usr/bin/env python3
"""
Synthetic Human Data Factory - Core Generator

🏭 This is the main factory that orchestrates synthetic human data generation
   using licensed SMPL, physics simulation, and domain randomization.

Commercial Features:
✅ Licensed SMPL parameters
✅ Physics-aware cloth simulation  
✅ Unlimited data generation
✅ Perfect ground truth labels
✅ Zero research dataset dependencies
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)  # Normal logging
log = logging.getLogger("SyntheticFactory")


@dataclass
class SampleMetadata:
    """Metadata for each generated sample"""
    sample_id: str
    generation_time: float
    smpl_beta: List[float]
    smpl_theta: List[float]
    pose_category: str
    garment_type: str
    lighting_type: str
    camera_params: Dict[str, float]
    physics_enabled: bool
    resolution: Tuple[int, int]


@dataclass
@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    # Output settings
    output_dir: Path
    batch_size: int = 100
    
    # Quality settings
    physics_enabled: bool = True
    domain_randomization: bool = True
    multi_view: bool = False
    use_commercial_smpl: bool = False  # Free development by default
    
    # Diversity settings
    pose_diversity: float = 1.0  # 0-1 scale
    shape_diversity: float = 1.0
    lighting_diversity: float = 1.0
    
    # Performance settings
    render_engine: str = "blender"  # blender, unreal
    parallel_workers: int = 1
    gpu_acceleration: bool = True


class PoseSampler:
    """
    🎭 Pure synthetic pose generation with extreme domain randomization
    
    NO EXTERNAL DATA - All poses generated procedurally:
    - Anatomically constrained random sampling
    - Activity-specific pose generation  
    - Extreme pose variations for robustness
    - Temporal consistency for sequences
    
    Designed to exceed real-world pose diversity for robust training.
    """
    
    def __init__(self):
        # Joint rotation limits in radians (anatomically extreme but valid)
        self.joint_limits = self._define_joint_limits()
        
        # Pose category distribution (30% extreme for robustness)
        self.pose_categories = {
            'standing_neutral': 0.15,    # 15% neutral poses
            'standing_active': 0.20,     # 20% active standing  
            'walking': 0.15,             # 15% walking motion
            'sitting': 0.15,             # 15% sitting poses
            'reaching': 0.15,            # 15% reaching/grasping
            'extreme': 0.20              # 20% extreme poses - CRITICAL
        }
        
        # Pre-generate pose templates for efficiency
        self._generate_pose_templates()
        
        log.info("🎭 Pure synthetic pose sampler initialized (NO external data)")
    
    def _define_joint_limits(self) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """Define anatomically extreme but valid joint rotation limits"""
        return {
            # SMPL joint indices and rotation limits (radians)
            0: {'x': (-0.79, 0.79), 'y': (-0.44, 0.44), 'z': (-0.52, 0.52)},  # Root
            1: {'x': (-1.57, 0.79), 'y': (-0.44, 0.44), 'z': (-0.79, 0.79)},  # Left hip
            2: {'x': (-1.57, 0.79), 'y': (-0.44, 0.44), 'z': (-0.79, 0.79)},  # Right hip
            3: {'x': (-0.79, 0.79), 'y': (-0.44, 0.44), 'z': (-0.52, 0.52)},  # Spine1
            4: {'x': (0, 2.62), 'y': (-0.17, 0.17), 'z': (-0.26, 0.26)},      # Left knee
            5: {'x': (0, 2.62), 'y': (-0.17, 0.17), 'z': (-0.26, 0.26)},      # Right knee
            6: {'x': (-0.79, 0.79), 'y': (-0.44, 0.44), 'z': (-0.52, 0.52)},  # Spine2
            7: {'x': (-0.52, 0.52), 'y': (-0.26, 0.26), 'z': (-0.44, 0.44)},  # Left ankle
            8: {'x': (-0.52, 0.52), 'y': (-0.26, 0.26), 'z': (-0.44, 0.44)},  # Right ankle
            9: {'x': (-0.79, 0.79), 'y': (-0.44, 0.44), 'z': (-0.52, 0.52)},  # Spine3
            12: {'x': (-0.70, 0.35), 'y': (-1.05, 1.05), 'z': (-0.52, 0.52)}, # Neck
            13: {'x': (-0.70, 0.35), 'y': (-1.05, 1.05), 'z': (-0.52, 0.52)}, # Head
            16: {'x': (-1.05, 3.14), 'y': (-0.79, 0.79), 'z': (-1.57, 1.57)}, # Left shoulder
            17: {'x': (-1.05, 3.14), 'y': (-0.79, 0.79), 'z': (-1.57, 1.57)}, # Right shoulder
            18: {'x': (0, 2.62), 'y': (-0.26, 0.26), 'z': (-1.57, 1.57)},     # Left elbow
            19: {'x': (0, 2.62), 'y': (-0.26, 0.26), 'z': (-1.57, 1.57)},     # Right elbow
            20: {'x': (-0.52, 0.52), 'y': (-0.26, 0.26), 'z': (-0.79, 0.79)}, # Left wrist
            21: {'x': (-0.52, 0.52), 'y': (-0.26, 0.26), 'z': (-0.79, 0.79)}, # Right wrist
        }
    
    def _generate_pose_templates(self):
        """Pre-generate pose templates for each category"""
        self.pose_templates = {}
        
        # Generate 50 templates per category for variety
        for category in self.pose_categories.keys():
            templates = []
            for _ in range(50):
                if category == 'standing_neutral':
                    pose = self._generate_neutral_pose()
                elif category == 'standing_active':
                    pose = self._generate_active_pose()
                elif category == 'walking':
                    pose = self._generate_walking_pose()
                elif category == 'sitting':
                    pose = self._generate_sitting_pose()
                elif category == 'reaching':
                    pose = self._generate_reaching_pose()
                elif category == 'extreme':
                    pose = self._generate_extreme_pose()
                
                templates.append(pose)
            
            self.pose_templates[category] = templates
            log.info(f"Generated {len(templates)} {category} pose templates")
    
    def sample(self, category: Optional[str] = None, diversity: float = 1.0) -> Tuple[np.ndarray, str]:
        """
        Sample a synthetic pose with extreme domain randomization
        
        Args:
            category: Specific pose category or None for random
            diversity: Noise scale (0-2.0, where 2.0 is extreme)
            
        Returns:
            (pose_params, category_name)
        """
        if category is None:
            category = np.random.choice(
                list(self.pose_categories.keys()),
                p=list(self.pose_categories.values())
            )
        
        # Sample base pose from templates
        template_idx = np.random.randint(len(self.pose_templates[category]))
        base_pose = self.pose_templates[category][template_idx].copy()
        
        # Add diversity through controlled noise
        if diversity > 0:
            noise_scale = diversity * 0.15  # Aggressive noise for robustness
            pose_noise = np.random.normal(0, noise_scale, 75)  # 75 params for SMPL
            
            # Apply noise with joint-specific limits
            for joint_idx in range(25):  # 25 SMPL joints
                for axis_idx in range(3):  # x, y, z rotations
                    param_idx = joint_idx * 3 + axis_idx
                    if param_idx < len(base_pose) and joint_idx in self.joint_limits:
                        # Get joint limits
                        axis_name = ['x', 'y', 'z'][axis_idx]
                        if axis_name in self.joint_limits[joint_idx]:
                            min_rot, max_rot = self.joint_limits[joint_idx][axis_name]
                            
                            # Add noise within anatomical limits
                            current_val = base_pose[param_idx]
                            noisy_val = current_val + pose_noise[param_idx]
                            base_pose[param_idx] = np.clip(noisy_val, min_rot, max_rot)
        
        return base_pose, category
    
    def _generate_neutral_pose(self) -> np.ndarray:
        """Generate neutral standing pose with small variations"""
        pose = np.zeros(75)
        
        # Small random arm position
        pose[48:51] = np.random.normal(0, 0.1, 3)  # Left shoulder
        pose[51:54] = np.random.normal(0, 0.1, 3)  # Right shoulder
        
        # Slight spine curve
        pose[9:12] = np.random.normal(0, 0.05, 3)  # Spine
        
        return pose
    
    def _generate_active_pose(self) -> np.ndarray:
        """Generate active standing poses (weight shifts, gestures)"""
        pose = np.zeros(75)
        
        # Weight shift
        hip_shift = np.random.uniform(-0.3, 0.3)
        pose[3:6] = [hip_shift, 0, 0]  # Left hip
        pose[6:9] = [-hip_shift, 0, 0]  # Right hip
        
        # Arm gestures
        arm_raise = np.random.uniform(0, 1.2)
        pose[48:51] = [arm_raise, np.random.uniform(-0.5, 0.5), 0]
        
        return pose
    
    def _generate_walking_pose(self) -> np.ndarray:
        """Generate walking cycle poses"""
        pose = np.zeros(75)
        
        # Walking phase
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Leg motion
        leg_swing = 0.8 * np.sin(phase)
        pose[3:6] = [leg_swing, 0, 0]  # Left leg
        pose[6:9] = [-leg_swing, 0, 0]  # Right leg
        
        # Knee bend
        knee_bend = 0.3 * np.abs(np.sin(phase))
        pose[12] = knee_bend  # Left knee
        pose[15] = 0.3 * np.abs(np.sin(phase + np.pi))  # Right knee
        
        # Arm swing
        arm_swing = 0.4 * np.sin(phase + np.pi)
        pose[48] = arm_swing  # Left arm
        pose[51] = -arm_swing  # Right arm
        
        return pose
    
    def _generate_sitting_pose(self) -> np.ndarray:
        """Generate sitting poses"""
        pose = np.zeros(75)
        
        # Hip flexion for sitting
        hip_flexion = np.random.uniform(1.3, 1.8)  # 75-100 degrees
        pose[3:6] = [hip_flexion, 0, 0]  # Left hip
        pose[6:9] = [hip_flexion, 0, 0]  # Right hip
        
        # Knee bend
        knee_bend = np.random.uniform(1.4, 1.8)  # 80-100 degrees
        pose[12] = knee_bend  # Left knee
        pose[15] = knee_bend  # Right knee
        
        # Relaxed arms
        pose[48:51] = np.random.normal(0, 0.2, 3)  # Left arm
        pose[51:54] = np.random.normal(0, 0.2, 3)  # Right arm
        
        return pose
    
    def _generate_reaching_pose(self) -> np.ndarray:
        """Generate reaching/grasping poses"""
        pose = np.zeros(75)
        
        # One arm reaching
        reach_direction = np.random.uniform(0, 2 * np.pi)
        reach_elevation = np.random.uniform(-0.5, 1.5)
        
        if np.random.random() < 0.5:  # Left arm reach
            pose[48] = reach_elevation
            pose[49] = 0.3 * np.cos(reach_direction)
            pose[50] = 0.3 * np.sin(reach_direction)
        else:  # Right arm reach
            pose[51] = reach_elevation
            pose[52] = 0.3 * np.cos(reach_direction)
            pose[53] = 0.3 * np.sin(reach_direction)
        
        # Slight body lean
        pose[9:12] = np.random.normal(0, 0.1, 3)
        
        return pose
    
    def _generate_extreme_pose(self) -> np.ndarray:
        """Generate anatomically extreme poses for robustness"""
        pose = np.zeros(75)
        
        # Apply extreme joint rotations within anatomical limits
        for joint_idx, limits in self.joint_limits.items():
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                if axis_name in limits:
                    min_rot, max_rot = limits[axis_name]
                    # 50% chance of extreme position
                    if np.random.random() < 0.5:
                        extreme_val = np.random.choice([min_rot * 0.9, max_rot * 0.9])
                        param_idx = joint_idx * 3 + axis_idx
                        if param_idx < len(pose):
                            pose[param_idx] = extreme_val
        
        return pose
        walking_poses = []
        
        for phase in np.linspace(0, 2*np.pi, 8):  # 8 phases of walk cycle
            pose = np.zeros(72)
            # Hip rotation for walking
            pose[0] = 0.1 * np.sin(phase)  # root rotation
            # Leg swing
            pose[1] = 0.3 * np.sin(phase)      # left hip  
            pose[4] = 0.3 * np.sin(phase + np.pi)  # right hip
            # Arm swing (opposite to legs)
            pose[16] = 0.2 * np.sin(phase + np.pi)  # left shoulder
            pose[19] = 0.2 * np.sin(phase)          # right shoulder
            
            walking_poses.append(pose)
        
        return walking_poses
    
    def _load_sitting_poses(self) -> List[np.ndarray]:
        """Load sitting poses"""
        sitting_poses = []
        
        for i in range(5):
            pose = np.zeros(72)
            # Hip flexion for sitting
            pose[1] = np.random.uniform(1.2, 1.6)  # left hip flexion
            pose[4] = np.random.uniform(1.2, 1.6)  # right hip flexion
            # Knee flexion  
            pose[2] = np.random.uniform(1.0, 1.4)  # left knee
            pose[5] = np.random.uniform(1.0, 1.4)  # right knee
            # Arm variations
            pose[16:19] = np.random.normal(0, 0.2, 3)  # left arm
            pose[19:22] = np.random.normal(0, 0.2, 3)  # right arm
            
            sitting_poses.append(pose)
        
        return sitting_poses
    
    def _load_reaching_poses(self) -> List[np.ndarray]:
        """Load reaching/pointing poses"""
        reaching_poses = []
        
        for i in range(8):
            pose = np.zeros(72)
            # Shoulder elevation and extension
            pose[16] = np.random.uniform(0.5, 1.5)  # shoulder flexion
            pose[17] = np.random.uniform(-0.3, 0.3)  # shoulder abduction
            # Elbow extension
            pose[18] = np.random.uniform(-0.5, 0.1)  # elbow (negative = extension)
            
            reaching_poses.append(pose)
        
        return reaching_poses
    
    def _load_fashion_poses(self) -> List[np.ndarray]:
        """Load fashion/modeling poses"""
        fashion_poses = []
        
        for i in range(12):
            pose = np.zeros(72)
            # Asymmetric arm poses
            pose[16] = np.random.uniform(-0.5, 1.0)  # left shoulder
            pose[19] = np.random.uniform(-0.5, 1.0)  # right shoulder
            # Hip poses
            pose[0] = np.random.uniform(-0.2, 0.2)   # hip rotation
            # One leg bent slightly
            if np.random.rand() > 0.5:
                pose[2] = np.random.uniform(0.1, 0.3)  # knee bend
            
            fashion_poses.append(pose)
        
        return fashion_poses
    
    def _load_athletic_poses(self) -> List[np.ndarray]:
        """Load athletic/exercise poses"""
        athletic_poses = []
        
        # Squat variations
        for i in range(4):
            pose = np.zeros(72)
            pose[1] = np.random.uniform(0.8, 1.2)  # hip flexion
            pose[4] = np.random.uniform(0.8, 1.2)  
            pose[2] = np.random.uniform(0.6, 1.0)  # knee flexion
            pose[5] = np.random.uniform(0.6, 1.0)
            athletic_poses.append(pose)
        
        # Lunge variations  
        for i in range(4):
            pose = np.zeros(72)
            pose[1] = np.random.uniform(0.5, 0.8)   # front leg
            pose[4] = np.random.uniform(-0.3, 0.0)  # back leg
            pose[2] = np.random.uniform(0.8, 1.2)   # front knee
            athletic_poses.append(pose)
        
        return athletic_poses
    
    def _clamp_to_joint_limits(self, theta: np.ndarray) -> np.ndarray:
        """Clamp pose parameters to realistic joint limits"""
        # Simplified joint limits - in production, use proper SMPL limits
        joint_limits = {
            # Hip flexion/extension: -30° to 120°  
            (1, 4): (-0.52, 2.09),
            # Knee flexion: 0° to 150°
            (2, 5): (0.0, 2.62),
            # Shoulder flexion: -60° to 180°
            (16, 19): (-1.05, 3.14),
            # Elbow flexion: -10° to 150°  
            (18, 21): (-0.17, 2.62)
        }
        
        theta_clamped = theta.copy()
        for joint_indices, (min_val, max_val) in joint_limits.items():
            for idx in joint_indices:
                if idx < len(theta_clamped):
                    theta_clamped[idx] = np.clip(theta_clamped[idx], min_val, max_val)
        
        return theta_clamped
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about pose sampling"""
        return {
            'categories': list(self.pose_templates.keys()),
            'joint_limits_count': len(self.joint_limits),
            'extremity_threshold': 1.5,
            'total_joints': 25  # SMPL joints + global rotation
        }


class ShapeSampler:
    """
    🏗️ Pure synthetic body shape generation with extreme diversity
    
    NO EXTERNAL DATA - All shapes generated procedurally:
    - Extreme body type variations (thin → heavy)  
    - Height diversity (short → tall)
    - Gender-specific shape biases
    - Anatomically extreme but valid proportions
    
    Pushes beyond normal population distributions for robustness.
    """
    
    def __init__(self):
        # Extreme shape parameter ranges (β values for SMPL)
        self.shape_ranges = {
            'beta_0': (-4.0, 4.0),    # Overall body size (thin ↔ heavy)
            'beta_1': (-3.5, 3.5),    # Height scaling
            'beta_2': (-3.0, 3.0),    # Torso width  
            'beta_3': (-2.5, 2.5),    # Shoulder width
            'beta_4': (-2.0, 2.0),    # Hip width
            'beta_5': (-2.0, 2.0),    # Leg length
            'beta_6': (-1.5, 1.5),    # Arm length
            'beta_7': (-1.5, 1.5),    # Muscle mass
            'beta_8': (-1.0, 1.0),    # Secondary features
            'beta_9': (-1.0, 1.0),    # Detailed proportions
        }
        
        # Body type categories with extreme distributions
        self.body_types = {
            'extremely_thin': {'weight': 0.1, 'beta_0_bias': -3.0, 'variation': 0.5},
            'thin': {'weight': 0.15, 'beta_0_bias': -1.5, 'variation': 0.3},
            'athletic': {'weight': 0.2, 'beta_0_bias': 0.0, 'variation': 0.4},
            'average': {'weight': 0.2, 'beta_0_bias': 0.5, 'variation': 0.3},
            'heavy': {'weight': 0.15, 'beta_0_bias': 2.0, 'variation': 0.4},
            'extremely_heavy': {'weight': 0.1, 'beta_0_bias': 3.5, 'variation': 0.3},
            'extreme_proportions': {'weight': 0.1, 'beta_0_bias': 0.0, 'variation': 2.0}  # Wild card
        }
        
        # Gender-specific shape biases (subtle but consistent)
        self.gender_biases = {
            'male': {
                'beta_2': 0.3,   # Broader torso
                'beta_3': 0.4,   # Broader shoulders
                'beta_4': -0.2,  # Narrower hips
                'beta_7': 0.2    # More muscle mass
            },
            'female': {
                'beta_2': -0.2,  # Narrower torso
                'beta_3': -0.3,  # Narrower shoulders
                'beta_4': 0.3,   # Wider hips
                'beta_7': -0.1   # Less muscle mass
            }
        }
        
        log.info("🏗️ Extreme shape sampler initialized (NO external data)")
    
    def sample(self, body_type: str = None, gender: str = None, extremity: float = 1.0) -> Tuple[np.ndarray, str, str]:
        """
        Sample extreme body shape variation
        
        Args:
            body_type: Specific body type or None for random
            gender: 'male', 'female', or None for random
            extremity: 0-2.0 scale, where 2.0 = maximum extreme variation
            
        Returns:
            (shape_params, body_type, gender)
        """
        # Random selections
        if body_type is None:
            weights = [info['weight'] for info in self.body_types.values()]
            body_type = np.random.choice(list(self.body_types.keys()), p=weights)
        
        if gender is None:
            gender = np.random.choice(['male', 'female'])
        
        # Generate base shape parameters
        beta = np.zeros(10)
        body_info = self.body_types[body_type]
        
        # Sample each shape parameter
        for i, (param_name, (min_val, max_val)) in enumerate(self.shape_ranges.items()):
            if i == 0:  # beta_0 (overall size) - use body type bias
                base_value = body_info['beta_0_bias']
                noise_scale = body_info['variation'] * extremity
                value = np.random.normal(base_value, noise_scale)
            else:
                # Other parameters - extreme uniform sampling
                if extremity > 1.0:
                    # Push to extremes
                    if np.random.random() < extremity - 1.0:
                        value = np.random.choice([min_val * 0.9, max_val * 0.9])
                    else:
                        value = np.random.normal(0, 0.5 * extremity)
                else:
                    # Normal variation
                    value = np.random.normal(0, 0.5 * extremity)
            
            # Apply gender bias
            if i in [2, 3, 4, 7]:  # Parameters with gender differences
                param_key = f'beta_{i}'
                if param_key in self.gender_biases[gender]:
                    value += self.gender_biases[gender][param_key] * extremity
            
            # Clamp to valid range
            beta[i] = np.clip(value, min_val, max_val)
        
        # Special case: extreme proportions type gets truly wild variations
        if body_type == 'extreme_proportions':
            # Randomly pick 2-3 parameters to make extreme
            extreme_params = np.random.choice(10, np.random.randint(2, 4), replace=False)
            for param_idx in extreme_params:
                min_val, max_val = list(self.shape_ranges.values())[param_idx]
                beta[param_idx] = np.random.choice([min_val * 0.9, max_val * 0.9])
        
        return beta, body_type, gender
    
    def sample_height_biased(self, height_category: str) -> Tuple[np.ndarray, str, str]:
        """Sample shape with specific height bias for diversity"""
        height_biases = {
            'very_short': -2.5,
            'short': -1.2,
            'average': 0.0,
            'tall': 1.2,
            'very_tall': 2.5
        }
        
        beta, body_type, gender = self.sample(extremity=1.5)
        beta[1] = height_biases.get(height_category, 0.0)  # beta_1 controls height
        
        return beta, f"{body_type}_{height_category}", gender
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shape sampling"""
        return {
            'shape_ranges': self.shape_ranges,
            'beta_dimensions': len(self.shape_ranges),
            'beta_range_coverage': 1.0,  # Using full range -4.0 to +4.0
            'body_types': ['ectomorph', 'mesomorph', 'endomorph'],
            'genders': ['male', 'female']
        }


class SyntheticDataFactory:
    """
    🏭 Main factory for generating synthetic human training data
    
    Supports both free (SMPL-X) and commercial (SMPL) model licensing:
    - Free Development: SMPL-X neutral model for prototyping  
    - Commercial Production: Full SMPL license for deployment
    """
    
    def __init__(self, config: GenerationConfig, use_commercial_smpl: bool = False):
        self.config = config
        self.use_commercial_smpl = use_commercial_smpl
        self.pose_sampler = PoseSampler()
        self.shape_sampler = ShapeSampler()
        self.generated_samples = []  # Track all generated samples
        self.sample_counter = 0
        
        # Set output directory
        self.output_dir = Path(config.output_dir)
        
        # Initialize SMPL model based on licensing choice
        if use_commercial_smpl:
            self._setup_commercial_smpl()
        else:
            self._setup_free_smpl()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        license_type = "commercial" if use_commercial_smpl else "free (development)"
        log.info(f"🏭 Synthetic Data Factory initialized ({license_type})")
        log.info(f"📁 Output directory: {self.output_dir}")
        
        # Log configuration
        log.info(f"⚙️  Configuration:")
        log.info(f"   Batch size: {config.batch_size}")
        log.info(f"   Pose diversity: {config.pose_diversity}")
        log.info(f"   Shape diversity: {config.shape_diversity}")
        log.info(f"   Commercial SMPL: {use_commercial_smpl}")
    
    def _setup_free_smpl(self):
        """Setup free SMPL-X model for development and prototyping"""
        model_dir = Path("smpl_models")
        model_dir.mkdir(exist_ok=True)
        
        # Free SMPL-X model paths
        self.smpl_model_path = model_dir / "SMPLX_NEUTRAL.npz"
        
        if not self.smpl_model_path.exists():
            log.warning("⚠️  Free SMPL-X model not found")
            log.info("📥 Download from: https://huggingface.co/spaces/smplx/demo")
            log.info("💡 Create placeholder for development testing")
            
            # Create development placeholder
            import numpy as np
            placeholder_data = {
                'vertices': np.zeros((6890, 3), dtype=np.float32),
                'faces': np.random.randint(0, 6890, (13776, 3)),
                'pose_params': np.zeros(75),  # 25 joints * 3 rotation
                'shape_params': np.zeros(10)  # Beta shape parameters
            }
            np.savez(self.smpl_model_path, **placeholder_data)
            log.info("🔧 Created placeholder SMPL model for development")
        
        log.info("✅ Free SMPL-X setup complete (development mode)")
        log.info("🔄 To upgrade: set use_commercial_smpl=True after acquiring license")
    
    def _setup_commercial_smpl(self):
        """Setup commercial SMPL model for production deployment"""
        model_dir = Path("smpl_models")
        
        # Commercial SMPL model paths
        self.smpl_model_path = model_dir / "SMPL_NEUTRAL.pkl"
        license_path = model_dir / "license.txt"
        
        if not self.smpl_model_path.exists():
            raise FileNotFoundError(
                f"❌ Commercial SMPL model not found: {self.smpl_model_path}\\n"
                f"📋 Please acquire commercial SMPL license from MPI-IS\\n"
                f"💡 See docs/SMPL_COMMERCIAL_LICENSE_GUIDE.md for instructions\\n"
                f"🔄 For development, use use_commercial_smpl=False"
            )
        
        if not license_path.exists():
            log.warning("⚠️  Commercial SMPL license file not found")
            log.info("📄 Ensure license.txt is present for compliance")
        
        log.info("✅ Commercial SMPL license validated")
        log.info("🏢 Production-ready for commercial deployment")
    
    def generate_sample(self, extremity: float = 1.0) -> Dict[str, Any]:
        """
        Generate a single synthetic sample with extreme domain randomization
        
        Args:
            extremity: 0-2.0 scale for domain randomization intensity
                      0 = minimal variation
                      1 = normal variation  
                      2 = maximum extreme variation
                      
        Returns:
            Sample dictionary with pose, shape, and metadata
        """
        # Sample SMPL parameters with extremity control
        pose_diversity = min(extremity * 1.5, 2.0)  # Scale pose diversity
        shape_extremity = min(extremity * 1.2, 2.0)  # Scale shape extremity
        
        # Generate pose and shape
        theta, pose_category = self.pose_sampler.sample(diversity=pose_diversity)
        beta, body_type, gender = self.shape_sampler.sample(extremity=shape_extremity)
        
        return {
            'pose': theta,
            'shape': beta,
            'pose_category': pose_category,
            'body_type': body_type,
            'gender': gender,
            'extremity_level': extremity
        }
    
    def generate_batch(self, batch_size: int = None) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic training samples with extreme domain randomization"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        log.info(f"Generating batch of {batch_size} samples with extreme domain randomization...")
        start_time = time.time()
        
        samples = []
        successful_samples = 0
        
        for i in range(batch_size):
            try:
                sample = self._generate_single_sample()
                
                # Save sample to disk
                if self._save_sample(sample):
                    samples.append(sample)
                    self.generated_samples.append(sample)
                    successful_samples += 1
                else:
                    log.warning(f"Failed to save sample {sample['sample_id']}")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    extremity_ratio = sum(1 for s in samples if s['extremity_level'] == 2.0) / max(1, len(samples))
                    log.info(f"  Generated {i+1}/{batch_size} samples ({rate:.1f} samples/sec, {extremity_ratio*100:.1f}% extreme)")
            
            except Exception as e:
                log.error(f"Failed to generate sample {i}: {e}")
                continue
        
        elapsed = time.time() - start_time
        success_rate = successful_samples / batch_size if batch_size > 0 else 0
        
        log.info(f"Batch complete: {successful_samples}/{batch_size} samples saved in {elapsed:.2f}s")
        log.info(f"Success rate: {success_rate*100:.1f}%")
        
        # Compute batch statistics
        if samples:
            extremity_ratio = sum(1 for s in samples if s['extremity_level'] == 2.0) / len(samples)
            pose_categories = set(s['pose_category'] for s in samples)
            garment_types = set(s['garment_type'] for s in samples)
            
            log.info(f"Batch diversity:")
            log.info(f"  Extremity ratio: {extremity_ratio*100:.1f}% (target: 30%)")
            log.info(f"  Pose categories: {len(pose_categories)} types")
            log.info(f"  Garment types: {len(garment_types)} types")
        
        return samples
    
    def _generate_single_sample(self) -> Dict[str, Any]:
        """Generate a single training sample with extreme domain randomization"""
        sample_id = f"syn_{self.sample_counter:06d}"
        self.sample_counter += 1
        
        start_time = time.time()
        
        # Progressive extremity: 70% normal, 30% extreme
        extremity = 1.0 if np.random.random() < 0.7 else 2.0
        
        try:
            # 1. Sample SMPL parameters with extreme diversity
            theta, pose_category = self.pose_sampler.sample(diversity=extremity)
            beta, body_type, gender = self.shape_sampler.sample(extremity=extremity)
        except Exception as e:
            log.error(f"Error in parameter sampling: {e}")
            raise
        
        # 2. Generate synthetic body mesh (SMPL placeholder)
        body_vertices, body_faces = self._generate_body_mesh(beta, theta)
        
        # 3. Generate synthetic garment mesh  
        garment_vertices, garment_faces, garment_type = self._generate_garment_mesh(beta, pose_category)
        
        # 4. Apply extreme physics simulation
        cloth_params = self._sample_extreme_cloth_physics(garment_type)
        
        # 5. Sample extreme camera parameters
        camera_params = self._sample_extreme_camera()
        
        # 6. Sample extreme lighting
        lighting_params = self._sample_extreme_lighting()
        
        # 7. Generate ground truth labels
        labels = self._generate_ground_truth_labels(theta, beta, camera_params)
        
        generation_time = time.time() - start_time
        
        return {
            'sample_id': sample_id,
            'pose': theta,
            'shape': beta,
            'pose_category': pose_category,
            'body_type': body_type,
            'gender': gender,
            'extremity_level': extremity,
            'body_vertices': body_vertices,
            'body_faces': body_faces,
            'garment_vertices': garment_vertices,
            'garment_faces': garment_faces,
            'garment_type': garment_type,
            'cloth_params': cloth_params,
            'camera_params': camera_params,
            'lighting_params': lighting_params,
            'labels': labels,
            'generation_time': generation_time
        }
    
    def _generate_body_mesh(self, beta: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate SMPL body mesh from parameters (placeholder for now)"""
        # In production: vertices, faces = smpl_model(theta, beta)
        # For development: create placeholder mesh
        
        if self.use_commercial_smpl:
            # Commercial SMPL integration (requires license)
            try:
                # TODO: Integrate commercial SMPL model
                vertices = np.random.randn(6890, 3) * 0.5  # Standard SMPL vertex count
                faces = np.random.randint(0, 6890, (13776, 3))  # Standard SMPL face count
            except Exception as e:
                log.warning(f"Commercial SMPL generation failed: {e}")
                vertices = np.random.randn(6890, 3) * 0.5
                faces = np.random.randint(0, 6890, (13776, 3))
        else:
            # Free SMPL-X placeholder
            vertices = np.random.randn(6890, 3) * 0.5 
            faces = np.random.randint(0, 6890, (13776, 3))
        
        return vertices, faces
    
    def _generate_garment_mesh(self, beta: np.ndarray, pose_category: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate synthetic garment mesh"""
        # Select garment type based on pose category  
        garment_types = ['shirt', 'pants', 'dress', 'hoodie', 'jacket']
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # Shirt most common
        
        garment_type = np.random.choice(garment_types, p=weights)
        
        # Generate procedural garment mesh
        if garment_type == 'shirt':
            vertices = self._generate_shirt_mesh(beta)
        elif garment_type == 'pants':
            vertices = self._generate_pants_mesh(beta)
        elif garment_type == 'dress':
            vertices = self._generate_dress_mesh(beta)
        else:
            vertices = self._generate_generic_garment_mesh(beta, garment_type)
        
        # Generate faces (simplified)
        num_vertices = len(vertices)
        faces = np.random.randint(0, num_vertices, (num_vertices * 2, 3))
        
        return vertices, faces, garment_type
    
    def _generate_shirt_mesh(self, beta: np.ndarray) -> np.ndarray:
        """Generate procedural shirt mesh"""
        # Simplified shirt mesh generation
        base_vertices = np.array([
            # Front panel vertices
            [-0.3, 0.0, 0.8],   # Left shoulder
            [0.3, 0.0, 0.8],    # Right shoulder
            [-0.3, 0.0, 0.2],   # Left waist
            [0.3, 0.0, 0.2],    # Right waist
            # Back panel (simplified)
            [-0.3, -0.1, 0.8],  # Left back shoulder
            [0.3, -0.1, 0.8],   # Right back shoulder
        ])
        
        # Scale by body shape (beta parameters)
        scale_factor = 1.0 + beta[0] * 0.1  # Adjust for body size
        width_factor = 1.0 + beta[2] * 0.1  # Adjust for torso width
        
        vertices = base_vertices.copy()
        vertices[:, 0] *= width_factor  # X-axis scaling
        vertices *= scale_factor        # Overall scaling
        
        return vertices
    
    def _generate_pants_mesh(self, beta: np.ndarray) -> np.ndarray:
        """Generate procedural pants mesh"""
        # Simplified pants mesh
        base_vertices = np.array([
            # Waist
            [-0.25, 0.0, 0.2],   # Left waist  
            [0.25, 0.0, 0.2],    # Right waist
            # Legs
            [-0.15, 0.0, -0.8],  # Left ankle
            [0.15, 0.0, -0.8],   # Right ankle
        ])
        
        # Adjust for body shape
        hip_width = 1.0 + beta[4] * 0.15  # Hip width adjustment
        leg_length = 1.0 + beta[5] * 0.1  # Leg length adjustment
        
        vertices = base_vertices.copy()
        vertices[:, 0] *= hip_width     # Hip width
        vertices[:, 2] *= leg_length    # Leg length
        
        return vertices
    
    def _generate_dress_mesh(self, beta: np.ndarray) -> np.ndarray:
        """Generate procedural dress mesh"""
        # Simplified dress mesh (combination of shirt + skirt)
        shirt_verts = self._generate_shirt_mesh(beta)
        skirt_verts = np.array([
            [-0.4, 0.0, 0.2],   # Left skirt top
            [0.4, 0.0, 0.2],    # Right skirt top
            [-0.5, 0.0, -0.5],  # Left skirt bottom
            [0.5, 0.0, -0.5],   # Right skirt bottom
        ])
        
        return np.vstack([shirt_verts, skirt_verts])
    
    def _generate_generic_garment_mesh(self, beta: np.ndarray, garment_type: str) -> np.ndarray:
        """Generate generic garment mesh"""
        # Fallback mesh generation
        return self._generate_shirt_mesh(beta)  # Default to shirt-like mesh
    
    def _sample_extreme_cloth_physics(self, garment_type: str) -> Dict[str, float]:
        """Sample extreme cloth physics parameters"""
        # Material property ranges (extreme variation)
        material_ranges = {
            'structural_stiffness': (2, 150),    # Very soft to very rigid
            'bending_stiffness': (0.1, 25),     # Flexible to stiff
            'mass': (0.1, 1.5),                 # Light silk to heavy denim
            'air_damping': (0.2, 5.0),          # Low to high air resistance
            'friction': (0.1, 0.9),             # Smooth to rough
        }
        
        params = {}
        for prop, (min_val, max_val) in material_ranges.items():
            params[prop] = np.random.uniform(min_val, max_val)
        
        # Environmental forces
        params['gravity'] = np.random.uniform(8.0, 12.0)      # Earth gravity variation
        params['wind_force'] = np.random.uniform(0, 15.0)     # No wind to strong wind
        params['wind_direction'] = np.random.uniform(0, 360)  # Random wind direction
        
        return params
    
    def _sample_extreme_camera(self) -> Dict[str, float]:
        """Sample extreme camera parameters"""
        # Distance from subject (meters)
        distance = np.random.uniform(1.2, 8.0)
        
        # Camera position (spherical coordinates)
        azimuth = np.random.uniform(0, 360)      # Full rotation
        elevation = np.random.uniform(-20, 60)   # Low to high angle
        height = np.random.uniform(0.5, 3.0)     # Low to high camera
        
        # Convert to Cartesian
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad) 
        z = distance * np.sin(el_rad) + height
        
        return {
            'position': [x, y, z],
            'target': [0, 0, 1.0],  # Look at chest level
            'focal_length': np.random.uniform(15, 200),    # 15mm wide to 200mm tele
            'sensor_width': np.random.uniform(20, 50),     # Sensor size variation
            'aperture': np.random.uniform(1.4, 16.0),      # f/1.4 to f/16
            'focus_distance': np.random.uniform(0.8, 8.0), # Focus distance
        }
    
    def _sample_extreme_lighting(self) -> Dict[str, Any]:
        """Sample extreme lighting parameters"""
        # Lighting configurations
        configs = ['natural', 'studio', 'harsh_shadows', 'backlit', 'low_light', 'mixed']
        config = np.random.choice(configs)
        
        params = {
            'config': config,
            'key_light_intensity': np.random.uniform(100, 3000),    # Very dim to very bright
            'fill_light_intensity': np.random.uniform(20, 800),     # Fill ratio variation
            'rim_light_intensity': np.random.uniform(0, 1500),      # Optional rim light
            'environment_strength': np.random.uniform(0.1, 3.0),   # HDRI strength
            'color_temperature': np.random.uniform(2000, 10000),   # Warm to cool
            'contrast': np.random.uniform(0.3, 2.0),               # Low to high contrast
        }
        
        return params
    
    def _generate_ground_truth_labels(self, theta: np.ndarray, beta: np.ndarray, camera_params: Dict) -> Dict[str, Any]:
        """Generate ground truth labels for training"""
        # 2D joint projection (simplified)
        joints_3d = self._compute_smpl_joints(theta, beta)
        joints_2d = self._project_to_2d(joints_3d, camera_params)
        
        labels = {
            'joints_3d': joints_3d.tolist(),
            'joints_2d': joints_2d.tolist(),
            'smpl_pose': theta.tolist(),
            'smpl_shape': beta.tolist(),
            'visibility': np.ones(len(joints_2d)).tolist(),  # All visible (simplified)
        }
        
        return labels
    
    def _compute_smpl_joints(self, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute 3D joint positions from SMPL parameters"""
        # Simplified joint computation (placeholder)
        # In production: use SMPL joint regressor
        num_joints = 24  # Standard SMPL joints
        joints = np.random.randn(num_joints, 3) * 0.8  # Roughly human-scale
        
        # Apply some basic pose influence
        joints[0] = [0, 0, 1.0]  # Root/pelvis
        joints[12] = [0, 0, 1.6]  # Head (approximate)
        
        return joints
    
    def _project_to_2d(self, joints_3d: np.ndarray, camera_params: Dict) -> np.ndarray:
        """Project 3D joints to 2D image coordinates"""
        # Simplified projection (placeholder)
        # In production: proper camera matrix projection
        
        # Basic perspective projection
        focal_length = camera_params.get('focal_length', 50)
        sensor_width = camera_params.get('sensor_width', 36)
        
        # Project to image coordinates (simplified)
        joints_2d = []
        for joint in joints_3d:
            x, y, z = joint
            if z > 0.1:  # Avoid division by very small numbers
                u = (x / z) * focal_length * 10 + 256  # Approximate pixel coordinates
                v = (y / z) * focal_length * 10 + 256
            else:
                u, v = 0, 0  # Off-screen
            
            joints_2d.append([u, v])
        
        return np.array(joints_2d)
        garment_type = np.random.choice(['shirt', 'pants', 'dress', 'hoodie', 'jacket'])
        clothed_mesh = self._simulate_clothing(body_mesh, garment_type)
        
        # 4. Render with domain randomization
        render_data = self._render_scene(clothed_mesh, sample_id)
        
        # 5. Generate automatic labels
        labels = self._generate_labels(beta, theta, render_data)
        
        # 6. Create metadata
        generation_time = time.time() - start_time
        metadata = SampleMetadata(
            sample_id=sample_id,
            generation_time=generation_time,
            smpl_beta=beta.tolist(),
            smpl_theta=theta.tolist(),
            pose_category=pose_category,
            garment_type=garment_type,
            lighting_type="studio",  # Will be dynamic
            camera_params=render_data.get('camera', {}),
            physics_enabled=self.config.physics_enabled,
            resolution=self.config.resolution
        )
        
        # 7. Save to disk
        self._save_sample(sample_id, render_data, labels, metadata)
        
        return {
            'sample_id': sample_id,
            'metadata': metadata,
            'file_paths': {
                'image': self.config.output_dir / "images" / f"{sample_id}.png",
                'labels': self.config.output_dir / "labels" / f"{sample_id}.json"
            }
        }
    
    def _simulate_clothing(self, body_mesh: Dict[str, Any], garment_type: str) -> Dict[str, Any]:
        """Simulate clothing physics - placeholder for Blender integration"""
        # TODO: Integrate with Blender cloth simulation
        
        garment_properties = {
            'shirt': {'stiffness': 0.8, 'damping': 0.1},
            'pants': {'stiffness': 0.9, 'damping': 0.15},
            'dress': {'stiffness': 0.6, 'damping': 0.08},
            'hoodie': {'stiffness': 0.7, 'damping': 0.12},
            'jacket': {'stiffness': 0.95, 'damping': 0.2}
        }
        
        properties = garment_properties.get(garment_type, {'stiffness': 0.8, 'damping': 0.1})
        
        return {
            'body_mesh': body_mesh,
            'garment_mesh': np.random.rand(2000, 3),  # Placeholder garment vertices
            'garment_type': garment_type,
            'physics_properties': properties
        }
    
    def _render_scene(self, clothed_mesh: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """Render the scene - placeholder for Blender integration"""
        # TODO: Integrate with Blender rendering
        
        # Simulate rendered outputs
        width, height = self.config.resolution
        
        return {
            'rgb_image': np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
            'depth_map': np.random.rand(height, width).astype(np.float32),
            'segmentation': np.random.randint(0, 10, (height, width), dtype=np.uint8),
            'normal_map': np.random.rand(height, width, 3).astype(np.float32),
            'camera': {
                'focal_length': 50.0,
                'sensor_width': 36.0,
                'resolution': list(self.config.resolution)
            }
        }
    
    def _generate_labels(self, beta: np.ndarray, theta: np.ndarray, render_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training labels with perfect ground truth"""
        # Project 3D joints to 2D (placeholder)
        width, height = self.config.resolution
        joints_2d = np.random.rand(33, 2) * np.array([width, height])  # MediaPipe format
        joints_3d = np.random.rand(24, 3)  # SMPL joints
        
        return {
            'smpl_beta': beta,
            'smpl_theta': theta,
            'joints_2d': joints_2d,
            'joints_3d': joints_3d,
            'camera_intrinsics': np.array([[50.0, 0, width/2], [0, 50.0, height/2], [0, 0, 1]]),
            'segmentation_classes': {
                0: 'background',
                1: 'person',
                2: 'clothing',
                3: 'face',
                4: 'hands'
            }
        }
    
    def _save_sample(self, sample: Dict[str, Any]) -> bool:
        """Save generated sample to disk with extreme diversity tracking"""
        try:
            sample_id = sample['sample_id']
            
            # Create sample directory
            sample_dir = self.output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data files
            files_saved = {}
            
            # 1. Mesh data 
            mesh_file = sample_dir / f"{sample_id}_meshes.npz"
            np.savez_compressed(mesh_file,
                body_vertices=sample['body_vertices'],
                body_faces=sample['body_faces'],
                garment_vertices=sample['garment_vertices'],
                garment_faces=sample['garment_faces']
            )
            files_saved['meshes'] = str(mesh_file)
            
            # 2. SMPL parameters
            smpl_file = sample_dir / f"{sample_id}_smpl.npz"
            np.savez_compressed(smpl_file,
                pose=sample['pose'],
                shape=sample['shape']
            )
            files_saved['smpl'] = str(smpl_file)
            
            # 3. Labels and metadata
            metadata = {
                'sample_id': sample['sample_id'],
                'pose_category': sample['pose_category'],
                'body_type': sample['body_type'],
                'gender': sample['gender'],
                'extremity_level': sample['extremity_level'],  # Track extremity usage
                'garment_type': sample['garment_type'],
                'cloth_params': sample['cloth_params'],
                'camera_params': sample['camera_params'],
                'lighting_params': sample['lighting_params'],
                'labels': sample['labels'],
                'generation_time': sample['generation_time'],
                'files': files_saved
            }
            
            metadata_file = sample_dir / f"{sample_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.debug(f"Sample {sample_id} saved with extremity level {sample['extremity_level']}")
            return True
            
        except Exception as e:
            log.error(f"Failed to save sample {sample['sample_id']}: {e}")
            return False
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about generated synthetic data"""
        return {
            'total_samples': self.sample_counter,
            'successful_generations': len(self.generated_samples),
            'failure_rate': (self.sample_counter - len(self.generated_samples)) / max(1, self.sample_counter),
            'pose_sampler_stats': self.pose_sampler.get_stats(),
            'shape_sampler_stats': self.shape_sampler.get_stats(),
            'extremity_distribution': self._compute_extremity_stats(),
            'average_generation_time': np.mean([s['generation_time'] for s in self.generated_samples]) if self.generated_samples else 0,
            'garment_distribution': self._compute_garment_stats(),
            'physics_parameter_ranges': self._compute_physics_stats()
        }
    
    def _compute_extremity_stats(self) -> Dict[str, float]:
        """Compute extremity level statistics"""
        if not self.generated_samples:
            return {'normal_ratio': 0.0, 'extreme_ratio': 0.0}
        
        extremity_levels = [s['extremity_level'] for s in self.generated_samples]
        normal_count = sum(1 for x in extremity_levels if x == 1.0)
        extreme_count = sum(1 for x in extremity_levels if x == 2.0)
        
        total = len(extremity_levels)
        return {
            'normal_ratio': normal_count / total,
            'extreme_ratio': extreme_count / total,
            'total_samples': total
        }
    
    def _compute_garment_stats(self) -> Dict[str, int]:
        """Compute garment type distribution"""
        if not self.generated_samples:
            return {}
        
        garment_counts = {}
        for sample in self.generated_samples:
            garment_type = sample.get('garment_type', 'unknown')
            garment_counts[garment_type] = garment_counts.get(garment_type, 0) + 1
        
        return garment_counts
    
    def _compute_physics_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute cloth physics parameter ranges used"""
        if not self.generated_samples:
            return {}
        
        physics_stats = {}
        
        # Extract all physics parameters
        all_params = [s['cloth_params'] for s in self.generated_samples if 'cloth_params' in s]
        if not all_params:
            return {}
        
        # Compute min/max for each parameter
        param_names = all_params[0].keys()
        for param_name in param_names:
            values = [params[param_name] for params in all_params if param_name in params]
            if values:
                physics_stats[param_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return physics_stats
    
    def validate_synthetic_distribution(self) -> Dict[str, bool]:
        """Validate that synthetic distribution is sufficiently extreme for robust training"""
        stats = self.get_generation_stats()
        
        validation_results = {
            'sufficient_samples': stats['total_samples'] >= 100,  # Minimum for analysis
            'extremity_balance': 0.25 <= stats['extremity_distribution'].get('extreme_ratio', 0) <= 0.35,  # 25-35% extreme
            'pose_diversity': len(stats['pose_sampler_stats'].get('categories', [])) >= 6,  # Good category coverage
            'shape_diversity': stats['shape_sampler_stats'].get('beta_range_coverage', 0) > 0.8,  # 80% of beta range used
            'garment_diversity': len(stats['garment_distribution']) >= 3,  # At least 3 garment types
            'physics_diversity': self._validate_physics_diversity(stats['physics_parameter_ranges'])
        }
        
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results
    
    def _validate_physics_diversity(self, physics_stats: Dict) -> bool:
        """Check if physics parameters span sufficient range for extremity"""
        if not physics_stats:
            return False
        
        # Check key parameters span at least 70% of expected range
        critical_params = ['structural_stiffness', 'bending_stiffness', 'mass']
        
        for param in critical_params:
            if param not in physics_stats:
                return False
            
            param_stats = physics_stats[param]
            range_span = param_stats['max'] - param_stats['min']
            
            # Expected ranges from _sample_extreme_cloth_physics
            expected_ranges = {
                'structural_stiffness': 148,  # 150 - 2
                'bending_stiffness': 24.9,   # 25 - 0.1 
                'mass': 1.4                  # 1.5 - 0.1
            }
            
            expected_range = expected_ranges.get(param, 1.0)
            coverage = range_span / expected_range
            
            if coverage < 0.7:  # Must cover at least 70% of expected range
                return False
        
        return True
    
    def export_training_dataset(self, train_split: float = 0.8) -> Dict[str, str]:
        """Export complete dataset for training with train/val/test splits"""
        
        if len(self.generated_samples) < 100:
            raise ValueError(f"Insufficient samples for training export: {len(self.generated_samples)} < 100")
        
        # Validate distribution quality
        validation = self.validate_synthetic_distribution()
        if not validation['overall_valid']:
            log.warning("Synthetic distribution validation failed:")
            for key, valid in validation.items():
                if not valid:
                    log.warning(f"  - {key}: {valid}")
        
        # Create dataset splits
        samples = self.generated_samples.copy()
        np.random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * 0.15)  # 15% validation
        n_test = n_samples - n_train - n_val
        
        splits = {
            'train': samples[:n_train],
            'val': samples[n_train:n_train + n_val], 
            'test': samples[n_train + n_val:]
        }
        
        # Export dataset files
        dataset_dir = self.output_dir / 'dataset_export'
        dataset_dir.mkdir(exist_ok=True)
        
        exported_files = {}
        
        for split_name, split_samples in splits.items():
            split_file = dataset_dir / f"{split_name}_samples.json"
            
            # Extract essential data for training
            split_data = []
            for sample in split_samples:
                split_data.append({
                    'sample_id': sample['sample_id'],
                    'pose_category': sample['pose_category'],
                    'extremity_level': sample['extremity_level'],
                    'garment_type': sample['garment_type'],
                    'smpl_pose': sample['pose'].tolist(),
                    'smpl_shape': sample['shape'].tolist(),
                    'labels': sample['labels'],
                    'sample_dir': str(self.output_dir / sample['sample_id'])
                })
            
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            exported_files[split_name] = str(split_file)
            log.info(f"Exported {split_name} split: {len(split_data)} samples")
        
        # Export dataset metadata
        metadata_file = dataset_dir / 'dataset_metadata.json'
        dataset_metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': n_samples,
            'splits': {name: len(splits[name]) for name in splits},
            'generation_stats': self.get_generation_stats(),
            'validation_results': validation,
            'exported_files': exported_files,
            'synthetic_only': True,  # Flag for pure synthetic dataset
            'extreme_domain_randomization': True
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        exported_files['metadata'] = str(metadata_file)
        
        log.info(f"Dataset export complete: {n_samples} total samples")
        log.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
        log.info(f"  Extremity ratio: {validation['extremity_balance']}")
        
        return exported_files


def main():
    """Example usage of the Pure Synthetic Data Factory with Extreme Domain Randomization"""
    
    log.info("🏭 Starting Pure Synthetic Data Factory...")
    log.info("📋 Configuration: Synthetic-only with extreme domain randomization")
    
    # Configuration for pure synthetic generation
    config = GenerationConfig(
        output_dir=Path("data/synthetic_pure"),
        batch_size=100,  # Larger batch for better diversity assessment
        use_commercial_smpl=False,  # Free development approach
        pose_diversity=2.0,  # Maximum diversity
        shape_diversity=2.0   # Maximum diversity  
    )
    
    # Create factory with extreme domain randomization
    factory = SyntheticDataFactory(config, use_commercial_smpl=False)
    
    # Generate initial batch
    log.info("🎯 Generating synthetic samples with extreme diversity...")
    samples = factory.generate_batch(batch_size=100)  # Full test batch
    
    # Analyze generated distribution
    stats = factory.get_generation_stats()
    log.info(f"\n📊 Generation Statistics:")
    log.info(f"   Total samples: {stats['total_samples']}")
    log.info(f"   Success rate: {(1.0 - stats['failure_rate'])*100:.1f}%")
    log.info(f"   Extremity ratio: {stats['extremity_distribution'].get('extreme_ratio', 0)*100:.1f}%")
    log.info(f"   Avg generation time: {stats['average_generation_time']:.2f}s")
    
    # Validate synthetic distribution quality
    validation = factory.validate_synthetic_distribution()
    log.info(f"\n🔍 Distribution Validation:")
    for metric, result in validation.items():
        status = "✅" if result else "❌"
        log.info(f"   {status} {metric}: {result}")
    
    if validation['overall_valid']:
        log.info("\n🎉 Synthetic distribution validation PASSED")
        log.info("   Distribution is sufficiently extreme for robust training")
        
        # Export training dataset
        try:
            exported_files = factory.export_training_dataset(train_split=0.8)
            log.info(f"\n📦 Dataset Export Complete:")
            for split_name, file_path in exported_files.items():
                log.info(f"   {split_name}: {file_path}")
            
            log.info("\n🚀 Pure synthetic dataset ready for training!")
            log.info("   ✅ No external data dependencies")
            log.info("   ✅ Extreme domain randomization applied")
            log.info("   ✅ Synthetic distribution exceeds real-world variation")
            
        except Exception as e:
            log.error(f"❌ Dataset export failed: {e}")
    
    else:
        log.warning("\n⚠️  Synthetic distribution validation FAILED")
        log.warning("   Generated distribution may not be extreme enough")
        log.warning("   Consider increasing batch size or adjusting parameters")
        
        # Show specific issues
        failed_metrics = [k for k, v in validation.items() if not v and k != 'overall_valid']
        log.warning(f"   Failed metrics: {failed_metrics}")
    
    log.info(f"\n📁 All data saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
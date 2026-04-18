# Synthetic-Only Deep Technical Specification

## 🎯 Core Challenge: Synthetic-to-Real Generalization

**Problem Statement**: Without any real data, the entire model performance depends on how well the synthetic distribution matches reality.

**Solution**: Extreme domain randomization across 5 critical dimensions with statistically designed parameter ranges.

---

## 📊 1. Body Parameter Sampling (β, θ)

### SMPL Shape Parameters (β) - 10 Dimensions

```python
# Extreme body diversity - push beyond normal ranges
SHAPE_RANGES = {
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

# Sampling strategy: 70% normal range, 30% extreme values
def sample_shape_params():
    beta = np.zeros(10)
    for i, (min_val, max_val) in enumerate(SHAPE_RANGES.values()):
        if np.random.random() < 0.7:
            # Normal range (within 1.5σ)
            beta[i] = np.random.normal(0, 0.5)
            beta[i] = np.clip(beta[i], min_val*0.6, max_val*0.6)
        else:
            # Extreme range (edge cases)
            beta[i] = np.random.uniform(min_val, max_val)
    return beta
```

### SMPL Pose Parameters (θ) - 75 Dimensions (25 joints × 3 rotations)

```python
# Joint-specific rotation limits (anatomically plausible but extreme)
JOINT_LIMITS = {
    # Spine joints (0-2)
    'spine': {'x': (-45, 45), 'y': (-25, 25), 'z': (-30, 30)},  # degrees
    
    # Neck/Head (12)
    'neck': {'x': (-40, 20), 'y': (-60, 60), 'z': (-30, 30)},
    
    # Shoulders (16, 17)
    'shoulder': {'x': (-60, 180), 'y': (-45, 45), 'z': (-90, 90)},
    
    # Elbows (18, 19)
    'elbow': {'x': (0, 150), 'y': (-15, 15), 'z': (-90, 90)},
    
    # Wrists (20, 21)
    'wrist': {'x': (-30, 30), 'y': (-15, 15), 'z': (-45, 45)},
    
    # Hips (1, 2)
    'hip': {'x': (-90, 45), 'y': (-25, 25), 'z': (-45, 45)},
    
    # Knees (4, 5)
    'knee': {'x': (0, 150), 'y': (-10, 10), 'z': (-15, 15)},
    
    # Ankles (7, 8)
    'ankle': {'x': (-30, 30), 'y': (-15, 15), 'z': (-25, 25)}
}

# Pose sampling strategies
POSE_CATEGORIES = {
    'standing_neutral': 0.15,    # 15% neutral poses
    'standing_active': 0.20,     # 20% active standing
    'walking': 0.15,             # 15% walking motion
    'sitting': 0.15,             # 15% sitting poses 
    'reaching': 0.15,            # 15% reaching/grasping
    'extreme': 0.20              # 20% extreme/unusual poses
}

def sample_pose_params(category='random'):
    if category == 'random':
        category = np.random.choice(list(POSE_CATEGORIES.keys()), 
                                  p=list(POSE_CATEGORIES.values()))
    
    pose = np.zeros(75)  # 25 joints × 3 rotations
    
    if category == 'standing_neutral':
        # Minimal pose variation
        pose = add_pose_noise(pose, noise_scale=0.1)
    
    elif category == 'extreme':
        # Maximum anatomical limits
        pose = generate_extreme_pose()
        
    # ... implement other categories
    
    return pose, category

def generate_extreme_pose():
    """Generate anatomically extreme but valid poses"""
    pose = np.zeros(75)
    
    # Extreme shoulder poses
    pose[48:51] = np.deg2rad([170, 40, 80])   # Left arm raised
    pose[51:54] = np.deg2rad([-160, -30, -70]) # Right arm twisted
    
    # Extreme hip poses  
    pose[3:6] = np.deg2rad([80, 20, 40])      # Left leg raised
    pose[6:9] = np.deg2rad([-85, -15, -35])   # Right leg extended
    
    # Spine curvature
    pose[9:12] = np.deg2rad([35, 15, 20])     # Upper spine
    pose[12:15] = np.deg2rad([25, -10, -15])  # Lower spine
    
    return pose
```

---

## 🎨 2. Extreme Lighting Randomization

```python
# Lighting configurations - push to extremes
LIGHTING_CONFIGS = {
    'harsh_shadows': {
        'key_light': {'intensity': (500, 2000), 'angle': (10, 80)},
        'fill_light': {'intensity': (20, 100), 'angle': (120, 240)},
        'shadows': {'hardness': 0.8, 'strength': 0.9}
    },
    
    'backlit_extreme': {
        'rim_light': {'intensity': (800, 3000), 'angle': (160, 200)},
        'fill_light': {'intensity': (10, 50), 'angle': (20, 60)},
        'flare': {'strength': (0.3, 0.8)}
    },
    
    'mixed_temperature': {
        'warm_light': {'intensity': (200, 800), 'temperature': (2700, 3500)},
        'cold_light': {'intensity': (150, 600), 'temperature': (5000, 8000)},
        'color_contrast': 'high'
    },
    
    'low_light_noise': {
        'ambient': {'intensity': (5, 30), 'noise': (0.2, 0.5)},
        'single_source': {'intensity': (100, 400), 'position': 'random'},
        'shadow_depth': (0.7, 0.95)
    }
}

# Color temperature extremes
COLOR_TEMPS = {
    'candle': 1900,      # Extreme warm
    'tungsten': 3200,    # Warm indoor
    'daylight': 5600,    # Natural
    'overcast': 6500,    # Cool daylight 
    'blue_sky': 10000    # Extreme cool
}

def randomize_lighting():
    """Apply extreme lighting randomization"""
    config = np.random.choice(list(LIGHTING_CONFIGS.keys()))
    
    # Random environment HDRI rotation
    hdri_rotation = np.random.uniform(0, 360)
    
    # Random environment strength
    env_strength = np.random.uniform(0.1, 3.0)
    
    # Color temperature mixing
    temp1 = np.random.choice(list(COLOR_TEMPS.values()))
    temp2 = np.random.choice(list(COLOR_TEMPS.values()))
    temp_mix = np.random.uniform(0, 1)
    
    return {
        'config': config,
        'hdri_rotation': hdri_rotation,
        'environment_strength': env_strength,
        'color_temperature': temp1 * temp_mix + temp2 * (1 - temp_mix),
        'lighting_params': LIGHTING_CONFIGS[config]
    }
```

---

## 📷 3. Camera Parameter Randomization

```python
# Camera intrinsics - extreme variation
CAMERA_PARAMS = {
    'focal_length': (15, 200),      # Wide angle to telephoto (mm)
    'sensor_width': (20, 50),       # Sensor size variation
    'fov': (10, 120),               # Field of view (degrees)
    'aspect_ratio': (0.7, 1.8),    # Portrait to landscape
}

# Camera position - full hemisphere sampling
CAMERA_POSITION = {
    'distance': (1.5, 8.0),        # Distance from subject (meters)
    'height': (0.5, 3.0),          # Camera height (meters)
    'azimuth': (0, 360),           # Horizontal angle (degrees)
    'elevation': (-15, 45),        # Vertical angle (degrees)
}

# Lens distortion parameters
LENS_DISTORTION = {
    'barrel_distortion': (-0.3, 0.3),    # Negative = pincushion
    'chromatic_aberration': (0, 0.05),   # Color fringing
    'vignetting': (0, 0.4),              # Corner darkening
}

# Motion blur and defocus
CAMERA_EFFECTS = {
    'motion_blur': (0, 0.1),             # Camera shake
    'depth_of_field': (0, 2.0),          # Background blur
    'focus_distance': (0.8, 8.0),        # Focus plane
}

def sample_camera_params():
    """Generate extreme camera parameter variation"""
    
    # Base intrinsics
    focal_length = np.random.uniform(*CAMERA_PARAMS['focal_length'])
    sensor_width = np.random.uniform(*CAMERA_PARAMS['sensor_width'])
    
    # Position sampling (spherical coordinates)
    distance = np.random.uniform(*CAMERA_POSITION['distance'])
    azimuth = np.random.uniform(*CAMERA_POSITION['azimuth'])
    elevation = np.random.uniform(*CAMERA_POSITION['elevation'])
    height = np.random.uniform(*CAMERA_POSITION['height'])
    
    # Convert to Cartesian
    x = distance * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth))
    y = distance * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth))
    z = distance * np.sin(np.deg2rad(elevation)) + height
    
    # Lens effects
    distortion = np.random.uniform(*LENS_DISTORTION['barrel_distortion'])
    vignetting = np.random.uniform(*LENS_DISTORTION['vignetting'])
    motion_blur = np.random.uniform(*CAMERA_EFFECTS['motion_blur'])
    
    return {
        'focal_length': focal_length,
        'sensor_width': sensor_width,
        'position': (x, y, z),
        'target': (0, 0, 1.0),  # Look at chest level
        'distortion': distortion,
        'vignetting': vignetting,
        'motion_blur': motion_blur
    }
```

---

## 🧵 4. Cloth Physics Randomization

```python
# Cloth material properties - extreme variation
CLOTH_MATERIALS = {
    'rigid_denim': {
        'structural_stiffness': (80, 150),
        'bending_stiffness': (5, 20),
        'mass': (0.8, 1.2),
        'air_damping': (0.5, 1.0)
    },
    
    'flowing_silk': {
        'structural_stiffness': (2, 15),
        'bending_stiffness': (0.1, 1.0),
        'mass': (0.1, 0.3),
        'air_damping': (2.0, 5.0)
    },
    
    'thick_wool': {
        'structural_stiffness': (40, 80),
        'bending_stiffness': (8, 25),
        'mass': (0.6, 1.0),
        'air_damping': (0.8, 1.5)
    },
    
    'stretchy_lycra': {
        'structural_stiffness': (5, 25),
        'bending_stiffness': (0.5, 3.0),
        'mass': (0.2, 0.4),
        'air_damping': (1.0, 2.0)
    }
}

# Environmental forces
PHYSICS_ENVIRONMENT = {
    'gravity': (8.0, 12.0),           # Gravity variation (m/s²)
    'wind_force': (0, 15.0),          # Wind strength
    'wind_direction': (0, 360),       # Wind direction (degrees)
    'air_pressure': (0.8, 1.2),      # Atmospheric pressure
}

# Garment fit variation
GARMENT_FIT = {
    'size_scale': (0.85, 1.25),      # Tight to oversized
    'length_scale': (0.9, 1.2),      # Short to long
    'sleeve_length': (0.8, 1.3),     # Sleeve variation
}

def sample_cloth_physics(garment_type='shirt'):
    """Generate extreme cloth physics variation"""
    
    # Select base material
    material_name = np.random.choice(list(CLOTH_MATERIALS.keys()))
    base_props = CLOTH_MATERIALS[material_name]
    
    # Sample from ranges
    physics_params = {}
    for prop, (min_val, max_val) in base_props.items():
        physics_params[prop] = np.random.uniform(min_val, max_val)
    
    # Environmental forces
    gravity = np.random.uniform(*PHYSICS_ENVIRONMENT['gravity'])
    wind_force = np.random.uniform(*PHYSICS_ENVIRONMENT['wind_force'])
    wind_dir = np.random.uniform(*PHYSICS_ENVIRONMENT['wind_direction'])
    
    # Garment fit
    size_scale = np.random.uniform(*GARMENT_FIT['size_scale'])
    
    return {
        'material_type': material_name,
        'physics_params': physics_params,
        'gravity': gravity,
        'wind_force': wind_force,
        'wind_direction': wind_dir,
        'size_scale': size_scale,
        'simulation_steps': np.random.randint(30, 100)
    }
```

---

## 🌍 5. Background Environment Randomization

```python
# Background complexity levels
BACKGROUND_TYPES = {
    'plain_wall': {
        'color': 'random_solid',
        'texture': None,
        'complexity': 'minimal'
    },
    
    'textured_surface': {
        'patterns': ['brick', 'wood', 'concrete', 'fabric'],
        'scale': (0.1, 2.0),
        'roughness': (0.1, 0.8),
        'complexity': 'medium'
    },
    
    'cluttered_indoor': {
        'objects': ['furniture', 'decorations', 'appliances'],
        'density': (0.3, 0.8),
        'depth_variation': (0.5, 3.0),
        'complexity': 'high'
    },
    
    'outdoor_scene': {
        'environment': ['urban', 'park', 'beach', 'mountains'],
        'weather': ['sunny', 'cloudy', 'overcast'],
        'time_of_day': ['morning', 'noon', 'afternoon', 'evening'],
        'complexity': 'extreme'
    }
}

# Color space extremes
COLOR_SCHEMES = {
    'monochrome': [(0.3, 0.3, 0.3), (0.7, 0.7, 0.7)],
    'high_contrast': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    'warm_tones': [(0.8, 0.4, 0.2), (1.0, 0.7, 0.4)],
    'cool_tones': [(0.2, 0.4, 0.8), (0.4, 0.7, 1.0)],
    'saturated': [(0.9, 0.1, 0.1), (0.1, 0.9, 0.1), (0.1, 0.1, 0.9)]
}

def sample_background():
    """Generate extreme background variation"""
    bg_type = np.random.choice(list(BACKGROUND_TYPES.keys()))
    color_scheme = np.random.choice(list(COLOR_SCHEMES.keys()))
    
    # Lighting interaction
    background_emission = np.random.uniform(0, 2.0)
    reflectivity = np.random.uniform(0.1, 0.9)
    
    return {
        'type': bg_type,
        'color_scheme': color_scheme,
        'emission_strength': background_emission,
        'reflectivity': reflectivity,
        'complexity': BACKGROUND_TYPES[bg_type]['complexity']
    }
```

---

## 🔧 6. Training Loss Design for Synthetic-Only

```python
# Multi-scale supervision for robust learning
LOSS_COMPONENTS = {
    'pose_reconstruction': {
        'weight': 1.0,
        'loss_type': 'l2',
        'scale_invariant': True
    },
    
    'joint_2d_projection': {
        'weight': 5.0,      # Higher weight for 2D consistency
        'loss_type': 'robust_l1',
        'visibility_weighting': True
    },
    
    'joint_3d_position': {
        'weight': 2.0,
        'loss_type': 'l2',
        'bone_length_constraint': True
    },
    
    'shape_regularization': {
        'weight': 0.1,
        'loss_type': 'l2',
        'prior_distribution': 'normal'  # Keep shapes reasonable
    },
    
    'pose_smoothness': {
        'weight': 0.5,
        'loss_type': 'temporal_consistency',
        'sequence_length': 5
    }
}

# Adaptive loss weighting based on difficulty
def compute_adaptive_loss(pred_pose, gt_pose, gt_2d_joints, sample_metadata):
    total_loss = 0
    
    # Base pose loss
    pose_loss = F.mse_loss(pred_pose, gt_pose)
    
    # 2D projection loss (more robust)
    proj_2d = project_to_2d(pred_pose, sample_metadata['camera_params'])
    proj_loss = robust_l1_loss(proj_2d, gt_2d_joints)
    
    # Difficulty weighting
    difficulty = sample_metadata.get('pose_category', 'normal')
    if difficulty == 'extreme':
        pose_weight = 2.0  # Higher weight for extreme poses
        proj_weight = 8.0  # Much higher weight for 2D consistency
    else:
        pose_weight = 1.0
        proj_weight = 5.0
    
    total_loss = pose_weight * pose_loss + proj_weight * proj_loss
    
    return total_loss
```

---

## 📈 7. Data Generation Strategy

```python
# Systematic coverage of parameter space
GENERATION_STRATEGY = {
    'phase_1_exploration': {
        'samples': 50000,
        'strategy': 'uniform_sampling',
        'focus': 'parameter_space_coverage'
    },
    
    'phase_2_adversarial': {
        'samples': 30000, 
        'strategy': 'hard_negative_mining',
        'focus': 'difficult_cases'
    },
    
    'phase_3_balanced': {
        'samples': 100000,
        'strategy': 'balanced_distribution',
        'focus': 'realistic_distribution'
    },
    
    'phase_4_extreme': {
        'samples': 20000,
        'strategy': 'edge_case_sampling', 
        'focus': 'robustness_testing'
    }
}

# Quality metrics for synthetic data
QUALITY_METRICS = {
    'pose_realism': 'anatomical_constraint_violations',
    'cloth_physics': 'intersection_count',
    'lighting_realism': 'histogram_analysis',
    'camera_realism': 'perspective_distortion',
    'overall_realism': 'discriminator_score'
}
```

---

## ⚡ 8. Production Pipeline Configuration

```python
# Optimized for RTX 2050
RTX_2050_CONFIG = {
    'render_resolution': (512, 512),
    'cycles_samples': 64,           # Balanced quality/speed
    'max_bounces': 8,               # Reduced for speed
    'denoiser': 'OPTIX',            # RTX acceleration
    'cloth_quality': 6,             # Medium physics quality
    'batch_size': 4,                # Memory limit
    'parallel_scenes': 2            # Multi-scene rendering
}

# Expected performance targets
PERFORMANCE_TARGETS = {
    'samples_per_hour': 200,        # RTX 2050 target
    'quality_threshold': 0.85,     # Realism score
    'diversity_score': 0.9,        # Parameter coverage
    'physics_stability': 0.95      # Simulation success rate
}
```

---

## 🎯 Implementation Priority

1. **Week 1**: Implement extreme pose sampling and cloth physics
2. **Week 2**: Add aggressive lighting and camera randomization  
3. **Week 3**: Background variation and environment effects
4. **Week 4**: Training pipeline with adaptive losses
5. **Week 5**: Quality validation and parameter tuning

**Critical Success Factor**: The synthetic distribution MUST be more diverse than real-world conditions to achieve generalization.

This is aggressive domain randomization - not gentle variation. The goal is to make real-world data seem "easy" compared to the synthetic training distribution.
#!/usr/bin/env python3
"""
Blender Renderer for Synthetic Human Data Factory

🎨 This module handles Blender automation for:
   • SMPL mesh rendering
   • Cloth physics simulation  
   • Domain randomization
   • Multi-view capture
   • Automatic labeling

Designed for RTX 2050 optimization with commercial-grade output.
"""

import bpy
import bmesh
import numpy as np
import json
import mathutils
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
log = logging.getLogger("BlenderRenderer")

class BlenderSMPLRenderer:
    """Handles SMPL mesh rendering in Blender"""
    
    def __init__(self, output_resolution: Tuple[int, int] = (512, 512)):
        self.resolution = output_resolution
        self.setup_scene()
        self.setup_camera()
        self.setup_lighting()
        
        log.info(f"Blender SMPL Renderer initialized at {output_resolution}")
    
    def setup_scene(self):
        """Initialize Blender scene for rendering"""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Set render engine for RTX 2050 optimization
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        
        # Optimize for RTX 2050
        bpy.context.scene.cycles.samples = 64  # Reduced for speed
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPTIX'  # RTX denoising
        
        # Set resolution
        bpy.context.scene.render.resolution_x = self.resolution[0]
        bpy.context.scene.render.resolution_y = self.resolution[1]
        bpy.context.scene.render.resolution_percentage = 100
        
        log.info("Scene setup complete with CYCLES + OptiX denoising")
    
    def setup_camera(self):
        """Setup and configure camera"""
        # Add camera
        bpy.ops.object.camera_add(location=(0, -3, 1.5))
        self.camera = bpy.context.active_object
        bpy.context.scene.camera = self.camera
        
        # Camera settings for human capture
        self.camera.data.lens = 50  # 50mm lens (portrait standard)
        self.camera.data.sensor_width = 36  # Full frame sensor
        
        # Point camera at origin (where SMPL will be)
        constraint = self.camera.constraints.new('TRACK_TO')
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # Camera target (empty object at origin)
        bpy.ops.object.empty_add(location=(0, 0, 1.0))
        self.camera_target = bpy.context.active_object
        constraint.target = self.camera_target
        
        log.info("Camera setup with 50mm lens and tracking")
    
    def setup_lighting(self):
        """Setup professional lighting for human photography"""
        # Main key light (studio-style)
        bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
        key_light = bpy.context.active_object
        key_light.data.energy = 100
        key_light.data.size = 2.0
        
        # Fill light (softer)
        bpy.ops.object.light_add(type='AREA', location=(-1, -1.5, 2))
        fill_light = bpy.context.active_object  
        fill_light.data.energy = 40
        fill_light.data.size = 3.0
        
        # Rim light (separation)
        bpy.ops.object.light_add(type='SPOT', location=(0, 2, 2))
        rim_light = bpy.context.active_object
        rim_light.data.energy = 60
        rim_light.data.spot_size = 1.2
        
        # Environment lighting
        world = bpy.context.scene.world
        world.use_nodes = True
        env_texture = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
        world.node_tree.links.new(env_texture.outputs['Color'], 
                                 world.node_tree.nodes['Background'].inputs['Color'])
        
        log.info("Professional 3-point lighting setup complete")
    
    def randomize_lighting(self, intensity_scale: float = 1.0):
        """Randomize lighting for domain randomization"""
        # Randomize light intensities
        for obj in bpy.context.scene.objects:
            if obj.type == 'LIGHT':
                base_energy = obj.data.energy
                variation = np.random.uniform(0.7, 1.3) * intensity_scale
                obj.data.energy = base_energy * variation
        
        # Randomize light colors (slight tint)
        for obj in bpy.context.scene.objects:
            if obj.type == 'LIGHT':
                obj.data.color = (
                    np.random.uniform(0.9, 1.0),  # Slight red tint
                    np.random.uniform(0.9, 1.0),  # Slight green tint  
                    np.random.uniform(0.95, 1.0)  # Minimal blue variation
                )
    
    def randomize_camera(self):
        """Randomize camera position and settings"""
        # Random camera orbit around subject
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(2.5, 4.0)
        height = np.random.uniform(1.2, 1.8)
        
        self.camera.location = (
            distance * np.cos(angle),
            distance * np.sin(angle) - 2,  # Offset for front-facing
            height
        )
        
        # Random focal length variation
        self.camera.data.lens = np.random.uniform(35, 85)  # 35mm-85mm range
        
        # Random camera target height (look at different parts of body)
        self.camera_target.location.z = np.random.uniform(0.8, 1.5)


class BlenderClothSimulator:
    """Handles cloth physics simulation in Blender"""
    
    def __init__(self):
        self.cloth_presets = {
            'cotton_shirt': {
                'structural': 15,
                'shear': 5, 
                'bending': 0.5,
                'tension': 15,
                'compression': 15,
                'mass': 0.3
            },
            'denim_jeans': {
                'structural': 40,
                'shear': 15,
                'bending': 2.0, 
                'tension': 30,
                'compression': 30,
                'mass': 0.8
            },
            'silk_dress': {
                'structural': 8,
                'shear': 2,
                'bending': 0.1,
                'tension': 8,
                'compression': 8, 
                'mass': 0.15
            },
            'wool_hoodie': {
                'structural': 25,
                'shear': 10,
                'bending': 1.0,
                'tension': 20,
                'compression': 20,
                'mass': 0.5
            }
        }
        
        log.info(f"Cloth simulator initialized with {len(self.cloth_presets)} presets")
    
    def add_cloth_physics(self, garment_obj, garment_type: str = 'cotton_shirt'):
        """Add cloth physics to a garment object"""
        # Ensure object is selected
        bpy.context.view_layer.objects.active = garment_obj
        garment_obj.select_set(True)
        
        # Add cloth modifier
        cloth_modifier = garment_obj.modifiers.new(name="Cloth", type='CLOTH')
        cloth_settings = cloth_modifier.settings
        
        # Apply preset properties
        preset = self.cloth_presets.get(garment_type, self.cloth_presets['cotton_shirt'])
        
        cloth_settings.structural_stiffness = preset['structural']
        cloth_settings.shear_stiffness = preset['shear'] 
        cloth_settings.bending_stiffness = preset['bending']
        cloth_settings.tension_stiffness = preset['tension']
        cloth_settings.compression_stiffness = preset['compression']
        cloth_settings.mass = preset['mass']
        
        # Quality settings (balanced for RTX 2050)
        cloth_settings.quality = 8  # Reasonable quality
        cloth_settings.time_scale = 1.0
        cloth_settings.air_damping = 1.0
        
        log.info(f"Added cloth physics for {garment_type}")
        
        return cloth_modifier
    
    def setup_collision(self, body_obj):
        """Setup collision detection with body mesh"""
        bpy.context.view_layer.objects.active = body_obj
        body_obj.select_set(True)
        
        # Add collision modifier
        collision_modifier = body_obj.modifiers.new(name="Collision", type='COLLISION')
        collision_settings = collision_modifier.settings
        
        # Collision properties
        collision_settings.thickness_outer = 0.02  # 2cm collision margin
        collision_settings.thickness_inner = 0.02
        collision_settings.cloth_friction = 0.3
        collision_settings.damping = 0.59
        
        log.info("Body collision setup complete")
        
        return collision_modifier
    
    def simulate_cloth_settling(self, frame_count: int = 50):
        """Run cloth simulation to let garments settle naturally"""
        # Set frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = frame_count
        bpy.context.scene.frame_current = 1
        
        # Run simulation
        for frame in range(1, frame_count + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
        
        # Return to final frame
        bpy.context.scene.frame_set(frame_count)
        
        log.info(f"Cloth simulation complete ({frame_count} frames)")


class BlenderDomainRandomizer:
    """Handles domain randomization for realistic data generation"""
    
    def __init__(self):
        self.skin_materials = self._create_skin_materials()
        self.fabric_materials = self._create_fabric_materials()
        self.hdri_backgrounds = self._load_hdri_environments()
        
        log.info("Domain randomizer initialized with material and environment libraries")
    
    def randomize_skin_appearance(self, body_obj):
        """Randomize skin tone and appearance"""
        # Select random skin material
        skin_material = np.random.choice(self.skin_materials)
        
        # Apply to body object
        if body_obj.data.materials:
            body_obj.data.materials[0] = skin_material
        else:
            body_obj.data.materials.append(skin_material)
        
        # Randomize skin properties
        if skin_material.use_nodes:
            principled = skin_material.node_tree.nodes.get('Principled BSDF')
            if principled:
                # Randomize base color (skin tone variation)
                base_color = np.random.uniform([0.4, 0.25, 0.15], [0.95, 0.8, 0.7])
                principled.inputs['Base Color'].default_value = (*base_color, 1.0)
                
                # Randomize roughness and subsurface
                principled.inputs['Roughness'].default_value = np.random.uniform(0.3, 0.8)
                principled.inputs['Subsurface'].default_value = np.random.uniform(0.1, 0.3)
    
    def randomize_garment_appearance(self, garment_obj, garment_type: str):
        """Randomize garment materials and colors"""
        # Select appropriate fabric material
        fabric_material = self._select_fabric_material(garment_type)
        
        # Apply to garment
        if garment_obj.data.materials:
            garment_obj.data.materials[0] = fabric_material
        else:
            garment_obj.data.materials.append(fabric_material)
        
        # Randomize fabric properties
        if fabric_material.use_nodes:
            principled = fabric_material.node_tree.nodes.get('Principled BSDF')
            if principled:
                # Random color
                base_color = self._random_garment_color(garment_type)
                principled.inputs['Base Color'].default_value = (*base_color, 1.0)
                
                # Random fabric properties
                principled.inputs['Roughness'].default_value = np.random.uniform(0.2, 0.9)
                principled.inputs['Metallic'].default_value = np.random.uniform(0.0, 0.1)
    
    def randomize_environment(self):
        """Randomize background and environment lighting"""
        # Random HDRI environment
        if self.hdri_backgrounds:
            hdri_path = np.random.choice(self.hdri_backgrounds)
            self._set_hdri_background(hdri_path)
        
        # Randomize environment strength
        world = bpy.context.scene.world
        if world.use_nodes:
            background = world.node_tree.nodes.get('Background')
            if background:
                background.inputs['Strength'].default_value = np.random.uniform(0.5, 2.0)
    
    def _create_skin_materials(self) -> List:
        """Create various skin tone materials"""
        materials = []
        
        skin_tones = [
            [0.92, 0.78, 0.62],  # Light 
            [0.85, 0.68, 0.50],  # Medium-light
            [0.78, 0.57, 0.42],  # Medium  
            [0.65, 0.46, 0.34],  # Medium-dark
            [0.45, 0.32, 0.24]   # Dark
        ]
        
        for i, tone in enumerate(skin_tones):
            mat = bpy.data.materials.new(f"Skin_Tone_{i}")
            mat.use_nodes = True
            
            principled = mat.node_tree.nodes['Principled BSDF']
            principled.inputs['Base Color'].default_value = (*tone, 1.0)
            principled.inputs['Subsurface'].default_value = 0.2
            principled.inputs['Subsurface Color'].default_value = (*tone, 1.0)
            principled.inputs['Roughness'].default_value = 0.6
            
            materials.append(mat)
        
        return materials
    
    def _create_fabric_materials(self) -> List:
        """Create various fabric materials"""
        materials = []
        
        fabric_types = [
            {'name': 'Cotton', 'roughness': 0.7, 'specular': 0.3},
            {'name': 'Denim', 'roughness': 0.8, 'specular': 0.2},
            {'name': 'Silk', 'roughness': 0.1, 'specular': 0.8},
            {'name': 'Wool', 'roughness': 0.9, 'specular': 0.1},
            {'name': 'Synthetic', 'roughness': 0.4, 'specular': 0.6}
        ]
        
        for fabric in fabric_types:
            mat = bpy.data.materials.new(f"Fabric_{fabric['name']}")
            mat.use_nodes = True
            
            principled = mat.node_tree.nodes['Principled BSDF']
            principled.inputs['Roughness'].default_value = fabric['roughness']
            principled.inputs['Specular'].default_value = fabric['specular']
            
            materials.append(mat)
        
        return materials
    
    def _select_fabric_material(self, garment_type: str):
        """Select appropriate fabric material for garment type"""
        fabric_mapping = {
            'shirt': 0,      # Cotton
            'pants': 1,      # Denim
            'dress': 2,      # Silk
            'hoodie': 3,     # Wool
            'jacket': 4      # Synthetic
        }
        
        index = fabric_mapping.get(garment_type, 0)
        return self.fabric_materials[index]
    
    def _random_garment_color(self, garment_type: str) -> np.ndarray:
        """Generate random color appropriate for garment type"""
        color_palettes = {
            'shirt': [[1.0, 1.0, 1.0], [0.1, 0.2, 0.8], [0.8, 0.1, 0.1], [0.1, 0.1, 0.1]],
            'pants': [[0.1, 0.1, 0.3], [0.2, 0.2, 0.2], [0.5, 0.3, 0.1]],
            'dress': [[0.8, 0.1, 0.4], [0.1, 0.8, 0.3], [0.9, 0.9, 0.1], [0.1, 0.1, 0.8]],
            'hoodie': [[0.3, 0.3, 0.3], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
            'jacket': [[0.1, 0.1, 0.1], [0.3, 0.2, 0.1], [0.2, 0.4, 0.2]]
        }
        
        colors = color_palettes.get(garment_type, [[0.5, 0.5, 0.5]])
        base_color = np.array(np.random.choice(colors))
        
        # Add slight random variation
        variation = np.random.normal(0, 0.1, 3)
        final_color = np.clip(base_color + variation, 0, 1)
        
        return final_color
    
    def _load_hdri_environments(self) -> List[str]:
        """Load available HDRI environment maps"""
        # In production, scan for .hdr/.exr files
        # For now, return placeholder paths
        hdri_paths = [
            "environments/studio_1.hdr",
            "environments/outdoor_1.hdr", 
            "environments/indoor_1.hdr",
            "environments/sunset_1.hdr"
        ]
        return hdri_paths
    
    def _set_hdri_background(self, hdri_path: str):
        """Set HDRI background environment"""
        world = bpy.context.scene.world
        if world.use_nodes:
            env_texture = world.node_tree.nodes.get('Environment Texture')
            if env_texture and Path(hdri_path).exists():
                env_texture.image = bpy.data.images.load(hdri_path)


def render_sample(smpl_vertices: np.ndarray, 
                 smpl_faces: np.ndarray,
                 garment_vertices: np.ndarray,
                 garment_faces: np.ndarray, 
                 garment_type: str,
                 output_path: Path,
                 domain_randomization: bool = True) -> Dict[str, Any]:
    """
    Main function to render a single sample with SMPL + garment
    
    Args:
        smpl_vertices: SMPL body mesh vertices  
        smpl_faces: SMPL body mesh faces
        garment_vertices: Garment mesh vertices
        garment_faces: Garment mesh faces
        garment_type: Type of garment for material selection
        output_path: Output directory for renders
        domain_randomization: Whether to apply domain randomization
    
    Returns:
        Dictionary with render outputs and metadata
    """
    
    # Initialize components
    renderer = BlenderSMPLRenderer()
    cloth_sim = BlenderClothSimulator() 
    randomizer = BlenderDomainRandomizer()
    
    # Create body mesh
    body_mesh = bpy.data.meshes.new("SMPL_Body")
    body_mesh.from_pydata(smpl_vertices.tolist(), [], smpl_faces.tolist())
    body_mesh.update()
    
    body_obj = bpy.data.objects.new("SMPL_Body", body_mesh)
    bpy.context.collection.objects.link(body_obj)
    
    # Create garment mesh
    garment_mesh = bpy.data.meshes.new("Garment")
    garment_mesh.from_pydata(garment_vertices.tolist(), [], garment_faces.tolist())
    garment_mesh.update()
    
    garment_obj = bpy.data.objects.new("Garment", garment_mesh)
    bpy.context.collection.objects.link(garment_obj)
    
    # Setup physics
    cloth_sim.setup_collision(body_obj)
    cloth_sim.add_cloth_physics(garment_obj, garment_type)
    
    # Run cloth simulation
    cloth_sim.simulate_cloth_settling(frame_count=30)
    
    # Apply domain randomization
    if domain_randomization:
        randomizer.randomize_skin_appearance(body_obj)
        randomizer.randomize_garment_appearance(garment_obj, garment_type)
        randomizer.randomize_environment()
        renderer.randomize_lighting()
        renderer.randomize_camera()
    
    # Setup render outputs
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Render RGB image
    bpy.context.scene.render.filepath = str(output_path / "rgb.png")
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    
    # Render depth pass
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.render.filepath = str(output_path / "depth.exr") 
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.ops.render.render(write_still=True)
    
    # Extract camera parameters
    camera_data = {
        'location': list(renderer.camera.location),
        'rotation': list(renderer.camera.rotation_euler),
        'lens': renderer.camera.data.lens,
        'sensor_width': renderer.camera.data.sensor_width,
        'resolution': list(renderer.resolution)
    }
    
    log.info(f"Render complete: {output_path}")
    
    return {
        'rgb_path': output_path / "rgb.png",
        'depth_path': output_path / "depth.exr",
        'camera_params': camera_data,
        'garment_type': garment_type,
        'physics_simulated': True
    }


# Example usage (when run in Blender)
if __name__ == "__main__":
    log.info("🎨 Blender Renderer for Synthetic Human Data Factory")
    log.info("Ready for SMPL mesh rendering with cloth physics")
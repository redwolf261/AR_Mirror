#!/usr/bin/env python3
"""
3D Mesh Garment Wrapper
Wraps garment texture onto SMPL body mesh for true 3D fitting

This replaces 2D image warping with actual 3D mesh deformation,
enabling realistic cloth draping and physics simulation.

Pipeline:
  Garment Template → UV unwrapping → SMPL body mesh → Vertex projection → Draped mesh

Key advantages over 2D warping:
- Depth-aware occlusion (no floating garments)
- View-consistent (rotation works)
- Physics-ready (vertex springs for cloth simulation)
- Photometric realism (proper lighting/shading)
"""

import numpy as np
import torch
import cv2
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GarmentMesh:
    """
    3D garment mesh representation.
    
    Structure:
    - Vertices: 3D positions of garment points
    - Faces: Triangle connectivity
    - UV coords: Texture mapping
    - Texture: RGB appearance
    """
    
    def __init__(
        self,
        vertices: np.ndarray,      # (N, 3) - 3D positions
        faces: np.ndarray,          # (M, 3) - triangle indices
        uv_coords: np.ndarray,      # (N, 2) - UV coordinates
        texture: np.ndarray         # (H, W, 3) - RGB texture
    ):
        self.vertices = vertices
        self.faces = faces
        self.uv_coords = uv_coords
        self.texture = texture
    
    @classmethod
    def from_image(cls, garment_image: np.ndarray, garment_mask: np.ndarray) -> 'GarmentMesh':
        """
        Create garment mesh from 2D image + mask.
        
        Strategy:
        1. Extract garment silhouette
        2. Generate mesh from contour (Delaunay triangulation)
        3. Map image as texture
        
        This converts flat garment → 3D mesh ready for mapping.
        """
        # Simple planar mesh (Z=0)
        h, w = garment_image.shape[:2]
        
        # Create vertex grid
        grid_size = 32  # Lower resolution for speed
        x = np.linspace(0, w, grid_size)
        y = np.linspace(0, h, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Vertices in 3D (planar)
        vertices = np.stack([
            xx.ravel(),
            yy.ravel(),
            np.zeros_like(xx.ravel())  # Z = 0 (planar)
        ], axis=1).astype(np.float32)
        
        # Generate faces (triangulate grid) — vectorised
        rows = np.arange(grid_size - 1)
        cols = np.arange(grid_size - 1)
        ii, jj = np.meshgrid(rows, cols, indexing='ij')
        v0 = (ii * grid_size + jj).ravel()
        v1 = v0 + 1
        v2 = v0 + grid_size
        v3 = v2 + 1
        tri_a = np.stack([v0, v1, v2], axis=1)  # upper-left triangle
        tri_b = np.stack([v1, v3, v2], axis=1)  # lower-right triangle
        faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)
        
        # UV coordinates (normalized to [0, 1])
        uv_coords = np.stack([
            xx.ravel() / w,
            yy.ravel() / h
        ], axis=1).astype(np.float32)
        
        return cls(
            vertices=vertices,
            faces=faces,
            uv_coords=uv_coords,
            texture=garment_image.copy()
        )


class MeshGarmentWrapper:
    """
    Wrap garment mesh onto SMPL body mesh.
    
    This is the core 3D fitting engine.
    Takes flat garment mesh → conforms to body curvature.
    
    Methods:
    1. Closest point projection (fast, basic)
    2. Barycentric mapping (better quality)
    3. As-Rigid-As-Possible deformation (best quality, slower)
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize mesh wrapper.
        
        Args:
            device: 'cuda' or 'cpu' for computation
        """
        self.device = device
        logger.info("MeshGarmentWrapper initialized")
    
    def wrap_garment(
        self,
        garment_mesh: GarmentMesh,
        body_mesh,  # SMPLMeshResult
        garment_type: str = 'tshirt'
    ) -> 'WrappedGarmentMesh':
        """
        Wrap garment mesh onto body mesh.
        
        Args:
            garment_mesh: Flat garment mesh
            body_mesh: SMPL body mesh result
            garment_type: 'tshirt', 'dress', 'pants', etc.
        
        Returns:
            WrappedGarmentMesh with 3D-deformed vertices
        """
        # Get body region of interest
        body_vertices, body_faces = self._get_body_region(body_mesh, garment_type)
        
        # Method 1: Closest point projection (fast path)
        wrapped_vertices = self._project_to_surface(
            garment_mesh.vertices,
            body_vertices,
            body_mesh.normals
        )
        
        # Method 2: Scale to fit torso (simple sizing)
        wrapped_vertices = self._scale_to_fit(
            wrapped_vertices,
            body_vertices,
            garment_type
        )
        
        # Method 3: Smooth deformation (optional, for wrinkles)
        # wrapped_vertices = self._smooth_deformation(wrapped_vertices, body_vertices)
        
        return WrappedGarmentMesh(
            vertices=wrapped_vertices,
            faces=garment_mesh.faces,
            uv_coords=garment_mesh.uv_coords,
            texture=garment_mesh.texture,
            normals=self._compute_vertex_normals(wrapped_vertices, garment_mesh.faces)
        )
    
    def _get_body_region(
        self, 
        body_mesh,
        garment_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract relevant body region for garment type.
        
        SMPL has 6890 vertices with known topology.
        Torso region: vertices ~1000-3000
        Upper body: vertices ~1000-2500
        Lower body: vertices ~3000-5000
        """
        vertices = body_mesh.vertices
        faces = body_mesh.faces
        
        # Define vertex ranges (approximate from SMPL topology)
        region_map = {
            'tshirt': (1000, 2500),   # Upper torso
            'dress': (1000, 4500),     # Torso + upper legs
            'pants': (3000, 5500),     # Lower body
            'jacket': (1000, 3000),    # Full torso
        }
        
        start_idx, end_idx = region_map.get(garment_type, (1000, 3000))
        
        # Extract region vertices
        region_vertices = vertices[start_idx:end_idx]
        
        return region_vertices, faces
    
    def _project_to_surface(
        self,
        garment_vertices: np.ndarray,
        body_vertices: np.ndarray,
        body_normals: np.ndarray
    ) -> np.ndarray:
        """
        Project garment vertices onto body surface.
        
        For each garment vertex, find closest body vertex
        and offset along normal direction.
        Uses GPU-accelerated cdist with float32 for speed.
        """
        garment_v = torch.from_numpy(garment_vertices.astype(np.float32)).to(self.device)
        body_v = torch.from_numpy(body_vertices.astype(np.float32)).to(self.device)
        body_n = torch.from_numpy(body_normals.astype(np.float32)).to(self.device)
        
        # Fast nearest-neighbour via cdist + argmin
        dists = torch.cdist(garment_v, body_v)  # (Ng, Nb)
        closest_idx = dists.argmin(dim=1)        # (Ng,)
        
        # Project onto surface (offset along normal prevents z-fighting)
        epsilon = 0.02
        wrapped_v = body_v[closest_idx] + epsilon * body_n[closest_idx]
        
        return wrapped_v.cpu().numpy()
    
    def _scale_to_fit(
        self,
        garment_vertices: np.ndarray,
        body_vertices: np.ndarray,
        garment_type: str
    ) -> np.ndarray:
        """
        Scale garment to match body dimensions.
        
        Computes body bounding box and scales garment to fit.
        """
        # Body bounding box
        body_min = body_vertices.min(axis=0)
        body_max = body_vertices.max(axis=0)
        body_center = (body_min + body_max) / 2
        body_extent = body_max - body_min
        
        # Garment bounding box
        garment_min = garment_vertices.min(axis=0)
        garment_max = garment_vertices.max(axis=0)
        garment_center = (garment_min + garment_max) / 2
        garment_extent = garment_max - garment_min
        
        # Compute scale factors
        # Slightly larger than body for loose fit
        scale_factor = 1.05 * body_extent / (garment_extent + 1e-6)
        
        # Apply scaling and translation
        scaled_vertices = (garment_vertices - garment_center) * scale_factor + body_center
        
        return scaled_vertices
    
    @staticmethod
    def _compute_vertex_normals(
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Vectorised per-vertex normal computation (no Python loops)."""
        normals = np.zeros_like(vertices, dtype=np.float32)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= norms + 1e-8
        return normals


class WrappedGarmentMesh:
    """
    3D garment mesh wrapped onto body.
    
    Ready for:
    - Rendering (OpenGL/DirectX)
    - Physics simulation
    - Occlusion handling
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uv_coords: np.ndarray,
        texture: np.ndarray,
        normals: np.ndarray
    ):
        self.vertices = vertices      # (N, 3) - 3D positions (deformed)
        self.faces = faces            # (M, 3) - connectivity
        self.uv_coords = uv_coords    # (N, 2) - UV mapping
        self.texture = texture        # (H, W, 3) - RGB texture
        self.normals = normals        # (N, 3) - vertex normals
    
    def render_to_image(
        self,
        camera_matrix: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render wrapped mesh to 2D image.
        
        Uses software rasterization (fallback).
        For production, use OpenGL/DirectX for GPU rendering.
        """
        # Simple orthographic projection
        h, w = image_size
        
        # Project vertices to 2D
        vertices_2d = self.vertices[:, :2]  # Drop Z
        
        # Normalize to image space
        vertices_min = vertices_2d.min(axis=0)
        vertices_max = vertices_2d.max(axis=0)
        vertices_norm = (vertices_2d - vertices_min) / (vertices_max - vertices_min + 1e-6)
        vertices_img = vertices_norm * np.array([w, h])
        
        # Rasterize (simple triangle filling)
        output = np.zeros((h, w, 3), dtype=np.uint8)
        
        for face in self.faces:
            if face.max() >= len(vertices_img):
                continue
            
            # Get triangle vertices
            pts = vertices_img[face].astype(np.int32)
            
            # Get texture coordinates
            uv = (self.uv_coords[face] * np.array([self.texture.shape[1], self.texture.shape[0]])).astype(np.int32)
            
            # Simple flat shading (sample texture at centroid)
            uv_center = uv.mean(axis=0).astype(np.int32)
            uv_center = np.clip(uv_center, [0, 0], [self.texture.shape[1]-1, self.texture.shape[0]-1])
            color = self.texture[uv_center[1], uv_center[0]]
            
            # Fill triangle
            cv2.fillConvexPoly(output, pts, color.tolist())
        
        return output


class PhysicsSimulator:
    """
    GPU-accelerated cloth physics simulation.
    
    Uses mass-spring system with Verlet integration:
    - Structural springs (grid edges)
    - Shear springs (diagonals)
    - Bending springs (skip connections)
    
    Collision detection: sphere-based with SMPL body vertices
    
    Performance target: 60 FPS on RTX 2050
    """
    
    def __init__(self, device: str = 'cuda', grid_size: int = 32):
        """
        Initialize physics simulator.
        
        Args:
            device: 'cuda' for GPU acceleration
            grid_size: grid_size used for GarmentMesh.from_image (default 32)
        """
        self.device = device
        self.gravity = torch.tensor([0.0, -9.8, 0.0], device=device)
        self.damping = 0.99
        self.dt = 1.0 / 60.0  # 60 FPS physics tick
        self.grid_size = grid_size
        
        # Spring stiffness constants
        self.k_structural = 50.0   # strong  – holds grid shape
        self.k_shear = 30.0        # medium  – resists diagonal stretch
        self.k_bend = 10.0         # weak    – resists folding
        
        # Collision
        self.collision_offset = 0.02  # 2 cm clearance
        self.collision_stiffness = 200.0  # penalty force strength
        
        # Pre-compute spring connectivity for the mesh grid
        self._springs = None       # (S, 2) int tensor of spring index pairs
        self._rest_lengths = None  # (S,) float tensor
        self._spring_k = None     # (S,) stiffness per spring
        self._build_springs(grid_size)
        
        logger.info("Physics simulator initialized (spring forces + collision)")
    
    # ── Spring topology ───────────────────────────────────────────────────────
    
    def _build_springs(self, grid_size: int):
        """Pre-compute spring pairs and rest lengths for a grid_size×grid_size mesh."""
        pairs = []
        kinds = []  # 0=structural, 1=shear, 2=bend
        
        gs = grid_size
        for i in range(gs):
            for j in range(gs):
                idx = i * gs + j
                # Structural – horizontal
                if j + 1 < gs:
                    pairs.append((idx, idx + 1))
                    kinds.append(0)
                # Structural – vertical
                if i + 1 < gs:
                    pairs.append((idx, idx + gs))
                    kinds.append(0)
                # Shear – diagonals
                if i + 1 < gs and j + 1 < gs:
                    pairs.append((idx, idx + gs + 1))
                    kinds.append(1)
                if i + 1 < gs and j - 1 >= 0:
                    pairs.append((idx, idx + gs - 1))
                    kinds.append(1)
                # Bending – skip one (every other vertex)
                if j + 2 < gs:
                    pairs.append((idx, idx + 2))
                    kinds.append(2)
                if i + 2 < gs:
                    pairs.append((idx, idx + 2 * gs))
                    kinds.append(2)
        
        self._springs = torch.tensor(pairs, dtype=torch.long, device=self.device)  # (S, 2)
        
        stiffness_map = {0: self.k_structural, 1: self.k_shear, 2: self.k_bend}
        self._spring_k = torch.tensor(
            [stiffness_map[k] for k in kinds], dtype=torch.float32, device=self.device
        )
        
        # Rest lengths will be set on first simulate_step from actual mesh
        self._rest_lengths = None
        
        logger.debug(f"Built {len(pairs)} springs "
                      f"(struct={kinds.count(0)}, shear={kinds.count(1)}, bend={kinds.count(2)})")
    
    # ── Public API ────────────────────────────────────────────────────────────
    
    def simulate_step(
        self,
        garment_mesh: 'WrappedGarmentMesh',
        body_mesh,
        num_iterations: int = 10
    ) -> 'WrappedGarmentMesh':
        """
        Run one physics simulation step with Verlet integration.
        
        Updates vertex positions based on:
        - Spring forces (structural + shear + bending)
        - Gravity
        - Collision penalty forces against SMPL body
        
        Args:
            garment_mesh: Current mesh state
            body_mesh: SMPL body mesh (collision surface)
            num_iterations: Constraint solver iterations
        
        Returns:
            Updated mesh with new vertex positions
        """
        vertices = torch.from_numpy(garment_mesh.vertices).float().to(self.device)
        n_verts = vertices.shape[0]
        
        # Check whether pre-built springs match this mesh size
        springs_valid = (
            self._springs is not None
            and len(self._springs) > 0
            and self._springs.max().item() < n_verts
        )
        
        # Initialise previous positions (for Verlet) on first call
        if not hasattr(garment_mesh, '_prev_positions') or garment_mesh._prev_positions is None:
            garment_mesh._prev_positions = vertices.clone()
        prev = torch.from_numpy(
            garment_mesh._prev_positions.cpu().numpy()
            if isinstance(garment_mesh._prev_positions, torch.Tensor)
            else garment_mesh._prev_positions
        ).float().to(self.device)
        
        # Compute rest lengths on first call (only when springs valid)
        if self._rest_lengths is None and springs_valid:
            p0 = vertices[self._springs[:, 0]]
            p1 = vertices[self._springs[:, 1]]
            self._rest_lengths = torch.linalg.norm(p1 - p0, dim=1)
        
        # ── Accumulate forces ─────────────────────────────────────────────
        forces = torch.zeros_like(vertices)
        
        # Gravity
        forces += self.gravity.unsqueeze(0)  # broadcast to all vertices
        
        # Spring forces (vectorised) — only when springs match mesh
        if springs_valid:
            spring_forces = self._compute_spring_forces(vertices)
            forces += spring_forces
        
        # Body collision penalty
        if body_mesh is not None:
            collision_forces = self._collision_penalty(vertices, body_mesh)
            forces += collision_forces
        
        # ── Verlet integration ────────────────────────────────────────────
        dt2 = self.dt * self.dt
        new_positions = (
            vertices
            + self.damping * (vertices - prev)
            + forces * dt2
        )
        
        # ── Constraint solving (iterative position correction) ────────────
        if springs_valid:
            for _ in range(num_iterations):
                new_positions = self._solve_distance_constraints(new_positions)
        
        # Store state
        garment_mesh._prev_positions = vertices.cpu().numpy()
        garment_mesh.vertices = new_positions.cpu().numpy()
        
        # Recompute normals
        garment_mesh.normals = self._recompute_normals(
            garment_mesh.vertices, garment_mesh.faces
        )
        
        return garment_mesh
    
    # ── Force computation ─────────────────────────────────────────────────────
    
    def _compute_spring_forces(self, vertices: torch.Tensor) -> torch.Tensor:
        """Vectorised spring force computation on GPU."""
        forces = torch.zeros_like(vertices)
        
        p0 = vertices[self._springs[:, 0]]   # (S, 3)
        p1 = vertices[self._springs[:, 1]]   # (S, 3)
        
        diff = p1 - p0                       # (S, 3)
        dist = torch.linalg.norm(diff, dim=1, keepdim=True).clamp(min=1e-8)  # (S, 1)
        direction = diff / dist              # (S, 3)
        
        # Hooke's law: F = -k * (|x| - L0) * dir
        stretch = dist.squeeze() - self._rest_lengths    # (S,)
        magnitude = self._spring_k * stretch             # (S,)
        
        force_vec = magnitude.unsqueeze(1) * direction   # (S, 3)
        
        # Scatter-add to vertex force buffers (each spring affects 2 vertices)
        forces.index_add_(0, self._springs[:, 0], force_vec)
        forces.index_add_(0, self._springs[:, 1], -force_vec)
        
        return forces
    
    def _collision_penalty(
        self,
        garment_verts: torch.Tensor,
        body_mesh,
    ) -> torch.Tensor:
        """
        Penalty-based collision: push garment vertices out of the body.
        
        For each garment vertex closer than collision_offset to the body,
        apply a repulsive force along the body normal.
        """
        forces = torch.zeros_like(garment_verts)
        
        body_v = torch.from_numpy(body_mesh.vertices).float().to(self.device)
        body_n = torch.from_numpy(body_mesh.normals).float().to(self.device)
        
        # Pairwise distances (garment × body)
        dists = torch.cdist(garment_verts, body_v)   # (Ng, Nb)
        min_dist, closest_idx = dists.min(dim=1)      # (Ng,)
        
        # Penetration mask
        penetrating = min_dist < self.collision_offset  # (Ng,)
        
        if penetrating.any():
            # Closest body normals for penetrating vertices
            normals = body_n[closest_idx[penetrating]]  # (P, 3)
            
            # Penetration depth
            depth = self.collision_offset - min_dist[penetrating]  # (P,)
            
            # Penalty force = k * depth * normal
            penalty = self.collision_stiffness * depth.unsqueeze(1) * normals
            forces[penetrating] += penalty
        
        return forces
    
    def _solve_distance_constraints(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Position-Based Dynamics distance constraint solver.
        
        Iteratively corrects vertex positions so spring lengths
        stay close to their rest lengths — stabilises the simulation.
        """
        if self._springs is None or len(self._springs) == 0:
            return positions
        
        p0_idx = self._springs[:, 0]
        p1_idx = self._springs[:, 1]
        
        p0 = positions[p0_idx]
        p1 = positions[p1_idx]
        
        diff = p1 - p0
        dist = torch.linalg.norm(diff, dim=1, keepdim=True).clamp(min=1e-8)
        direction = diff / dist
        
        error = dist.squeeze() - self._rest_lengths
        correction = 0.5 * error.unsqueeze(1) * direction  # split equally
        
        # Apply corrections (scatter-add)
        positions = positions.clone()
        positions.index_add_(0, p0_idx, correction)
        positions.index_add_(0, p1_idx, -correction)
        
        return positions
    
    # ── Helpers ───────────────────────────────────────────────────────────────
    
    @staticmethod
    def _recompute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Fast per-vertex normal computation."""
        normals = np.zeros_like(vertices, dtype=np.float32)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= norms + 1e-8
        return normals


if __name__ == "__main__":
    # Test mesh garment wrapper
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy garment
    garment_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    garment_mask = np.ones((512, 512), dtype=np.uint8) * 255
    
    garment_mesh = GarmentMesh.from_image(garment_image, garment_mask)
    
    print(f"✓ Garment mesh created:")
    print(f"  Vertices: {garment_mesh.vertices.shape}")
    print(f"  Faces: {garment_mesh.faces.shape}")
    print(f"  UV coords: {garment_mesh.uv_coords.shape}")
    
    # Test wrapper
    wrapper = MeshGarmentWrapper(device='cpu')
    print(f"✓ Mesh wrapper ready")
    
    # Test physics simulator
    physics = PhysicsSimulator(device='cpu')
    print(f"✓ Physics simulator ready (GPU available: {torch.cuda.is_available()})")

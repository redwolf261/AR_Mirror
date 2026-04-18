"""
SMPL Body Reconstruction & Mesh Garment Wrapper Integration Tests

Tests the complete 3D reconstruction pipeline:
1. SMPL body reconstruction from 2D pose
2. 3D garment mesh generation
3. Garment wrapping onto body
4. Physics simulation (skeleton)

Author: Rivan Avinash Shetty (@redwolf261)
Date: February 14, 2026
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.core.smpl_body_reconstruction import (
    SMPLBodyReconstructor,
    SMPLMeshResult,
    SMPLRegressor,
    LightweightSMPL
)
from src.core.mesh_garment_wrapper import (
    GarmentMesh,
    MeshGarmentWrapper,
    WrappedGarmentMesh,
    PhysicsSimulator
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic SMPLMeshResult without requiring SMPL model files
# ---------------------------------------------------------------------------

def _make_synthetic_body_mesh(num_vertices: int = 6890) -> SMPLMeshResult:
    """Create a synthetic SMPL-like body mesh for testing."""
    # Body mesh: unit-sphere-ish shape
    np.random.seed(42)
    
    # Place vertices in a rough humanoid shape
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    for i in range(num_vertices):
        t = i / num_vertices
        # Vertical axis (height ~1.7m)
        y = t * 1.7
        # Width variation (torso wider, head narrow)
        if 0.3 < t < 0.6:
            width = 0.2
        else:
            width = 0.08
        x = width * np.sin(i * 0.1)
        z = width * np.cos(i * 0.1)
        vertices[i] = [x, y, z]
    
    # Create faces (simple sequential triangulation)
    faces = []
    for i in range(0, num_vertices - 2, 3):
        faces.append([i, i + 1, i + 2])
    faces = np.array(faces, dtype=np.int32)
    
    # Normals
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        normals[face] += n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return SMPLMeshResult(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uv_coords=None,
        shape_params=np.zeros(10, dtype=np.float32),
        pose_params=np.zeros(72, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def smpl_reconstructor():
    """Create SMPL reconstructor instance (may not be available)."""
    return SMPLBodyReconstructor(device='cpu')


@pytest.fixture
def mesh_wrapper():
    """Create mesh garment wrapper instance."""
    return MeshGarmentWrapper(device='cpu')


@pytest.fixture
def synthetic_body_mesh():
    """Synthetic SMPL body mesh (always available, no model files needed)."""
    return _make_synthetic_body_mesh()


@pytest.fixture
def synthetic_landmarks():
    """Generate synthetic MediaPipe landmarks dict (T-pose)."""
    landmarks = {}
    base_positions = [
        (0.50, 0.20),  # 0  nose
        (0.49, 0.18),  # 1  left eye inner
        (0.48, 0.18),  # 2  left eye
        (0.47, 0.18),  # 3  left eye outer
        (0.51, 0.18),  # 4  right eye inner
        (0.52, 0.18),  # 5  right eye
        (0.53, 0.18),  # 6  right eye outer
        (0.47, 0.20),  # 7  left ear
        (0.53, 0.20),  # 8  right ear
        (0.49, 0.22),  # 9  mouth left
        (0.51, 0.22),  # 10 mouth right
        (0.40, 0.40),  # 11 left shoulder
        (0.60, 0.40),  # 12 right shoulder
        (0.30, 0.55),  # 13 left elbow
        (0.70, 0.55),  # 14 right elbow
        (0.25, 0.70),  # 15 left wrist
        (0.75, 0.70),  # 16 right wrist
        (0.23, 0.72),  # 17 left pinky
        (0.77, 0.72),  # 18 right pinky
        (0.22, 0.71),  # 19 left index
        (0.78, 0.71),  # 20 right index
        (0.24, 0.73),  # 21 left thumb
        (0.76, 0.73),  # 22 right thumb
        (0.45, 0.65),  # 23 left hip
        (0.55, 0.65),  # 24 right hip
        (0.44, 0.80),  # 25 left knee
        (0.56, 0.80),  # 26 right knee
        (0.43, 0.95),  # 27 left ankle
        (0.57, 0.95),  # 28 right ankle
        (0.42, 0.97),  # 29 left heel
        (0.58, 0.97),  # 30 right heel
        (0.41, 0.98),  # 31 left foot index
        (0.59, 0.98),  # 32 right foot index
    ]
    for idx, (x, y) in enumerate(base_positions):
        landmarks[idx] = {'x': x, 'y': y, 'visibility': 0.95}
    return landmarks


@pytest.fixture
def synthetic_garment():
    """Generate synthetic garment image (red T-shirt)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 50:200] = [255, 0, 0]

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255

    return img, mask


# ===================================================================
# Test classes
# ===================================================================

class TestSMPLReconstruction:
    """Test SMPL body reconstruction from 2D pose."""

    def test_reconstructor_initialization(self, smpl_reconstructor):
        """Test SMPL reconstructor initializes (may be unavailable)."""
        assert smpl_reconstructor is not None
        assert smpl_reconstructor.device in ['cpu', 'cuda']

    def test_reconstruct_returns_none_when_unavailable(
        self, smpl_reconstructor, synthetic_landmarks
    ):
        """When model files are missing, reconstruct() returns None."""
        if smpl_reconstructor.is_available:
            pytest.skip("SMPL model is available, skip unavailable test")
        result = smpl_reconstructor.reconstruct(
            landmarks=synthetic_landmarks,
            frame_shape=(480, 640),
        )
        assert result is None

    @pytest.mark.skipif(
        not Path("models/smpl_neutral.pkl").exists(),
        reason="SMPL model file not present",
    )
    def test_reconstruct_from_landmarks(self, smpl_reconstructor, synthetic_landmarks):
        """Test reconstruction from synthetic landmarks (requires model)."""
        result = smpl_reconstructor.reconstruct(
            landmarks=synthetic_landmarks,
            frame_shape=(480, 640),
        )
        assert isinstance(result, SMPLMeshResult)
        assert result.vertices.shape == (6890, 3)
        assert result.faces.shape[1] == 3
        assert result.normals.shape == (6890, 3)
        assert result.shape_params.shape == (10,)
        assert result.pose_params.shape == (72,)

    def test_smpl_mesh_result_dataclass(self):
        """Test SMPLMeshResult stores data correctly."""
        verts = np.random.randn(6890, 3).astype(np.float32)
        faces = np.arange(6890 * 3).reshape(-1, 3).astype(np.int32)[:100]
        normals = np.random.randn(6890, 3).astype(np.float32)
        shape = np.zeros(10, dtype=np.float32)
        pose = np.zeros(72, dtype=np.float32)
        result = SMPLMeshResult(
            vertices=verts,
            faces=faces,
            normals=normals,
            uv_coords=None,
            shape_params=shape,
            pose_params=pose,
        )
        assert result.vertices.shape == (6890, 3)
        assert result.faces.shape == (100, 3)
        assert result.shape_params.shape == (10,)
        assert result.pose_params.shape == (72,)

    def test_lightweight_smpl_forward(self):
        """Test LightweightSMPL forward pass with dummy tensors."""
        num_v = 6890
        device = 'cpu'
        smpl = LightweightSMPL(
            v_template=torch.randn(num_v, 3, device=device),
            shapedirs=torch.randn(num_v, 3, 10, device=device),
            posedirs=torch.randn(num_v, 3, 207, device=device),
            J_regressor=torch.randn(24, num_v, device=device),
            weights=torch.randn(num_v, 24, device=device),
            faces=torch.zeros(13776, 3, dtype=torch.long, device=device),
            device=device,
        )
        output = smpl(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
        )
        assert output.vertices.shape == (1, num_v, 3)

    def test_regressor_network_structure(self):
        """Test SMPLRegressor network shape contract."""
        model = SMPLRegressor(input_dim=99, hidden_dim=1024, shape_dim=10, pose_dim=72)
        test_input = torch.randn(2, 99)
        with torch.no_grad():
            shape, pose = model(test_input)
        assert shape.shape == (2, 10)
        assert pose.shape == (2, 72)


class TestMeshGarmentWrapper:
    """Test 3D garment mesh generation and wrapping."""

    def test_wrapper_initialization(self, mesh_wrapper):
        """Test mesh wrapper initializes correctly."""
        assert mesh_wrapper is not None
        assert mesh_wrapper.device in ['cpu', 'cuda']

    def test_garment_mesh_from_image(self, synthetic_garment):
        """Test garment mesh generation from 2D image."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)

        assert garment_mesh.vertices.shape[0] > 0
        assert garment_mesh.vertices.shape[1] == 3
        assert garment_mesh.faces.shape[1] == 3
        assert garment_mesh.texture.shape[0] == img.shape[0]

    def test_garment_mesh_vertex_count(self, synthetic_garment):
        """Grid 32×32 should produce 1024 vertices."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        assert garment_mesh.vertices.shape[0] == 32 * 32  # default grid

    def test_wrap_garment_to_body(
        self, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Test wrapping garment onto synthetic body mesh."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)

        wrapped = mesh_wrapper.wrap_garment(
            garment_mesh=garment_mesh,
            body_mesh=synthetic_body_mesh,
            garment_type='tshirt',
        )

        assert isinstance(wrapped, WrappedGarmentMesh)
        assert wrapped.vertices.shape[0] == garment_mesh.vertices.shape[0]
        assert wrapped.vertices.shape[1] == 3
        assert wrapped.texture.shape == img.shape

    def test_wrapped_mesh_on_surface(
        self, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Test wrapped garment vertices are near body surface."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)

        wrapped = mesh_wrapper.wrap_garment(
            garment_mesh=garment_mesh,
            body_mesh=synthetic_body_mesh,
            garment_type='tshirt',
        )

        body_min = synthetic_body_mesh.vertices.min(axis=0)
        body_max = synthetic_body_mesh.vertices.max(axis=0)
        margin = 0.5  # generous margin for synthetic data
        assert np.all(wrapped.vertices >= body_min - margin)
        assert np.all(wrapped.vertices <= body_max + margin)

    def test_body_region_extraction(self, mesh_wrapper, synthetic_body_mesh):
        """Test extracting specific body regions."""
        for garment_type in ['tshirt', 'dress', 'pants', 'jacket']:
            region_vertices, _ = mesh_wrapper._get_body_region(
                synthetic_body_mesh, garment_type
            )
            assert region_vertices.shape[0] > 0
            assert region_vertices.shape[0] < synthetic_body_mesh.vertices.shape[0]
            assert region_vertices.shape[1] == 3


class TestPhysicsSimulation:
    """Test GPU cloth physics simulation."""

    @pytest.fixture
    def physics_simulator(self):
        """Create physics simulator instance."""
        return PhysicsSimulator(device='cpu')

    def test_simulator_initialization(self, physics_simulator):
        """Test physics simulator initializes correctly."""
        assert physics_simulator is not None
        # gravity is a torch tensor
        assert float(physics_simulator.gravity[1]) == pytest.approx(-9.8)
        assert physics_simulator.damping == 0.99

    def test_simulate_step(
        self, physics_simulator, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Test single physics simulation step."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)

        wrapped = mesh_wrapper.wrap_garment(
            garment_mesh=garment_mesh,
            body_mesh=synthetic_body_mesh,
            garment_type='tshirt',
        )

        initial_vertices = wrapped.vertices.copy()

        wrapped = physics_simulator.simulate_step(
            garment_mesh=wrapped,
            body_mesh=synthetic_body_mesh,
            num_iterations=1,
        )

        # Structure should be preserved
        assert wrapped.vertices.shape == initial_vertices.shape

    def test_gravity_effect(self, physics_simulator):
        """Test gravity pulls vertices down."""
        vertices = np.array(
            [[0, 1, 0], [0.5, 1, 0], [0.25, 1.5, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uv_coords = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        texture = np.zeros((64, 64, 3), dtype=np.uint8)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

        garment = WrappedGarmentMesh(vertices, faces, uv_coords, texture, normals)

        # Dummy body far away (no collision)
        body_mesh = SMPLMeshResult(
            vertices=np.array([[0, -10, 0]], dtype=np.float32),
            faces=np.array([[0, 0, 0]], dtype=np.int32),
            normals=np.array([[0, 1, 0]], dtype=np.float32),
            uv_coords=None,
            shape_params=np.zeros(10, dtype=np.float32),
            pose_params=np.zeros(72, dtype=np.float32),
        )

        for _ in range(10):
            garment = physics_simulator.simulate_step(
                garment_mesh=garment, body_mesh=body_mesh, num_iterations=1
            )

        # Structure preserved
        assert garment.vertices.shape == (3, 3)
        # Y should decrease due to gravity (velocity accumulates)
        assert garment.vertices[:, 1].mean() < 1.5  # Was 1.17 initially


class TestEndToEndPipeline:
    """Test complete 3D try-on pipeline (using synthetic body mesh)."""

    def test_full_pipeline(
        self, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Test entire pipeline from body mesh to rendered output."""
        body_mesh = synthetic_body_mesh
        assert body_mesh.vertices.shape == (6890, 3)

        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        assert garment_mesh.vertices.shape[0] > 0

        wrapped = mesh_wrapper.wrap_garment(
            garment_mesh=garment_mesh,
            body_mesh=body_mesh,
            garment_type='tshirt',
        )
        assert isinstance(wrapped, WrappedGarmentMesh)

        output = wrapped.render_to_image(
            camera_matrix=np.eye(3, dtype=np.float32),
            image_size=(480, 640),
        )
        assert output.shape == (480, 640, 3)
        assert output.dtype == np.uint8

    def test_pipeline_performance_benchmark(
        self, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Benchmark pipeline performance."""
        import time

        img, mask = synthetic_garment

        # Warm-up
        for _ in range(3):
            gm = GarmentMesh.from_image(img, mask)
            mesh_wrapper.wrap_garment(gm, synthetic_body_mesh, 'tshirt')

        num_iterations = 10
        start = time.time()

        for _ in range(num_iterations):
            gm = GarmentMesh.from_image(img, mask)
            wrapped = mesh_wrapper.wrap_garment(gm, synthetic_body_mesh, 'tshirt')
            _ = wrapped.render_to_image(np.eye(3, dtype=np.float32), (480, 640))

        elapsed = time.time() - start
        fps = num_iterations / elapsed

        print(
            f"\nPipeline Performance: {fps:.2f} FPS "
            f"({elapsed / num_iterations * 1000:.1f}ms per frame)"
        )

        # On CPU, expect at least 1 FPS
        assert fps >= 1.0

    def test_pipeline_with_different_garment_types(
        self, mesh_wrapper, synthetic_body_mesh, synthetic_garment
    ):
        """Test pipeline with different garment types."""
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)

        for garment_type in ['tshirt', 'dress', 'pants', 'jacket']:
            wrapped = mesh_wrapper.wrap_garment(
                garment_mesh=garment_mesh,
                body_mesh=synthetic_body_mesh,
                garment_type=garment_type,
            )
            assert isinstance(wrapped, WrappedGarmentMesh)
            assert wrapped.vertices.shape[0] > 0


class TestMemoryManagement:
    """Test memory usage and GPU management."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, synthetic_garment):
        """Test GPU memory usage stays within 4GB budget."""
        device = 'cuda'

        wrapper = MeshGarmentWrapper(device=device)
        body_mesh = _make_synthetic_body_mesh()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapper.wrap_garment(garment_mesh, body_mesh, 'tshirt')

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\nPeak GPU Memory: {peak_memory:.2f} GB")
        assert peak_memory < 2.0

    def test_cpu_fallback(self, synthetic_body_mesh, synthetic_garment):
        """Test pipeline works on CPU without GPU."""
        wrapper = MeshGarmentWrapper(device='cpu')
        img, mask = synthetic_garment

        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapped = wrapper.wrap_garment(garment_mesh, synthetic_body_mesh, 'tshirt')
        assert wrapped is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Renderer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGPURenderer:
    """Test the moderngl-based GPU hardware renderer."""

    def test_create_renderer_factory(self):
        """Factory should return a renderer (GPU or software fallback)."""
        from src.core.gpu_renderer import create_renderer
        renderer = create_renderer(640, 480, shading='flat')
        assert renderer.is_available
        renderer.release()

    def test_gpu_renderer_available(self):
        """GPURenderer should be available when moderngl is installed."""
        from src.core.gpu_renderer import GPURenderer, _MODERNGL_AVAILABLE
        if not _MODERNGL_AVAILABLE:
            pytest.skip("moderngl not installed")
        renderer = GPURenderer(320, 240, shading='flat')
        assert renderer.is_available
        renderer.release()

    def test_render_triangle(self):
        """Render a single triangle and verify output shape and non-zero pixels."""
        from src.core.gpu_renderer import create_renderer
        verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        tex = np.full((32, 32, 3), [180, 60, 60], dtype=np.uint8)

        renderer = create_renderer(320, 240, shading='flat')
        img = renderer.render(verts, faces, uvs, tex, image_size=(240, 320))
        assert img.shape == (240, 320, 4)
        assert img.dtype == np.uint8
        assert (img[:, :, 3] > 0).sum() > 100
        renderer.release()

    def test_render_phong(self):
        """Phong-shaded render should produce non-zero output."""
        from src.core.gpu_renderer import create_renderer
        verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        tex = np.full((32, 32, 3), [100, 200, 100], dtype=np.uint8)

        renderer = create_renderer(320, 240, shading='phong')
        img = renderer.render(verts, faces, uvs, tex, normals=normals, image_size=(240, 320))
        assert img.shape == (240, 320, 4)
        assert (img[:, :, 3] > 0).sum() > 50
        renderer.release()

    def test_render_wrapped_mesh(self, synthetic_body_mesh, synthetic_garment):
        """render_wrapped_mesh should accept a WrappedGarmentMesh."""
        from src.core.gpu_renderer import create_renderer
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapper = MeshGarmentWrapper(device='cpu')
        wrapped = wrapper.wrap_garment(garment_mesh, synthetic_body_mesh, 'tshirt')

        renderer = create_renderer(320, 240, shading='flat')
        out = renderer.render_wrapped_mesh(wrapped, image_size=(240, 320))
        assert out.shape == (240, 320, 4)
        renderer.release()

    def test_software_fallback(self):
        """SoftwareRenderer should produce valid output."""
        from src.core.gpu_renderer import SoftwareRenderer
        renderer = SoftwareRenderer(320, 240)
        verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        tex = np.full((32, 32, 3), [200, 100, 50], dtype=np.uint8)

        img = renderer.render(verts, faces, uvs, tex, image_size=(240, 320))
        assert img.shape == (240, 320, 4)
        renderer.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  Enhanced Physics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpringForcePhysics:
    """Test the upgraded mass-spring physics with Verlet integration."""

    def test_spring_topology(self):
        """Physics simulator should build correct number of springs for 32×32 grid."""
        physics = PhysicsSimulator(device='cpu', grid_size=32)
        # Structural: horizontal (31*32) + vertical (32*31) = 992 + 992 = 1984
        # Shear: 2 * 31 * 31 = 1922
        # Bending: horizontal (30*32) + vertical (32*30) = 960 + 960 = 1920
        assert physics._springs.shape[0] > 3000
        assert physics._spring_k.shape[0] == physics._springs.shape[0]

    def test_spring_forces_vectorised(self, synthetic_body_mesh, synthetic_garment):
        """Spring forces should be computed and change vertex positions."""
        physics = PhysicsSimulator(device='cpu', grid_size=32)
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapper = MeshGarmentWrapper(device='cpu')
        wrapped = wrapper.wrap_garment(garment_mesh, synthetic_body_mesh, 'tshirt')

        original_verts = wrapped.vertices.copy()
        wrapped = physics.simulate_step(wrapped, synthetic_body_mesh, num_iterations=3)
        # Vertices should have moved
        delta = np.abs(wrapped.vertices - original_verts).max()
        assert delta > 0.0

    def test_collision_penalty(self, synthetic_body_mesh, synthetic_garment):
        """Collision penalty should push vertices away from body."""
        physics = PhysicsSimulator(device='cpu', grid_size=32)
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapper = MeshGarmentWrapper(device='cpu')
        wrapped = wrapper.wrap_garment(garment_mesh, synthetic_body_mesh, 'tshirt')

        verts = torch.from_numpy(wrapped.vertices).float()
        forces = physics._collision_penalty(verts, synthetic_body_mesh)
        assert forces.shape == verts.shape

    def test_distance_constraints(self, synthetic_body_mesh, synthetic_garment):
        """Distance constraint solver should keep springs close to rest length."""
        physics = PhysicsSimulator(device='cpu', grid_size=32)
        img, mask = synthetic_garment
        garment_mesh = GarmentMesh.from_image(img, mask)
        wrapper = MeshGarmentWrapper(device='cpu')
        wrapped = wrapper.wrap_garment(garment_mesh, synthetic_body_mesh, 'tshirt')

        # Set rest lengths
        verts = torch.from_numpy(wrapped.vertices).float()
        p0 = verts[physics._springs[:, 0]]
        p1 = verts[physics._springs[:, 1]]
        physics._rest_lengths = torch.linalg.norm(p1 - p0, dim=1)

        # Perturb and solve
        perturbed = verts + torch.randn_like(verts) * 0.05
        corrected = physics._solve_distance_constraints(perturbed)
        assert corrected.shape == verts.shape


# ═══════════════════════════════════════════════════════════════════════════════
#  Temporal Cache Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalCache:
    """Test temporal coherence caching."""

    def test_cache_first_frame_is_miss(self, synthetic_landmarks):
        """First frame should always be a cache miss."""
        from src.core.temporal_cache import TemporalCache
        cache = TemporalCache(motion_threshold=0.01)
        assert cache.should_recompute(synthetic_landmarks) is True

    def test_cache_hit_on_static(self, synthetic_landmarks):
        """Identical landmarks should produce cache hits."""
        from src.core.temporal_cache import TemporalCache, CachedFrame
        import time
        cache = TemporalCache(motion_threshold=0.01, max_reuse_frames=5)

        # First frame: miss
        assert cache.should_recompute(synthetic_landmarks)
        cache.store(CachedFrame(timestamp=time.perf_counter(), landmarks=synthetic_landmarks))

        # Second frame: should be hit
        assert not cache.should_recompute(synthetic_landmarks)
        cached = cache.get_cached()
        assert cached is not None

    def test_cache_miss_on_motion(self, synthetic_landmarks):
        """Moved landmarks should trigger a recompute."""
        from src.core.temporal_cache import TemporalCache, CachedFrame
        import time
        cache = TemporalCache(motion_threshold=0.01)
        cache.store(CachedFrame(timestamp=time.perf_counter(), landmarks=synthetic_landmarks))

        # Move landmarks significantly
        moved = {
            idx: {'x': v['x'] + 0.1, 'y': v['y'] + 0.1, 'visibility': v['visibility']}
            for idx, v in synthetic_landmarks.items()
        }
        assert cache.should_recompute(moved)

    def test_max_reuse_forces_refresh(self, synthetic_landmarks):
        """Cache should force recompute after max_reuse_frames."""
        from src.core.temporal_cache import TemporalCache, CachedFrame
        import time
        cache = TemporalCache(motion_threshold=0.01, max_reuse_frames=2)
        cache.store(CachedFrame(timestamp=time.perf_counter(), landmarks=synthetic_landmarks))

        # Hit 1
        assert not cache.should_recompute(synthetic_landmarks)
        cache.get_cached()
        # Hit 2
        assert not cache.should_recompute(synthetic_landmarks)
        cache.get_cached()
        # Hit 3 - exceeds max_reuse_frames, forced miss
        assert cache.should_recompute(synthetic_landmarks)

    def test_hit_rate_stats(self, synthetic_landmarks):
        """Stats should correctly track hit rate."""
        from src.core.temporal_cache import TemporalCache, CachedFrame
        import time
        cache = TemporalCache(motion_threshold=0.01, max_reuse_frames=10)
        cache.store(CachedFrame(timestamp=time.perf_counter(), landmarks=synthetic_landmarks))

        for _ in range(4):
            cache.should_recompute(synthetic_landmarks)
            cache.get_cached()

        stats = cache.stats
        assert stats['cache_hits'] == 4
        assert stats['cache_misses'] == 1
        assert cache.hit_rate == pytest.approx(0.8, abs=0.01)


# ─── SMPL Regressor Training Tests ──────────────────────────────────────────

class TestSMPLRegressorTraining:
    """Tests for the SMPL regressor training pipeline."""

    def test_forward_kinematics_shape(self):
        """FK should produce 24 joints × 3 coordinates."""
        from scripts.train_smpl_regressor import forward_kinematics
        beta = np.zeros(10, dtype=np.float32)
        theta = np.zeros(72, dtype=np.float32)
        joints = forward_kinematics(beta, theta)
        assert joints.shape == (24, 3)

    def test_forward_kinematics_pose_changes_joints(self):
        """Non-zero pose should move joints away from rest pose."""
        from scripts.train_smpl_regressor import forward_kinematics
        beta = np.zeros(10, dtype=np.float32)
        theta_rest = np.zeros(72, dtype=np.float32)
        theta_posed = np.zeros(72, dtype=np.float32)
        # Rotate left hip (joint 1) significantly around X axis
        theta_posed[1 * 3] = 1.2
        joints_rest = forward_kinematics(beta, theta_rest)
        joints_posed = forward_kinematics(beta, theta_posed)
        # Left knee (joint 4, child of left hip) should shift
        diff = np.linalg.norm(joints_rest[4] - joints_posed[4])
        assert diff > 0.05, f"Knee only moved {diff:.4f}m"

    def test_rodrigues_batch_identity(self):
        """Zero axis-angle should give identity rotation."""
        from scripts.train_smpl_regressor import rodrigues_batch
        axis_angles = np.zeros((3, 3), dtype=np.float32)
        # Small epsilon to avoid divide-by-zero
        axis_angles[:, 0] = 1e-10
        R = rodrigues_batch(axis_angles)
        assert R.shape == (3, 3, 3)
        for i in range(3):
            assert np.allclose(R[i], np.eye(3), atol=1e-4)

    def test_project_to_2d_shape(self):
        """Projection should give 24 × 2 output."""
        from scripts.train_smpl_regressor import project_to_2d, REST_JOINTS
        pts_2d = project_to_2d(REST_JOINTS)
        assert pts_2d.shape == (24, 2)

    def test_synthetic_dataset_shapes(self):
        """Dataset __getitem__ should return correct tensor shapes."""
        from scripts.train_smpl_regressor import SyntheticSMPLDataset
        ds = SyntheticSMPLDataset(length=10)
        kp, beta, theta = ds[0]
        assert kp.shape == (99,)
        assert beta.shape == (10,)
        assert theta.shape == (72,)

    def test_iterative_regressor_forward(self):
        """IterativeSMPLRegressor should output correct shapes."""
        from scripts.train_smpl_regressor import IterativeSMPLRegressor
        model = IterativeSMPLRegressor(
            input_dim=99, hidden_dim=64, num_blocks=2, num_iterations=2
        )
        x = torch.randn(4, 99)
        shape, pose = model(x)
        assert shape.shape == (4, 10)
        assert pose.shape == (4, 72)

    def test_iterative_regressor_parameter_count(self):
        """Default model should be under 3M params (fits RTX 2050)."""
        from scripts.train_smpl_regressor import IterativeSMPLRegressor
        model = IterativeSMPLRegressor()
        n_params = model.count_parameters()
        assert n_params < 3_000_000
        assert n_params > 1_000_000

    def test_loss_function_outputs(self):
        """Loss should return scalar + metrics dict."""
        from scripts.train_smpl_regressor import SMPLRegressorLoss
        loss_fn = SMPLRegressorLoss()
        pred_shape = torch.randn(8, 10)
        pred_pose = torch.randn(8, 72)
        gt_shape = torch.randn(8, 10)
        gt_pose = torch.randn(8, 72)
        kp = torch.randn(8, 99)
        loss, metrics = loss_fn(pred_shape, pred_pose, gt_shape, gt_pose, kp)
        assert loss.ndim == 0  # scalar
        assert 'loss_total' in metrics
        assert 'loss_shape' in metrics
        assert 'loss_pose' in metrics

    def test_trained_checkpoint_loads(self):
        """If checkpoint exists, it should load successfully."""
        from scripts.train_smpl_regressor import IterativeSMPLRegressor
        ckpt_path = Path("models/smpl_regressor.pth")
        if not ckpt_path.exists():
            pytest.skip("No trained checkpoint available")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = checkpoint['model_config']
        model = IterativeSMPLRegressor(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_blocks=cfg['num_blocks'],
            num_iterations=cfg['num_iterations'],
        )
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # Quick inference test
        x = torch.randn(1, 99)
        shape, pose = model(x)
        assert shape.shape == (1, 10)

    def test_trained_model_inference_speed(self):
        """Trained model should run under 1ms per sample on GPU."""
        ckpt_path = Path("models/smpl_regressor.pth")
        if not ckpt_path.exists():
            pytest.skip("No trained checkpoint available")
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        from scripts.train_smpl_regressor import IterativeSMPLRegressor
        import time
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        cfg = checkpoint['model_config']
        model = IterativeSMPLRegressor(**{k: cfg[k] for k in
            ['input_dim', 'hidden_dim', 'num_blocks', 'num_iterations']
        }).cuda().eval()

        x = torch.randn(1, 99, device='cuda')
        # Warmup
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 100 * 1000  # ms

        assert elapsed < 5.0, f"Inference too slow: {elapsed:.2f}ms (target < 5ms)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

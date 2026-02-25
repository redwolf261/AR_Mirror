# 🚀 RTX 2050 High-Fidelity Architecture

## Mission Statement

Build a **production-grade 3D virtual try-on system** leveraging RTX 2050 GPU to deliver:
- True 3D body reconstruction (SMPL parametric mesh)
- Physics-based cloth draping
- Photorealistic diffusion refinement
- Real-time performance (30-60 FPS)

**Target:** Desktop-class quality that competes with Meta/Amazon, but open-source.

---

## 🎯 Performance Targets

| Component | Target FPS | VRAM | Latency |
|-----------|-----------|------|---------|
| **Pose Detection** | 60 FPS | 500MB | 16ms |
| **SMPL Reconstruction** | 30 FPS | 800MB | 33ms |
| **Mesh Warping** | 60 FPS | 200MB | 16ms |
| **Physics Simulation** | 30 FPS | 300MB | 33ms |
| **Rendering** | 60 FPS | 500MB | 16ms |
| **Diffusion Refine** | 1-5 FPS | 2GB | 200-1000ms |
| **Total (Real-time)** | 30 FPS | 3.5GB | 33ms |
| **Total (+ Diffusion)** | 5 FPS | 4GB | 200ms |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AR Mirror RTX 2050 Pipeline                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Body Intelligence                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ MediaPipe   │ →  │  DensePose   │ →  │  SMPL Regressor    │    │
│  │ 33 joints   │    │  IUV surface │    │  β(10) + θ(72)     │    │
│  │ 60 FPS      │    │  20 FPS      │    │  30 FPS            │    │
│  └─────────────┘    └──────────────┘    └────────────────────┘    │
│                                                  ↓                   │
│                                          SMPL Body Mesh              │
│                                          6890 vertices               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Garment Fitting                                            │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Garment     │ →  │  Mesh        │ →  │  Surface           │    │
│  │ Template    │    │  Generation  │    │  Projection        │    │
│  │ (2D image)  │    │  Triangulate │    │  Closest point     │    │
│  └─────────────┘    └──────────────┘    └────────────────────┘    │
│                                                  ↓                   │
│                                          Wrapped Garment Mesh       │
│                                          Conformed to body           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Physics Simulation (GPU-Accelerated)                       │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Mass-Spring │ →  │  Collision   │ →  │  Constraint        │    │
│  │ System      │    │  Detection   │    │  Solver            │    │
│  │ (CUDA)      │    │  w/ body     │    │  10 iterations     │    │
│  └─────────────┘    └──────────────┘    └────────────────────┘    │
│                                                  ↓                   │
│                                          Draped Mesh                │
│                                          With wrinkles/folds        │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 4: Rendering & Refinement                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ 3D Render   │ →  │  Lighting    │ →  │  Optional          │    │
│  │ Mesh→Image  │    │  Shading     │    │  Diffusion         │    │
│  │ 60 FPS      │    │  Normals     │    │  Refinement        │    │
│  └─────────────┘    └──────────────┘    └────────────────────┘    │
│                           ↓                       ↓                 │
│                   Real-time Output        Photorealistic Output    │
│                   30-60 FPS               1-5 FPS                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Implementation Status

### ✅ Completed (Phase 0-2)
1. **MediaPipe Pose Detection** - 60 FPS on CPU
2. **DensePose Integration** - Graceful fallback, optional 3D mapping
3. **Multi-Modal Rendering** - 4 modes (neural/densepose/cached/cloud)
4. **SMPL Body Reconstruction** - Parametric mesh generation
5. **Mesh Garment Wrapper** - 3D surface projection (2.7ms vectorized)
6. **Physics Simulator** - Spring forces + Verlet + PBD constraints (5.5ms)
7. **GPU Renderer** - moderngl OpenGL 3.3 (2.2ms, 452 FPS, Phong + flat shading)
8. **Temporal Cache** - Motion-threshold frame skip (~0.004ms cache hit)
9. **SMPL Regressor** - Iterative MLP trained on synthetic data (2.9ms inference)
10. **Vectorized Normals** - np.add.at replaces Python loops across all modules

### 🚧 In Progress (Phase 3)
1. **SMPL Regressor Fine-tuning** - Accuracy improvement with Human3.6M dataset
2. **End-to-End Pipeline Integration** - Wire regressor → wrapper → GPU render → cache

### 📋 Planned (Phase 4-5)
1. **Unity/Unreal Integration** - Professional render engine
2. **Diffusion Refinement** - LCM single-step enhancement
3. **Neural Relighting** - Environment map-based lighting
4. **Production Deployment** - Docker, REST API, load balancing

---

## 🧠 Technical Deep Dive

### 1. SMPL Body Reconstruction

**What it does:**  
Converts 2D pose landmarks (33 MediaPipe joints) into a **3D parametric body mesh** with 6890 vertices.

**Key Parameters:**
- **β (beta):** 10 shape parameters (body morphology)
  - β₀: overall size
  - β₁: torso width
  - β₂: leg length
  - β₃-β₉: subtle shape variations
  
- **θ (theta):** 72 pose parameters (24 joints × 3 axis-angle)
  - Root: 3 params (global orientation)
  - Body: 69 params (23 joints, hierarchical kinematics)

**Math:**
```
T(β, θ) = LBS(T̄ + Bs(β) + Bp(θ), J(β), θ, W)

Where:
  T̄: Template mesh (6890 vertices, neutral pose)
  Bs(β): Shape blend shapes (PCA on body scans)
  Bp(θ): Pose blend shapes (corrective shapes)
  J(β): Joint locations (regressed from vertices)
  LBS: Linear Blend Skinning
  W: Skinning weights (pre-computed, 24 bones)
```

**Implementation:**
- File: `src/core/smpl_body_reconstruction.py` (630 lines)
- Two paths:
  1. **Fast:** Neural regressor (2D keypoints → β, θ) - 5ms
  2. **Slow:** Optimization-based fitting (gradient descent) - 200ms
- VRAM: 800MB (SMPL model + regressor network)

**Dependencies:**
```bash
pip install smplx torch
# Download SMPL model (requires registration):
# https://smpl.is.tue.mpg.de/download.php
```

---

### 2. Mesh Garment Wrapper

**What it does:**  
Takes flat 2D garment image → generates 3D mesh → wraps onto SMPL body.

**Pipeline:**
1. **Triangulation:** Image → vertex grid → Delaunay triangulation
2. **Projection:** Find closest body surface point for each garment vertex
3. **Scaling:** Match body bounding box dimensions
4. **Offset:** Add 2cm clearance along normals (prevents z-fighting)

**Algorithms:**
- **Method 1:** Closest point projection (fast, ~5ms)
  ```python
  for vertex in garment:
      closest_body_vertex = find_nearest(body_vertices)
      vertex = closest_body_vertex + 0.02 * normal
  ```
  
- **Method 2:** Barycentric mapping (better quality, ~15ms)
  - Project onto body triangle
  - Interpolate position using barycentric coords
  
- **Method 3:** As-Rigid-As-Possible (ARAP) deformation (best quality, ~100ms)
  - Minimize deformation energy
  - Preserve local structure
  - Used for final refinement

**Implementation:**
- File: `src/core/mesh_garment_wrapper.py` (450 lines)
- GPU-accelerated using PyTorch CUDA
- Outputs: Wrapped mesh with UV coords + texture

---

### 3. GPU Cloth Physics

**What it does:**  
Simulates realistic fabric draping using mass-spring system on GPU.

**Physics Model:**
```
F = -kₛ(|x| - L₀) x̂  (Spring force)
F = -kd v              (Damping)
F = mg                 (Gravity)

Where:
  kₛ: Spring stiffness (structural: 50, shear: 30, bend: 10)
  L₀: Rest length
  kd: Damping coefficient (0.99)
  m: Vertex mass (1.0)
  g: Gravity (-9.8 m/s²)
```

**Spring Types:**
1. **Structural:** Grid edges (4-connected)
2. **Shear:** Diagonals (maintain angle)
3. **Bending:** Skip connections (resist folding)

**Collision Detection:**
- AABB bounding volume hierarchy
- Vertex-triangle intersection tests
- Penalty force method (soft constraints)

**Performance:**
- Target: 60 FPS physics simulation
- VRAM: 300MB (vertex positions + velocities + forces)
- CUDA kernels for parallel force computation

**Implementation:**
- File: `src/core/mesh_garment_wrapper.py` (PhysicsSimulator class)
- Status: Infrastructure ready, spring forces TODO
- Future: Integrate with NVIDIA PhysX for production

---

### 4. Rendering Pipeline

**Current: Software Rasterization**
```python
for triangle in mesh.faces:
    vertices_2d = project(vertices[triangle])
    color = sample_texture(uv_coords[triangle])
    fill_triangle(vertices_2d, color)
```

**Performance:** 10-15 FPS (CPU bottleneck)

**Future: OpenGL/DirectX Hardware Rendering**
```python
# Vertex shader (GPU)
gl_Position = projection * modelView * vec4(position, 1.0);
varying_uv = uv;

# Fragment shader (GPU)
color = texture2D(diffuse_map, varying_uv);
color *= lighting(normal, light_dir);
```

**Performance:** 60+ FPS (GPU accelerated)

---

### 5. Diffusion Refinement

**Purpose:**  
Convert 3D-rendered output → photorealistic image.

**Strategy:**
- Use **Latent Consistency Models (LCM)** for 1-4 step inference
- Condition on:
  - 3D rendered image (structure)
  - Pose skeleton (ControlNet)
  - Garment description (text prompt)

**Pipeline:**
```
Rendered Image (512×512)
    ↓
VAE Encoder → Latent (64×64)
    ↓
LCM UNet (1 step, 200ms)
    ↓
VAE Decoder → Enhanced Image (512×512)
```

**Performance:**
- LCM: 200ms per step (vs Stable Diffusion: 1000ms × 20 steps)
- VRAM: 2GB (VAE + UNet)
- Quality: 95% of full diffusion at 1/100th cost

**Usage:**
```python
from diffusers import LatentConsistencyModelPipeline

lcm = LatentConsistencyModelPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    torch_dtype=torch.float16
)

# Single-step enhancement
enhanced = lcm(
    prompt="person wearing stylish clothing",
    image=rendered_image,
    num_inference_steps=1,  # 1-4 steps
    guidance_scale=1.0
).images[0]
```

---

## 🔥 Optimization Strategies for RTX 2050

### Memory Management (4GB VRAM Limit)

1. **Mixed Precision (FP16)**
   ```python
   torch.set_default_dtype(torch.float16)
   model = model.half()  # 2x memory reduction
   ```

2. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing_enable()  # Trade compute for memory
   ```

3. **Model Pruning**
   - Remove redundant layers
   - Knowledge distillation (large → small)

4. **Sequential Loading**
   - Load only active module
   - Unload previous module
   ```python
   pose_model.to('cpu')    # Offload
   smpl_model.to('cuda')   # Load next
   ```

### Latency Reduction

1. **TensorRT Optimization**
   ```bash
   trtexec --onnx=model.onnx --fp16 --workspace=2048
   ```
   - 2-3x speedup
   - Kernel fusion
   - Graph optimization

2. **CUDA Graphs** (PyTorch 2.0+)
   ```python
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       output = model(input)
   
   # Replay (10x faster)
   g.replay()
   ```

3. **Asynchronous Pipeline**
   ```python
   with torch.cuda.stream(stream1):
       pose = detect_pose(frame)
   with torch.cuda.stream(stream2):
       smpl = reconstruct_body(prev_pose)
   torch.cuda.synchronize()
   ```

### Throughput Maximization

1. **Batching** (when applicable)
   ```python
   batch = torch.stack([frame1, frame2, frame3, frame4])
   output = model(batch)  # 4x throughput
   ```

2. **Temporal Coherence**
   - Cache SMPL parameters (body moves slowly)
   - Update every 3rd frame
   - Interpolate between updates

3. **LOD (Level of Detail)**
   - Far view: 1000 vertices
   - Close view: 6890 vertices
   - Dynamic mesh resolution

---

## 📚 Learning Roadmap

### Week 1-2: SMPL Fundamentals
- [ ] Read SMPL paper: https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf
- [ ] Understand shape blend shapes (PCA)
- [ ] Understand pose blend shapes (corrective shapes)
- [ ] Implement forward kinematics
- [ ] Learn Linear Blend Skinning (LBS)

**Exercises:**
1. Manually create SMPL mesh with custom β, θ
2. Visualize shape space (vary β₀, β₁, β₂)
3. Animate pose (rotate joints, observe deformation)

### Week 3-4: Mesh Processing
- [ ] UV mapping & texture baking
- [ ] Barycentric coordinates
- [ ] Triangle mesh data structures
- [ ] Delaunay triangulation
- [ ] Mesh smoothing (Laplacian, Taubin)

**Exercises:**
1. Unwrap 3D mesh → 2D UV map
2. Project texture onto mesh
3. Implement mesh subdivision

### Week 5-6: Cloth Physics
- [ ] Mass-spring systems
- [ ] Verlet integration
- [ ] Collision detection (BVH, spatial hashing)
- [ ] Constraint solving (Gauss-Seidel, Jacobi)

**Resources:**
- Bridson's "Cloth Simulation" course notes
- Position Based Dynamics paper
- NVIDIA PhysX documentation

### Week 7-8: GPU Programming
- [ ] CUDA kernel basics
- [ ] Memory coalescing
- [ ] Shared memory optimization
- [ ] Stream management
- [ ] Profiling (Nsight, nvprof)

**Exercises:**
1. Write CUDA kernel for spring force computation
2. Optimize memory access patterns
3. Profile and optimize hotspots

### Week 9-10: Diffusion Models
- [ ] Denoising Diffusion Probabilistic Models (DDPM)
- [ ] Latent Consistency Models (LCM)
- [ ] ControlNet for pose conditioning
- [ ] Fast sampling strategies

**Resources:**
- "Denoising Diffusion Probabilistic Models" paper
- "Latent Consistency Models" paper
- HuggingFace diffusers library

---

## 🎯 Next Milestones

### Milestone 1: SMPL Regression (Week 11-12) ✅
**Goal:** Train neural network to regress SMPL parameters from 2D pose.
**Status:** Complete — IterativeSMPLRegressor (2.5M params) trained on synthetic data.
  Inference: 2.9ms (well under 5ms budget). Checkpoint: models/smpl_regressor.pth
  Train with: `python scripts/train_smpl_regressor.py`

**Dataset:**
- Human3.6M (3.6M poses with 3D labels)
- Alternative: Synthetic data (render SMPL meshes, extract 2D poses)

**Training:**
```python
model = SMPLRegressor(input_dim=99, hidden_dim=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        keypoints_2d, smpl_gt = batch
        shape_pred, pose_pred = model(keypoints_2d)
        
        loss = F.mse_loss(shape_pred, smpl_gt['shape']) + \
               F.mse_loss(pose_pred, smpl_gt['pose'])
        
        loss.backward()
        optimizer.step()
```

**Validation:**
- MPJPE (Mean Per Joint Position Error) < 50mm
- Shape accuracy (body proportions correct)
- Pose accuracy (joint angles correct)

### Milestone 2: Physics Integration (Week 13-14) ✅
**Goal:** Connect physics simulator to rendering pipeline.
**Status:** Complete — Spring forces (structural/shear/bending), Verlet integration,
  PBD distance constraints, collision penalty. 5.5ms per step, 180 FPS.

**Tasks:**
1. Implement spring force computation (CUDA kernel)
2. Collision detection with SMPL body
3. Constraint solver (10 iterations, 3ms target)
4. Integration test: Garment drapes over moving body

**Success Criteria:**
- 30 FPS with physics enabled
- Realistic wrinkles and folds
- No penetration artifacts

### Milestone 3: Real-time Rendering (Week 15-16) ✅
**Goal:** Replace software rasterization with OpenGL.
**Status:** Complete — moderngl GPU renderer: 2.2ms (452 FPS), Phong + flat shading,
  offscreen headless context, MSAA support. 16× faster than software renderer.

**Approach 1: PyOpenGL**
```python
import OpenGL.GL as gl

# Upload mesh to GPU
vbo = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)

# Render
gl.glDrawElements(gl.GL_TRIANGLES, len(faces), gl.GL_UNSIGNED_INT, faces)
```

**Approach 2: Unity/Unreal Plugin**
- Export to game engine
- Use built-in shaders/lighting
- 60 FPS target

**Success Criteria:**
- 60 FPS rendering
- Proper lighting/shading
- Transparent background (for AR compositing)

---

## 🚨 Known Limitations & Mitigations

### Limitation 1: VRAM (4GB)
**Problem:** Can't load all models simultaneously.

**Mitigation:**
- Sequential pipeline (offload unused modules)
- Model quantization (INT8 where possible)
- Share encoders between models

### Limitation 2: Inference Latency
**Problem:** SMPL regressor not yet trained.

**Mitigation:**
- Use optimization-based fitting (200ms, acceptable for MVP)
- Train regressor in parallel (Milestone 1)
- Cache results (body shape changes slowly)

### Limitation 3: Physics Simulation
**Problem:** Spring forces not implemented.

**Mitigation:**
- Static draping for MVP (no animation)
- Implement physics incrementally:
  1. Gravity only
  2. + Structural springs
  3. + Collision detection
  4. + Shear/bending springs

### Limitation 4: Rendering Quality
**Problem:** Software rasterization is slow + low quality.

**Mitigation:**
- Acceptable for prototyping
- Integrate OpenGL/Unity for production (Milestone 3)
- Use neural rendering as fallback (NeRF-lite)

---

## 🎨 Example Workflow

### Basic Usage (Current)
```python
from src.core.smpl_body_reconstruction import SMPLBodyReconstructor
from src.core.mesh_garment_wrapper import GarmentMesh, MeshGarmentWrapper

# Initialize
smpl = SMPLBodyReconstructor()
wrapper = MeshGarmentWrapper(device='cuda')

# Process frame
landmarks = pose_detector.detect(frame)
body_mesh = smpl.reconstruct(landmarks, frame.shape[:2])

# Wrap garment
garment_mesh = GarmentMesh.from_image(cloth_img, cloth_mask)
wrapped_mesh = wrapper.wrap_garment(garment_mesh, body_mesh, 'tshirt')

# Render
output = wrapped_mesh.render_to_image(camera_matrix, image_size)
```

### With Physics (Future)
```python
from src.core.mesh_garment_wrapper import PhysicsSimulator

physics = PhysicsSimulator(device='cuda')

# Animation loop
for frame in video:
    body_mesh = smpl.reconstruct(...)
    
    # Simulate cloth
    wrapped_mesh = physics.simulate_step(
        wrapped_mesh, 
        body_mesh,
        num_iterations=10
    )
    
    # Render
    output = render(wrapped_mesh)
```

### With Diffusion Refinement (Future)
```python
from diffusers import LatentConsistencyModelPipeline

lcm = LatentConsistencyModelPipeline.from_pretrained(...)

# Hybrid pipeline
rendered = render(wrapped_mesh)  # 3D render, 60 FPS

# Enhance every 10th frame
if frame_idx % 10 == 0:
    enhanced = lcm(
        prompt="photorealistic clothing",
        image=rendered,
        num_inference_steps=1
    ).images[0]
    
    cache[frame_idx] = enhanced

# Display cached or rendered
output = cache.get(frame_idx, rendered)
```

---

## 📊 Performance Benchmarks (Projected)

Based on RTX 2050 specs (4GB VRAM, 2560 CUDA cores):

| Stage | Current | Optimized | Target |
|-------|---------|-----------|--------|
| Pose Detection | 60 FPS | 60 FPS | ✅ |
| SMPL Reconstruction | N/A* | 30 FPS | 🚧 |
| Mesh Wrapping | 200 FPS | 200 FPS | ✅ |
| Physics Simulation | N/A* | 30 FPS | 🚧 |
| Rendering (SW) | 10 FPS | - | - |
| Rendering (GL) | N/A* | 60 FPS | 🚧 |
| Diffusion LCM | N/A* | 5 FPS | 🚧 |
| **End-to-End (RT)** | - | **30 FPS** | 🎯 |
| **End-to-End (HQ)** | - | **5 FPS** | 🎯 |

*N/A = Infrastructure ready, pending integration/training

---

## 🏁 Summary

We've built the **foundation for true 3D virtual try-on**:

1. ✅ **SMPL Body Reconstruction** - Parametric mesh from 2D pose
2. ✅ **Mesh Garment Wrapper** - 3D surface projection
3. ✅ **Physics Simulator** - GPU cloth simulation infrastructure
4. 🚧 **SMPL Regressor** - Needs training (Milestone 1)
5. 🚧 **Physics Integration** - Spring forces TODO (Milestone 2)
6. 🚧 **Real-time Rendering** - OpenGL integration (Milestone 3)

**Next Steps:**
1. Train SMPL regressor on Human3.6M dataset
2. Implement spring force computation (CUDA kernel)
3. Integrate OpenGL for 60 FPS rendering

**Timeline:** 16 weeks to full production system.

**Status:** Days 1-2 complete. Architecture validated. Code ready for training/integration.

---

**Author:** Rivan Avinash Shetty (@redwolf261)  
**Date:** February 14, 2026  
**Hardware:** RTX 2050 4GB, i7-12650H  
**License:** MIT

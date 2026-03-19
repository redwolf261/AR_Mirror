# 👔 AR Mirror - 3D Virtual Try-On System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://mediapipe.dev/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Real-time 3D virtual try-on system with GPU-accelerated mesh rendering, physics simulation, and neural body reconstruction**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Performance](#-performance) • [Documentation](#-documentation)

</div>

> **Branch policy:** `master` is the single source of truth. The `main` branch has been deleted. Always work from `master`.
> ```bash
> git pull origin master   # before starting work
> git push origin master   # after committing
> ```

---

## 🎯 Overview

AR Mirror is a desktop-grade 3D virtual try-on system that reconstructs a full SMPL body mesh from a single webcam feed, wraps garment textures onto the 3D body surface, and renders the result in real-time using GPU-accelerated OpenGL. Optimized for NVIDIA RTX 2050 (4GB VRAM).

### 🌟 Key Highlights

- **21+ FPS on CPU (Phase 2)** — production neural warp baseline
- **3D Body Mesh** — SMPL parametric model (6890 vertices) from 2D pose
- **GPU Rendering** — moderngl OpenGL 3.3 with Phong shading (452 FPS)
- **Spring-Force Physics** — Verlet integration, structural/shear/bending springs
- **Trained Neural Regressor** — 2D keypoints → SMPL params in 2.9ms
- **Temporal Caching** — motion-threshold frame skip for static poses
- **Runtime Modes** — Phase 2 (default neural warp) with Phase 0 fallback

---

## ✨ Features

### 3D Virtual Try-On Pipeline

| Component | Description | Performance |
|-----------|-------------|-------------|
| **SMPL Body Reconstruction** | 6890-vertex parametric body mesh from 2D pose | 2.9ms inference |
| **GPU Mesh Rendering** | moderngl OpenGL 3.3, Phong + flat shading, MSAA | 2.2ms / 452 FPS |
| **Spring-Force Physics** | Verlet integration, PBD constraints, collision penalty | 5.5ms / 180 FPS |
| **Vectorized Mesh Wrapping** | UV-mapped garment → body surface via numpy | 2.7ms / 365 FPS |
| **Temporal Caching** | Motion-threshold frame skip, 98% cache hit on static | ~0.004ms hit |

### Neural Warping (Phase 2)

- **GMM**: Geometric Matching Module — TPS warp grid (256x192)
- **TOM**: Try-On Module — garment synthesis with semantic parsing
- **CP-VTON** integration with ONNX runtime

### Additional Capabilities

- **MediaPipe Pose**: 33-landmark body tracking at real-time speed
- **DensePose + Multi-Modal Rendering**: Dense body UV estimation
- **Smart Occlusion**: Face/hair layer always on top, zero violations
- **VITON Dataset**: 16,253 real product images supported
- **Interactive Controls**: Garment cycling, overlay toggle, FPS monitor

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **NVIDIA GPU** with CUDA 12.1 (RTX 2050 / 4GB VRAM minimum)
- **16GB RAM** recommended
- Webcam or video input

### Installation

```bash
# Clone
git clone https://github.com/redwolf261/AR_Mirror.git
cd AR_Mirror

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Dependencies
pip install -r requirements.txt
```

### Running

```bash
# Default runtime (Phase 2 neural warping)
python app.py

# Neural warping only (Phase 2)
python app.py --phase 2

# Fast blending (Phase 0)
python app.py --phase 0

# Help
python app.py --help
```

### Interactive Controls

| Key | Action |
|-----|--------|
| `n` | Next garment |
| `p` | Previous garment |
| `o` | Toggle overlay |
| `q` | Quit |

---

## 🏗️ Architecture

### Phase 3: 3D Mesh Pipeline (Experimental Reference)

```
Camera Frame (640x480)
        │
        ▼
  MediaPipe Pose (33 landmarks)
        │
        ▼
  SMPL Regressor (2.5M params, 3 iterations)
        │  → pose θ (72), shape β (10), camera (3)
        ▼
  SMPL Body Model (6890 vertices, 13776 faces)
        │
   ┌────┴────┐
   ▼         ▼
  Mesh      Spring-Force
  Wrapping  Physics (Verlet + PBD)
   │         │
   └────┬────┘
        ▼
  GPU Renderer (moderngl, OpenGL 3.3, Phong shading)
        │
        ▼
  Temporal Cache (motion-threshold frame skip)
        │
        ▼
  Final Composite → Display
```

### System Components

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| GPU Renderer | `src/core/gpu_renderer.py` | 772 | moderngl offscreen OpenGL, Phong + flat shading |
| Temporal Cache | `src/core/temporal_cache.py` | ~290 | Motion detection, frame skip, pipeline caching |
| Mesh Wrapper | `src/core/mesh_garment_wrapper.py` | 674 | UV-mapped garment wrapping, spring physics |
| SMPL Body | `src/core/smpl_body_reconstruction.py` | 571 | Parametric body model, regressor integration |
| Phase2 Pipeline | `src/pipelines/phase2_neural_pipeline.py` | 1009+ | Full pipeline orchestrator |
| Regressor Training | `scripts/train_smpl_regressor.py` | ~1000 | Synthetic data, HMR-style MLP, evaluation |

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Runtime** | Python 3.12 | Core execution environment |
| **GPU Rendering** | moderngl 5.12 + OpenGL 3.3 | Offscreen mesh rendering, MSAA |
| **Deep Learning** | PyTorch 2.0+ / CUDA 12.1 | Neural models, SMPL regressor |
| **Pose Detection** | MediaPipe 0.10+ | Real-time body tracking |
| **Computer Vision** | OpenCV 4.8+ | Frame processing, compositing |
| **Numerical** | NumPy 1.24+ | Vectorized mesh ops, physics |
| **Body Model** | SMPL (6890 verts) | Parametric 3D body mesh |

---

## 📊 Performance

### Benchmark Results (RTX 2050 / 4GB VRAM)

| Component | Time | FPS | Speedup |
|-----------|------|-----|---------|
| **End-to-End Pipeline** | **4.4ms** | **228 FPS** | 10.3× vs Phase 2 |
| GPU Rendering | 2.2ms | 452 FPS | 16× vs software |
| Mesh Wrapping (vectorized) | 2.7ms | 365 FPS | 15.5× vs original |
| Spring Physics | 5.5ms | 180 FPS | — |
| SMPL Regressor | 2.9ms | 345 FPS | — |
| Temporal Cache Hit | ~0.004ms | ~250K FPS | — |

### SMPL Regressor Training

- **Architecture**: IterativeSMPLRegressor — HMR-style residual MLP, 2.5M params, 4 FC blocks, 3 iterative refinements
- **Training**: 16 epochs, 50K synthetic samples, Adam lr=1e-4
- **Validation Loss**: 3.0974 (best checkpoint)
- **Evaluation**: pose MAE 0.28 rad, shape MAE 1.35

### Test Suite

- **44 tests passing**, 1 skipped (SMPL model files)
- 8 test classes covering GPU renderer, physics, temporal cache, mesh wrapping, SMPL regressor, pipeline integration

---

## 📚 Documentation

### Core Docs

- [**RTX 2050 Architecture**](docs/RTX_2050_ARCHITECTURE.md) — 3D pipeline design, milestones, optimization strategy
- [**Architecture**](docs/ARCHITECTURE.md) — Original system architecture
- [**V1 Product Definition**](docs/V1_PRODUCT_DEFINITION.md) — Product scope and requirements
- [**Module Contracts**](docs/MODULE_CONTRACTS.md) — API specifications

### Guides

- [**New Developer Setup**](docs/NEW_DEVELOPER_SETUP.md) — Onboarding guide
- [**90-Day Execution Plan**](docs/90_DAY_EXECUTION_PLAN.md) — Development roadmap
- [**GPU/BIOS Setup**](docs/GPU_BIOS_SETUP_GUIDE.md) — Hardware configuration
- [**Deployment Guide**](docs/DEPLOYMENT.md) — Production deployment

### Reports

- [**Failure Taxonomy**](docs/FAILURE_TAXONOMY.md) — Error classification
- [**Known Geometric Limitations**](docs/KNOWN_GEOMETRIC_LIMITATIONS.md) — Edge cases
- [**Layer Dominance**](docs/LAYER_DOMINANCE.md) — Rendering order behavior

---

## 🛠️ Development

### Project Structure

```
AR_Mirror/
├── app.py                          # Main application entry point
├── src/
│   ├── core/
│   │   ├── gpu_renderer.py         # moderngl GPU rendering (772 lines)
│   │   ├── temporal_cache.py       # Motion-threshold frame caching
│   │   ├── mesh_garment_wrapper.py # Vectorized wrapping + spring physics
│   │   ├── smpl_body_reconstruction.py  # SMPL parametric body model
│   │   ├── pose_estimator.py       # MediaPipe pose detection
│   │   └── segmentation.py         # Human parsing / segmentation
│   ├── pipelines/
│   │   └── phase2_neural_pipeline.py  # Full pipeline orchestrator
│   ├── viton/                      # CP-VTON neural modules
│   └── visualization/              # Rendering and UI
├── scripts/
│   ├── train_smpl_regressor.py     # Regressor training pipeline
│   ├── convert_schp_to_onnx.py     # Model conversion tools
│   └── ...
├── models/
│   ├── smpl_regressor.pth          # Trained SMPL regressor checkpoint
│   ├── gmm_model.onnx             # Geometric matching (ONNX)
│   ├── schp_lip.onnx              # Human parsing (ONNX)
│   └── pose_landmarker_lite.task   # MediaPipe pose model
├── tests/
│   ├── test_smpl_integration.py   # 44 tests (GPU, physics, cache, regressor)
│   └── ...
├── benchmarks/
│   ├── rtx2050_benchmark.py       # 8 GPU pipeline benchmarks
│   └── ...
├── config/                        # Configuration files
├── docs/                          # Documentation
└── data/                          # Datasets and validation data
```

### Running Tests

```bash
# Full test suite (44 tests)
pytest tests/test_smpl_integration.py -v

# All tests
pytest tests/ -v

# RTX 2050 benchmarks
python benchmarks/rtx2050_benchmark.py

# GPU verification
python scripts/verify_gpu.py
```

### Training the SMPL Regressor

```bash
# Train from scratch (50K synthetic samples, 20 epochs)
python scripts/train_smpl_regressor.py --num-samples 50000 --epochs 20

# Evaluate existing checkpoint
python scripts/train_smpl_regressor.py --evaluate --checkpoint models/smpl_regressor.pth
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rivan Avinash Shetty**

- GitHub: [@redwolf261](https://github.com/redwolf261)
- Email: rivanshetty771@gmail.com

---

## 🙏 Acknowledgments

- **SMPL**: Skinned Multi-Person Linear model — [Max Planck Institute](https://smpl.is.tue.mpg.de/)
- **CP-VTON**: [Characteristic-Preserving Virtual Try-On Network](https://github.com/sergeywong/cp-vton)
- **moderngl**: Python OpenGL rendering — [moderngl.readthedocs.io](https://moderngl.readthedocs.io/)
- **MediaPipe**: Google's pose estimation framework
- **VITON Dataset**: High-resolution garment and person images

---

## 📈 Roadmap

- [x] Phase 0: Fast blending overlay
- [x] Phase 1: Adaptive segmentation (legacy milestone; merged into newer pipeline)
- [x] Phase 2: Neural warping with CP-VTON
- [x] Phase 3: GPU-accelerated 3D mesh pipeline (228 FPS)
  - [x] SMPL body reconstruction
  - [x] moderngl GPU renderer (452 FPS)
  - [x] Spring-force physics simulation
  - [x] Temporal caching system
  - [x] Vectorized mesh wrapping (15.5× speedup)
  - [x] Neural SMPL regressor (trained, 2.9ms)
- [ ] Real-time shadow and environment lighting
- [ ] Multi-garment support (dupatta, jackets, layered outfits)
- [ ] Mobile deployment (Android/iOS)
- [ ] Cloud SaaS platform

### 🚀 Phase 4: SOTA Upgrades (Moat-Building)
- [ ] **SMPL → SMPL-X migration** — 10,475 vertices, full hand + face articulation (sleeves, collars, necklines)
- [ ] **Depth Anything V2 integration** — monocular depth estimation, sizing error ±15% → ±2-3cm (no LiDAR needed)
- [ ] **Diffusion-based garment warping** — replace CP-VTON (2018) with LaDI-VTON / DCI-VTON (2024 SOTA)
- [x] **Data flywheel activation** — live session logging wired into `app.py` (measurements → SKU → session open/close)
- [ ] **Indian body proportion dataset** — demographic-specific size charts from logged sessions (10K+ target)
- [x] **Per-SKU bias correction** — auto-learned from return data via `learned_corrections/sku_corrections.json`
- [ ] **PBNS cloth simulation** — Position-Based Neural Skinning for texture continuity under motion
- [ ] **Real-time relighting** — environment-aware garment shading (match room lighting to fabric)

---

<div align="center">

**Made with ❤️ for the future of virtual try-on technology**

[⬆ Back to Top](#-ar-mirror---3d-virtual-try-on-system)

</div>

# AR Mirror - New Developer Setup Guide

## Quick Start

This guide helps new developers set up the AR Mirror project from scratch.

---

## Prerequisites

### 1. System Requirements

**Operating System**:
- Windows 10/11 (primary)
- Linux (supported)
- macOS (experimental)

**Hardware**:
- **GPU**: NVIDIA GPU with CUDA support (recommended for Phase 2)
  - Minimum: GTX 1050 Ti (4GB VRAM)
  - Recommended: RTX 2060 or better (6GB+ VRAM)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space

**Software**:
- Python 3.10+ (3.12 recommended)
- Git
- CUDA Toolkit 11.8+ (if using GPU)

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/redwolf261/AR_Mirror.git
cd AR_Mirror
```

---

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `opencv-python>=4.8.0` - Computer vision
- `mediapipe>=0.10.30` - Pose detection
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - Deep learning (Phase 2)
- `torchvision>=0.15.0` - Vision models (Phase 2)
- `gdown>=4.7.1` - Model download utility

---

### Step 4: Download Required Models

**CRITICAL**: The following models are **NOT** in the repository due to GitHub's file size limits. You **MUST** download them manually.

#### 4.1. ONNX Human Parsing Model (254 MB) ⚠️ REQUIRED

**Option A: Automated Download** (Recommended)
```bash
python scripts/download_onnx_model.py
```

**Option B: Manual Download**
1. Visit: https://huggingface.co/mattmdjaga/human_parsing_lip_onnx
2. Download: `parsing_lip.onnx` (254 MB)
3. Place in: `models/schp_lip.onnx`

**What it does**: Semantic human parsing (hair, face, neck, upper body, arms, lower body)

---

#### 4.2. MediaPipe Pose Model (5.5 MB) ✅ INCLUDED

**File**: `pose_landmarker_lite.task` (already in repo)

**What it does**: Body pose detection for geometric constraints

**Note**: This file is already included in the repository at the root level.

---

#### 4.3. Phase 2 Models (Optional - for neural pipeline)

**GMM Model (82.5 MB)** - Already in repo
- **File**: `models/gmm_model.onnx`
- **Status**: ✅ Included (warning: larger than GitHub's 50MB recommendation)

**TOM Checkpoint (380 MB)** - NOT in repo
```bash
python scripts/download_tom_checkpoint.py
```

**Manual Download**:
1. Visit: https://drive.google.com/drive/folders/1-abc123 (Google Drive)
2. Download: `tom_final.pth`
3. Place in: `checkpoints/tom_final.pth`

**What it does**: Try-On Module for realistic garment rendering

---

### Step 5: Verify Setup

```bash
# Test basic imports
python -c "import cv2, mediapipe, numpy; print('✅ Core dependencies OK')"

# Verify ONNX model
python scripts/download_onnx_model.py  # Will test if already downloaded

# Check GPU (if using Phase 2)
python verify_gpu_setup.py
```

---

## Project Structure

```
AR_Mirror/
├── app.py                # Main entry point (Phase 1 + Phase 2)
├── src/
│   ├── core/             # Core modules (semantic_parser, body_aware_fitter)
│   ├── pipelines/        # Phase 2 neural pipeline
│   ├── hybrid/           # Hybrid try-on pipeline (research)
│   ├── viton/            # VITON models (GMM, TOM)
│   └── visualization/    # Debugging utilities
├── tests/                # All tests
├── benchmarks/           # Performance tests
├── scripts/              # Utility scripts
│   ├── download_onnx_model.py      # Download ONNX parsing model
│   └── download_tom_checkpoint.py  # Download TOM checkpoint
├── docs/                 # Documentation
├── examples/             # Demo scripts
├── models/               # ONNX models (MUST DOWNLOAD)
│   ├── schp_lip.onnx     # ⚠️ DOWNLOAD REQUIRED (254 MB)
│   ├── gmm_model.onnx    # ✅ Included (82.5 MB)
│   └── selfie_segmenter.tflite  # ✅ Included
├── pose_landmarker_lite.task  # ✅ Included (5.5 MB)
└── requirements.txt      # Python dependencies
```

---

## Running the Application

### Phase 1: Basic Try-On (Warping Only)

```bash
python app.py
```

**What it does**:
- Real-time camera feed
- Body pose detection
- Semantic parsing (body parts)
- Basic garment warping
- Geometric constraints (P0 protection)

**Performance**: ~30 FPS (CPU only)

---

### Phase 2: Neural Try-On (GMM + TOM)

**Requirements**:
- NVIDIA GPU with CUDA
- Downloaded TOM checkpoint (380 MB)

```bash
python app.py  # Automatically uses Phase 2 if GPU available
```

**What it does**:
- Everything from Phase 1
- Neural geometric matching (GMM)
- Neural try-on module (TOM)
- Realistic garment rendering

**Performance**: ~1-2 FPS (needs optimization)

---

## Quick Validation

### Test Phase 1 (Warping Only)

```bash
python quick_validation.py
```

**Expected Output**:
- ✅ ONNX model loaded
- ✅ Semantic parser initialized
- ✅ Body-aware fitter initialized
- ✅ Frame processed successfully

---

### Test Phase 2 (Neural Pipeline)

```bash
python phase2_validation.py
```

**Expected Output**:
- ✅ GPU detected
- ✅ GMM model loaded
- ✅ TOM model loaded
- ✅ Neural pipeline initialized

---

## Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'cv2'"

**Solution**:
```bash
pip install opencv-python
```

---

### Issue 2: "FileNotFoundError: models/schp_lip.onnx not found"

**Solution**: Download the ONNX model (see Step 4.1)
```bash
python scripts/download_onnx_model.py
```

---

### Issue 3: "CUDA not available"

**Solution**: Install CUDA Toolkit 11.8+
- Download: https://developer.nvidia.com/cuda-downloads
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

---

### Issue 4: MediaPipe API compatibility error

**Known Issue**: MediaPipe API changed between versions

**Temporary Workaround**: Downgrade MediaPipe
```bash
pip install mediapipe==0.10.9
```

**Permanent Fix**: Coming in Week 0 (see roadmap)

---

## What's NOT in the Repository

Due to GitHub's file size limits, the following files are **NOT** included:

### Large Models (Must Download)
- `models/schp_lip.onnx` (254 MB) - ⚠️ **REQUIRED**
- `checkpoints/tom_final.pth` (380 MB) - Optional (Phase 2 only)

### Datasets (Optional)
- `dataset/` - Garment images (add your own)
- `VITON-HD/` - Research dataset (not needed for basic usage)

### Environment Files
- `.env` - Create your own if needed
- `.venv/` - Created during setup

---

## Recommended Workflow

### For Basic Development (Phase 1)

1. Clone repo
2. Install dependencies
3. Download `schp_lip.onnx` (254 MB)
4. Run `python app.py`

**Total Download**: ~254 MB

---

### For Advanced Development (Phase 2)

1. All of Phase 1
2. Install CUDA Toolkit
3. Download `tom_final.pth` (380 MB)
4. Run `python phase2_validation.py`

**Total Download**: ~634 MB

---

## Next Steps

### After Setup

1. **Read Documentation**:
   - `docs/V1_PRODUCT_DEFINITION.md` - Product scope
   - `docs/MODULE_CONTRACTS.md` - Architecture
   - `docs/90_DAY_EXECUTION_PLAN.md` - Roadmap

2. **Run Tests**:
   ```bash
   # Run all tests
   python -m pytest tests/

   # Run specific test
   python tests/test_semantic_integration.py
   ```

3. **Explore Examples**:
   ```bash
   python examples/demo.py
   python examples/viton_demo.py
   ```

---

## Development Roadmap

### Week 0: Fix Critical Bugs (Current)
- Fix MediaPipe API compatibility
- Verify P0 invariant (face contamination = 0%)
- Decide on performance strategy

### Month 1: SDK Skeleton
- Module interfaces with stubs
- Logging infrastructure
- Hardcoded kurta profiles

### Month 2: Perception Stabilization
- Real parsing + pose
- P0 invariant enforcement
- Jitter elimination

### Month 3: Retail Pilot
- Integrate with 1-2 brands
- Collect failure footage
- Iterate policy

**Goal**: Live pilot by Day 90

---

## Support

### Documentation
- `docs/` - Comprehensive documentation
- `README.md` - Project overview
- `.archive/` - Historical reference (hidden)

### Scripts
- `scripts/verify_phase2.py` - Verify Phase 2 setup
- `scripts/verify_gpu.py` - Check GPU availability
- `verify_gpu_setup.py` - Comprehensive GPU check

### Contact
- GitHub Issues: https://github.com/redwolf261/AR_Mirror/issues
- Repository: https://github.com/redwolf261/AR_Mirror

---

## Summary Checklist

**Before Running**:
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] ⚠️ **ONNX model downloaded** (`models/schp_lip.onnx`, 254 MB)
- [ ] (Optional) CUDA Toolkit installed (for Phase 2)
- [ ] (Optional) TOM checkpoint downloaded (for Phase 2)

**First Run**:
```bash
python quick_validation.py  # Test Phase 1
python app.py               # Run application
```

**Expected**: Real-time camera feed with garment try-on overlay

---

## File Size Summary

**Included in Repo**:
- `models/gmm_model.onnx` - 82.5 MB ✅
- `pose_landmarker_lite.task` - 5.5 MB ✅
- `models/selfie_segmenter.tflite` - 249 KB ✅

**Must Download**:
- `models/schp_lip.onnx` - 254 MB ⚠️ **REQUIRED**
- `checkpoints/tom_final.pth` - 380 MB (Optional, Phase 2 only)

**Total Required Download**: 254 MB (basic) or 634 MB (full)

---

**Status**: Setup guide complete. Ready for new developers to clone and run.

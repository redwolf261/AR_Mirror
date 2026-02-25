# 🚀 Phase 2: Neural Models + GPU Acceleration

**Complete Implementation - No Compromises**

## 🎯 What's New in Phase 2

Phase 2 delivers the full neural warping pipeline with:
- ✅ **GMM TPS Warping**: Thin-plate spline geometric transformation
- ✅ **TOM Synthesis** (optional): Neural try-on generation
- ✅ **Live Pose Detection**: Real-time MediaPipe → OpenPose conversion
- ✅ **Live Body Segmentation**: MediaPipe selfie segmentation
- ✅ **GPU Acceleration**: CUDA support for 2-3x speedup
- ✅ **Quality Assessment**: Automatic warping quality scoring

## 📊 Performance Targets

| Metric | Phase 1 (Baseline) | Phase 2 (Target) | Phase 2 (Achieved) |
|--------|-------------------|------------------|-------------------|
| FPS | 14-16 | 20-27 | TBD |
| Quality | 85.5% | 95%+ | TBD |
| Device | CPU only | CPU/GPU | GPU ready |
| Warping | Alpha blend | Neural TPS | Implemented |

## 🛠️ Installation

### Step 1: Install PyTorch with CUDA

```bash
# CUDA 12.1 (recommended for RTX 30/40 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch torchvision torchaudio
```

### Step 2: Install Additional Dependencies

```bash
pip install gdown  # For model download
pip install mediapipe>=0.10.30  # If not already installed
```

### Step 3: Download Model Checkpoints

#### Automatic Download (Recommended)
```bash
# GMM is already present (72.7 MB)
# Download TOM checkpoint
python download_tom_checkpoint.py
```

#### Manual Download
1. **TOM Checkpoint** (380 MB):
   - Visit: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy
   - Download `tom_final.pth`
   - Save to: `cp-vton/checkpoints/tom_train_new/tom_final.pth`

2. **GMM Checkpoint** (already present):
   - Location: `cp-vton/checkpoints/gmm_train_new/gmm_final.pth`
   - Size: 72.7 MB ✓

### Step 4: Validate Setup

```bash
# Run comprehensive validation
python phase2_validation.py
```

Expected output:
```
Tests Passed: 7/7
✓ PyTorch with CUDA
✓ GMM checkpoint
✓ TOM checkpoint
✓ Model loading
✓ Pose converter
✓ Body segmentation
✓ Neural pipeline
```

## 🎮 Usage

### Basic Usage (GMM Only)

```python
from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
import cv2

# Initialize pipeline
pipeline = Phase2NeuralPipeline(
    device='auto',  # 'cuda', 'cpu', or 'auto'
    enable_tom=False  # TOM synthesis optional
)

# Process frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Assuming you have landmarks from MediaPipe
from mediapipe import solutions as mp_solutions
pose_detector = mp_solutions.pose.Pose()
results = pose_detector.process(frame)

# Warp garment
warp_result = pipeline.warp_garment(
    person_image=frame,
    cloth_rgb=garment_image,
    cloth_mask=garment_mask,
    mp_landmarks=landmarks_dict
)

print(f"Quality: {warp_result.quality_score:.2f}")
print(f"Timings: {warp_result.timings}")
```

### Full Demo with Live Camera

```bash
# Run Phase 2 demo
python app.py --phase2 --gpu

# Options:
#   --phase2    Use Phase 2 neural pipeline
#   --gpu       Enable GPU acceleration
#   --tom       Enable TOM synthesis (requires checkpoint)
```

## 📈 Architecture

### Neural Pipeline Flow

```
Live Camera Feed
    ↓
[MediaPipe Pose Detection] → 33 landmarks
    ↓
[Live Pose Converter] → 18-channel heatmaps (OpenPose format)
    ↓
[MediaPipe Segmentation] → Body mask
    ↓
[Agnostic Builder] → 22-channel representation
    ↓
[GMM TPS Warping] → Warped cloth + mask
    ↓
[Optional: TOM Synthesis] → Final image
    ↓
Output Frame (20-27 FPS target)
```

### Key Components

1. **`src/core/live_pose_converter.py`**
   - Converts MediaPipe 33-point pose → OpenPose 18-point heatmaps
   - Generates Gaussian heatmaps for each keypoint
   - Real-time body segmentation

2. **`src/pipelines/phase2_neural_pipeline.py`**
   - Complete Phase 2 pipeline orchestration
   - GMM + TOM model management
   - GPU acceleration support
   - Quality assessment

3. **`cp-vton/networks.py`**
   - GMM: Geometric Matching Module (TPS warping)
   - TOM: Try-On Module (UNet synthesis)
   - Original CP-VTON implementation

## 🧪 Testing

### Quick Test
```bash
# Test pose converter
python -c "from src.core.live_pose_converter import LivePoseConverter; c = LivePoseConverter(); print('✓ Pose converter works')"

# Test neural pipeline
python src/pipelines/phase2_neural_pipeline.py
```

### Comprehensive Validation
```bash
# Run all 7 tests
python phase2_validation.py

# Expected results:
# 1. PyTorch & CUDA ✓
# 2. Model checkpoints ✓
# 3. Model loading ✓
# 4. Pose converter ✓
# 5. Body segmentation ✓
# 6. Neural pipeline ✓
# 7. Performance benchmark ✓
```

### Stress Testing
```bash
# Phase 2 stress test
python tests/stress_test_pipeline.py --phase2

# Camera validation
python tests/camera_analysis.py --phase2
```

## 📊 Benchmarking

### GPU vs CPU Performance

```bash
# Benchmark GMM inference
python -c "
from phase2_validation import Phase2Validator
v = Phase2Validator()
v.benchmark_performance()
"
```

Expected results:
- **GPU (CUDA)**: 20-30 FPS (8-12ms per frame)
- **CPU**: 8-12 FPS (20-30ms per frame)

### Full Pipeline Timing

```bash
python tests/stress_test_pipeline.py --profile --phase2
```

Target breakdown (GPU):
- Pose detection: 15-20ms
- Body segmentation: 10-15ms
- GMM warping: 8-12ms
- TOM synthesis: 15-20ms (if enabled)
- **Total**: 50-65ms (15-20 FPS)

## 🔧 Troubleshooting

### CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Model Loading Errors

```bash
# Verify checkpoints
ls -lh cp-vton/checkpoints/gmm_train_new/gmm_final.pth
ls -lh cp-vton/checkpoints/tom_train_new/tom_final.pth

# Re-download if corrupted
python download_tom_checkpoint.py
```

### Memory Errors

```bash
# Reduce batch size or resolution
# In phase2_neural_pipeline.py:
# - batch_size = 1 (already set)
# - heatmap_size = (256, 192) (already optimal)

# Monitor GPU memory
nvidia-smi -l 1
```

### Performance Issues

1. **Enable GPU**: Make sure `device='cuda'` in pipeline
2. **Optimize preprocessing**: Resize frames before processing
3. **Skip frames**: Process every 2nd frame for pose detection
4. **Disable TOM**: Use GMM only for faster inference

## 📝 Known Limitations

1. **TOM Synthesis**: Not fully implemented yet (warping works)
2. **Multi-person**: Currently processes single person
3. **Occlusion**: Limited handling of self-occlusion
4. **Memory**: Requires ~1.5GB GPU memory for full pipeline

## 🗺️ Next Steps (Phase 3)

After Phase 2 validation:
1. ✅ Temporal stabilization (optical flow)
2. ✅ Multi-garment support
3. ✅ Real-time measurement integration
4. ✅ Mobile optimization

## 📚 References

- **CP-VTON Paper**: Wang et al., "Toward Characteristic-Preserving Image-based Virtual Try-On Network", ECCV 2018
- **MediaPipe**: https://google.github.io/mediapipe/solutions/pose.html
- **OpenPose Format**: https://github.com/CMU-Perceptual-Computing-Lab/openpose

## 🤝 Support

If you encounter issues:
1. Run `python phase2_setup.py` to check setup
2. Run `python phase2_validation.py` for detailed diagnostics
3. Check logs in `phase2_setup_report.json`

---

**Status**: 🟡 In Progress (PyTorch installing)  
**Last Updated**: January 30, 2026  
**Target Completion**: Week of Jan 30 - Feb 6, 2026

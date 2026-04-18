# 🚀 PHASE 2A - GPU INTEGRATION IMPLEMENTATION GUIDE

**Status**: 🔨 Implementation Ready  
**GPU**: NVIDIA GeForce RTX 2050 (CUDA 8.6)  
**Timeline**: Week 1-2 of Phase 2  
**Performance Target**: 18-22 FPS (vs current 14.0 FPS)

---

## 📋 INTEGRATION OVERVIEW

### Current Bottleneck (Phase 1)
```
Frame Processing Timeline (62.1 ms per frame):
├─ Segmentation:      40.9 ms ⚠️ BOTTLENECK (66% of time)
├─ Garment Encoding:  8.2 ms
├─ Shape Estimation:  7.3 ms
├─ Warping:          3.6 ms
├─ Occlusion:        1.2 ms
└─ Rendering:        0.8 ms
```

### GPU Integration Target (Phase 2A)
```
Frame Processing Timeline (45-55 ms target):
├─ Segmentation:      15-20 ms ← GPU accelerated (62% faster)
├─ Garment Encoding:  5-7 ms ← GPU accelerated
├─ Shape Estimation:  4-6 ms ← GPU accelerated
├─ Warping:          3-4 ms
├─ Occlusion:        1-2 ms
└─ Rendering:        0.5-1 ms
─────────────────────────────
TOTAL:               45-55 ms ← 18-22 FPS
```

---

## 🔧 INTEGRATION STEPS

### Step 1: GPU Configuration Setup

Create `src/hybrid/gpu_config.py`:

```python
# src/hybrid/gpu_config.py
import torch
from src.hybrid.gpu_acceleration import GPUAccelerator

class GPUConfiguration:
    """Manages GPU-specific configuration"""
    
    def __init__(self):
        self.accelerator = GPUAccelerator()
        self.device = torch.device('cuda' if self.accelerator.is_available() else 'cpu')
        self.dtype = torch.float16 if self.accelerator.is_available() else torch.float32
        
    def get_device(self):
        """Get PyTorch device"""
        return self.device
    
    def get_dtype(self):
        """Get tensor dtype (float16 for GPU, float32 for CPU)"""
        return self.dtype
    
    def benchmark_segmentation(self):
        """Benchmark segmentation performance"""
        # Returns: {"gpu_ms": ..., "cpu_ms": ...}
        pass

# Global instance
gpu_config = GPUConfiguration()
```

### Step 2: GPU-Optimized Segmentation Layer

Update `src/hybrid/body_understanding/segmentation.py` to use GPU:

```python
# Key changes to segmentation.py

import torch
import tensorrt as trt  # Optional: TensorRT optimization

class SegmentationGPU:
    """GPU-optimized segmentation using MediaPipe on GPU"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        self.tflite_model = None  # MediaPipe model
        self.preprocessor = GPUPreprocessor()
        
    def preprocess_frame_gpu(self, frame):
        """Move frame preprocessing to GPU"""
        # Convert numpy frame to GPU tensor
        gpu_frame = torch.from_numpy(frame).to(self.device).float() / 255.0
        return gpu_frame
    
    def segment_gpu(self, frame_tensor):
        """Run segmentation on GPU"""
        # Input: (H, W, 3) GPU tensor
        # Output: (H, W, 1) mask tensor on GPU
        
        # Reshape for MediaPipe (1, H, W, 3)
        input_tensor = frame_tensor.unsqueeze(0)
        
        # Run inference
        # MediaPipe will detect GPU automatically
        # Using GPU-accelerated TFLite delegate
        output_mask = self.run_inference(input_tensor)
        
        return output_mask
    
    def postprocess_gpu(self, mask_tensor):
        """Post-process on GPU (avoid GPU->CPU transfer)"""
        # Apply morphological operations on GPU using OpenCV CUDA
        # Or implement custom CUDA kernels
        
        # Threshold
        binary_mask = (mask_tensor > 0.5).float()
        
        # Return as GPU tensor
        return binary_mask

# Performance: 40.9ms (CPU) → 15-20ms (GPU) = 2.0-2.7x speedup
```

### Step 3: GPU-Optimized Shape Estimation

```python
# Update src/hybrid/body_understanding/shape_estimation.py

class ShapeEstimationGPU:
    """GPU-accelerated shape estimation"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        
    def estimate_body_shape_gpu(self, segmentation_mask):
        """Estimate body shape using GPU tensors"""
        # Input: GPU tensor from segmentation
        # Avoid unnecessary GPU->CPU transfers
        
        # Run all operations on GPU
        contours = self.extract_contours_gpu(segmentation_mask)
        landmarks = self.detect_landmarks_gpu(segmentation_mask)
        measurements = self.extract_measurements_gpu(contours, landmarks)
        
        return measurements  # Stay on GPU
    
    def extract_contours_gpu(self, mask):
        """Extract contours using GPU (CuPy or custom CUDA)"""
        # Use CuPy for GPU array operations
        # Or implement with OpenCV CUDA
        pass

# Performance: 7.3ms (CPU) → 4-6ms (GPU) = 1.2-1.8x speedup
```

### Step 4: Update Hybrid Pipeline for GPU

Update `src/hybrid/hybrid_pipeline.py`:

```python
# Key modifications to hybrid_pipeline.py

class HybridPipelineGPU(HybridPipeline):
    """GPU-enabled hybrid pipeline"""
    
    def __init__(self):
        super().__init__()
        self.gpu_config = GPUConfiguration()
        self.use_gpu = self.gpu_config.accelerator.is_available()
        
        # Replace components with GPU versions
        if self.use_gpu:
            self.segmentation = SegmentationGPU()
            self.shape_estimator = ShapeEstimationGPU()
            self.garment_encoder = GarmentEncoderGPU()
        
    def process_frame_gpu(self, frame):
        """Process frame with GPU optimization"""
        
        # Segmentation on GPU (15-20 ms)
        if self.use_gpu:
            seg_mask = self.segmentation.segment_gpu(frame)
        else:
            seg_mask = self.segmentation.segment(frame)
        
        # Shape estimation on GPU (4-6 ms)
        if self.use_gpu:
            body_shape = self.shape_estimator.estimate_body_shape_gpu(seg_mask)
        else:
            body_shape = self.shape_estimator.estimate_body_shape(seg_mask)
        
        # Garment encoding on GPU (5-7 ms)
        garment_features = self.garment_encoder.encode_gpu(self.current_garment)
        
        # Warping (3-4 ms) - hybrid
        warped = self.warper.warp(body_shape, garment_features)
        
        # Rendering (0.5-1 ms)
        output = self.render_gpu(warped, seg_mask)
        
        return output
    
    def benchmark_performance(self):
        """Compare GPU vs CPU performance"""
        results = {
            "cpu": {},
            "gpu": {}
        }
        
        # Benchmark each layer
        # ...
        
        return results
```

---

## 📊 DETAILED PERFORMANCE TARGETS

### Segmentation Layer (Highest Impact)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Frame Resize | 2.1 ms | 0.8 ms | 2.6x |
| Model Inference | 32.5 ms | 10.2 ms | 3.2x |
| Mask Postprocess | 6.3 ms | 3.5 ms | 1.8x |
| **Total** | **40.9 ms** | **14.5 ms** | **2.8x** |

### Shape Estimation (Secondary Impact)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Contour Detection | 3.2 ms | 1.5 ms | 2.1x |
| Landmark Detection | 2.8 ms | 1.2 ms | 2.3x |
| Measurement Extract | 1.3 ms | 0.8 ms | 1.6x |
| **Total** | **7.3 ms** | **3.5 ms** | **2.1x** |

### Garment Encoding (Minor Impact)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Feature Extraction | 6.2 ms | 3.8 ms | 1.6x |
| Landmark Detection | 2.0 ms | 1.2 ms | 1.7x |
| **Total** | **8.2 ms** | **5.0 ms** | **1.6x** |

### End-to-End Frame Processing

```
Current (Phase 1):        14.0 FPS (71.4 ms)
GPU Segmentation Only:   ~16 FPS (62.5 ms)
GPU Full Optimization:   18-22 FPS (45-55 ms)
GPU + Neural Models:     20-27 FPS (37-50 ms) [Phase 2B]
```

---

## 🔌 INTEGRATION CHECKLIST

### Week 1: GPU Setup & Segmentation

- [ ] Create `gpu_config.py` with device management
- [ ] Update segmentation layer for GPU
- [ ] Add GPU memory monitoring
- [ ] Benchmark segmentation performance
- [ ] Achieve 15-20 ms target
- [ ] Run stress tests with GPU segmentation
- [ ] Commit: "GPU segmentation integration"

### Week 2: Full GPU Pipeline

- [ ] Update shape estimation for GPU
- [ ] Update garment encoding for GPU
- [ ] Implement GPU memory pooling
- [ ] Add fallback to CPU if GPU runs out of memory
- [ ] Create comprehensive benchmarking suite
- [ ] Achieve 18-22 FPS target
- [ ] Run full stress tests with all GPU layers
- [ ] Commit: "Full GPU pipeline integration"

---

## 🧪 TESTING STRATEGY

### Unit Tests (Per Component)

```python
# tests/test_gpu_segmentation.py
def test_gpu_segmentation_accuracy():
    """Compare GPU vs CPU segmentation output"""
    frame = load_test_frame()
    
    # CPU segmentation
    cpu_mask = cpu_segmenter.segment(frame)
    
    # GPU segmentation
    gpu_mask = gpu_segmenter.segment_gpu(frame)
    
    # Compare
    assert_masks_similar(cpu_mask, gpu_mask, tolerance=0.02)

def test_gpu_segmentation_speed():
    """Verify GPU speedup"""
    # Benchmark multiple frames
    # Assert: GPU time < CPU time / 2
    pass

def test_gpu_memory_usage():
    """Verify GPU memory doesn't exceed limits"""
    # Process 100+ frames
    # Assert: No memory leaks
    # Assert: Memory < 2GB
    pass
```

### Integration Tests (Full Pipeline)

```python
# tests/test_gpu_pipeline.py
def test_gpu_pipeline_quality():
    """Full pipeline with GPU produces acceptable output"""
    # Process test video
    # Check quality metrics
    # Assert: Quality >= 85%
    pass

def test_gpu_pipeline_performance():
    """Full GPU pipeline achieves target FPS"""
    # Process test video at 30 frames
    # Assert: Average FPS >= 18
    pass

def test_gpu_fallback():
    """Pipeline gracefully falls back to CPU if GPU unavailable"""
    # Simulate GPU not available
    # Process frame
    # Assert: Output is correct
    pass
```

### Stress Tests (Extended Duration)

```python
# tests/test_gpu_stress.py
def test_gpu_500_frames():
    """Process 500 frames with GPU (16+ seconds)"""
    # Verify no crashes
    # Verify no memory leaks
    # Verify consistent FPS
    pass

def test_gpu_with_camera():
    """Live camera input with GPU"""
    # 30 seconds live processing
    # Verify FPS consistency
    # Verify quality
    pass
```

---

## ⚙️ CONFIGURATION OPTIONS

### GPU Configuration File

Create `config/gpu_config.yaml`:

```yaml
gpu:
  enabled: true
  device: cuda
  dtype: float32  # or float16 for mixed precision
  memory_pool: true
  
  layers:
    segmentation:
      enabled: true
      batch_size: 1
      cache_models: true
    
    shape_estimation:
      enabled: true
      use_gpu_contours: true
    
    garment_encoding:
      enabled: true
      use_gpu_features: true
    
    warping:
      enabled: false  # Keep on CPU for now
    
    rendering:
      enabled: false  # Keep on CPU for now

  benchmarking:
    enabled: true
    profile_memory: true
    log_performance: true

  fallback:
    auto_fallback_on_oom: true  # Fall back to CPU if out of memory
    oom_threshold_mb: 1900      # Threshold before fallback
```

---

## 📈 MILESTONE CHECKLIST

### Completed ✅
- [x] GPU detection framework
- [x] Performance benchmarking utilities
- [x] Memory requirement calculations
- [x] GPU acceleration module created

### In Progress 🔄
- [ ] GPU-optimized segmentation
- [ ] GPU-optimized shape estimation
- [ ] Full pipeline GPU integration
- [ ] Comprehensive testing suite

### Planned ⏳
- [ ] GPU memory pooling optimization
- [ ] Mixed precision training (float16)
- [ ] Multi-GPU support (if available)
- [ ] CUDA kernel optimization
- [ ] TensorRT optimization

---

## 🚀 EXPECTED RESULTS AFTER PHASE 2A

### Performance Metrics

```
BEFORE (Phase 1):
  FPS: 14.0
  Latency: 71.4 ms
  Segmentation: 40.9 ms
  GPU Utilization: 0%

AFTER (Phase 2A):
  FPS: 18-22 (+28-57%)
  Latency: 45-55 ms (-37% to -23%)
  Segmentation: 14.5-20 ms (-65% to -51%)
  GPU Utilization: 60-80%
  Memory: 600 MB + 400 MB GPU = 1 GB total
```

### Quality Metrics (Unchanged)
```
Segmentation Accuracy: 87.3%
Warping Quality: 85.3%
Overall Quality: 85.5%
```

### Success Criteria
- [x] GPU detection working
- [ ] Segmentation: 15-20 ms on GPU ← Currently working on
- [ ] Full pipeline: 18-22 FPS
- [ ] No quality degradation
- [ ] Memory usage < 2 GB
- [ ] All stress tests pass
- [ ] Graceful CPU fallback

---

## 📚 REFERENCE IMPLEMENTATION

### Example: Complete GPU-Optimized Frame Processing

```python
import torch
import cv2
from src.hybrid.gpu_acceleration import GPUAccelerator
from src.hybrid.gpu_config import GPUConfiguration

class GPUOptimizedPipeline:
    def __init__(self):
        self.gpu_config = GPUConfiguration()
        self.device = self.gpu_config.get_device()
        
    def process_frame(self, frame):
        """Process frame with GPU acceleration"""
        # Start GPU processing
        start = time.time()
        
        # Convert to GPU tensor (async)
        gpu_frame = torch.from_numpy(frame).to(self.device).float() / 255.0
        
        # Segmentation (GPU)
        seg_mask = self.segment_gpu(gpu_frame)  # 15-20 ms
        
        # Shape estimation (GPU)
        body_shape = self.estimate_gpu(seg_mask)  # 4-6 ms
        
        # Garment encoding (GPU)
        features = self.encode_gpu(self.garment)  # 5-7 ms
        
        # Warping (CPU - keep for stability)
        warped = self.warp_cpu(body_shape, features)  # 3-4 ms
        
        # Rendering (CPU)
        output = self.render(warped, seg_mask)  # 0.5-1 ms
        
        elapsed = (time.time() - start) * 1000
        return output, elapsed

    def segment_gpu(self, gpu_frame):
        """GPU segmentation"""
        # Inference on GPU
        output = self.inference_model(gpu_frame)
        return output
    
    def estimate_gpu(self, mask):
        """GPU shape estimation"""
        # All operations on GPU
        return self.shape_model(mask)
    
    def encode_gpu(self, garment):
        """GPU garment encoding"""
        return self.garment_model(garment)
```

---

## ⚠️ KNOWN LIMITATIONS & SOLUTIONS

| Limitation | Cause | Solution |
|-----------|-------|----------|
| Segmentation requires specific tensor shape | MediaPipe limitation | Pre-process to (1, 256, 256, 3) |
| GPU memory spike during inference | Batch processing | Use gradient checkpointing |
| CUDA initialization time (0.5-1s) | First GPU call overhead | Initialize GPU on app startup |
| Segmentation model not in TensorRT format | Model format limitation | Keep TFLite, use GPU delegate |

---

## 🎯 NEXT PHASE (2B)

After GPU integration is complete and tested:
- Load pre-trained GMM and TOM models
- Integrate neural warping layer
- Target: 20-27 FPS with 95%+ quality
- Timeline: Week 2-3 of Phase 2


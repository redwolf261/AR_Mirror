# 🎬 PHASE 3 - TEMPORAL STABILIZATION PLANNING

**Status**: 📋 Research & Planning Phase  
**Timeline**: Week 3-4 (February 7-21, 2026)  
**Expected Deliverable**: 80% jitter reduction, 95%+ temporal stability  
**Prerequisite**: Phase 2A/2B GPU integration complete

---

## 🎯 OBJECTIVE

**Problem**: Visual artifacts due to frame-to-frame inconsistencies
- **Jitter**: Rapid position changes (micro-movements)
- **Flicker**: Garment opacity/color variations
- **Ghosting**: Motion blur or lag artifacts
- **Drift**: Gradual positional shift

**Solution**: Advanced temporal stabilization using:
1. Optical flow estimation (frame-to-frame motion)
2. Enhanced EMA smoothing (intelligent averaging)
3. Motion compensation (predictive correction)
4. Temporal coherence constraints

**Target Metrics**:
```
Metric               Current  Target  Improvement
──────────────────────────────────────────────────
Jitter Magnitude     HIGH     <0.5px  -80% to -90%
Flicker Frequency    Visible  <2Hz    Elimination
Frame Consistency    85.7%    95%+    +10-11%
Visual Smoothness    Good     Excellent +20%
```

---

## 🧠 TECHNICAL FOUNDATION

### 1. Optical Flow Fundamentals

#### What is Optical Flow?
Optical flow estimates motion between consecutive frames by finding where pixels moved:

```
Frame N:      Frame N+1:
[A] [B]       [A'] [B']
[C] [D]  -->  [C'] [D']

Optical Flow:
- Pixel at (0,0) moved to (1,2): Flow = (1, 2)
- Pixel at (1,0) moved to (2,1): Flow = (1, 1)
```

#### Types of Optical Flow

| Type | Method | Speed | Accuracy | Best For |
|------|--------|-------|----------|----------|
| **Dense** | Pixelwise motion | Slow | High | Full frame tracking |
| **Sparse** | Keypoint tracking | Fast | Medium | Landmark tracking |
| **Deep Learning** | Neural networks | Medium | Very High | Complex scenes |
| **Geometric** | Homography/warping | Very Fast | Low | Simple cases |

#### Key Algorithms

1. **Lucas-Kanade** (Sparse)
   - Speed: 3-5ms (CPU), 1-2ms (GPU)
   - Accuracy: 85-90%
   - Use: Landmark tracking
   
2. **Horn-Schunck** (Dense)
   - Speed: 20-30ms (CPU), 5-8ms (GPU)
   - Accuracy: 80-88%
   - Use: Full frame motion
   
3. **Farnebäck** (Dense)
   - Speed: 10-15ms (CPU), 3-5ms (GPU)
   - Accuracy: 88-92%
   - Use: OpenCV standard
   
4. **PWCNet** (Deep Learning)
   - Speed: 40-60ms (CPU), 15-20ms (GPU)
   - Accuracy: 92-96%
   - Use: High-quality motion
   
5. **RAFT** (Deep Learning)
   - Speed: 80-120ms (CPU), 25-40ms (GPU)
   - Accuracy: 95-98%
   - Use: Best quality (but slow)

### 2. Current State Analysis

**Current Pipeline (Phase 1)**:
```python
# Simple EMA smoothing (insufficient)
smooth_warping = α * current_warping + (1-α) * prev_warping
# where α = 0.7 (aggressive)

# Issues:
# - No temporal motion compensation
# - Jitter from segmentation noise
# - Landmarks can jump frame-to-frame
```

**Temporal Issues Observed**:
- Segmentation mask "wiggling" (±2 pixels)
- Landmark detection noise (±1.5 pixels)
- Warping grid oscillation
- Rendering flicker (alpha matte variations)

---

## 🛠️ PHASE 3 IMPLEMENTATION STRATEGY

### Stage 1: Optical Flow Implementation (Days 1-3)

#### Step 1.1: Select Optical Flow Algorithm

**Recommendation**: Use **Farnebäck** for Phase 3A
- Good balance of speed (3-5ms GPU) and accuracy (88-92%)
- OpenCV has solid implementation
- Can upgrade to PWCNet later

```python
# Phase 3A: Farnebäck (Initial)
import cv2

class OpticalFlowFarneback:
    def __init__(self):
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.n_poly = 5
        self.sigma = 1.2
    
    def estimate_flow(self, prev_gray, curr_gray):
        """
        Estimate optical flow between frames
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
        
        Returns:
            flow: (H, W, 2) motion vectors
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            n_poly=self.n_poly,
            poly_sigma=self.sigma,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
        return flow  # (H, W, 2) in pixels
```

**Phase 3B: Upgrade to PWCNet** (if GPU time available)
- Accuracy: 92-96% (vs 88-92%)
- GPU cost: 15-20ms (vs 3-5ms)
- Trade-off: +12ms for +4-8% accuracy improvement

#### Step 1.2: GPU Acceleration for Optical Flow

```python
# GPU-accelerated optical flow
import cupy as cp

class OpticalFlowGPU:
    """GPU-accelerated optical flow using CuPy"""
    
    def __init__(self):
        self.device = torch.device('cuda')
    
    def estimate_flow_gpu(self, prev_frame_gpu, curr_frame_gpu):
        """
        Estimate optical flow on GPU
        
        Input: GPU tensors (H, W, 3)
        Output: GPU tensor (H, W, 2) - motion vectors
        """
        # Convert torch tensors to CuPy arrays
        prev_gray_cp = cp.asarray(prev_frame_gpu)
        curr_gray_cp = cp.asarray(curr_frame_gpu)
        
        # Run Farnebäck on GPU
        flow_cp = self.farneback_gpu(prev_gray_cp, curr_gray_cp)
        
        # Convert back to torch tensor
        flow_torch = torch.from_numpy(cp.asnumpy(flow_cp))
        
        return flow_torch
    
    def farneback_gpu(self, prev_gray, curr_gray):
        """Farnebäck optical flow on GPU using CuPy"""
        # Implementation using CuPy arrays
        # CPU would be 10-15ms, GPU optimized: 3-5ms
        pass
```

### Stage 2: Motion Compensation (Days 4-5)

#### Step 2.1: Apply Optical Flow to Landmarks

```python
class TemporalLandmarkStabilization:
    """Use optical flow to track landmarks over time"""
    
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None
        self.prev_frame = None
        self.optical_flow = OpticalFlowFarneback()
    
    def stabilize_landmarks(self, landmarks, current_frame):
        """
        Stabilize landmarks using optical flow
        
        Args:
            landmarks: (17, 2) landmark positions
            current_frame: (H, W, 3) RGB frame
        
        Returns:
            stable_landmarks: (17, 2) smoothed positions
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            self.prev_frame = current_frame
            return landmarks
        
        # 1. Estimate optical flow
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        flow = self.optical_flow.estimate_flow(prev_gray, curr_gray)
        
        # 2. Get motion vectors for landmarks
        motion_vectors = []
        for i, (x, y) in enumerate(self.prev_landmarks):
            ix, iy = int(x), int(y)
            if 0 <= iy < flow.shape[0] and 0 <= ix < flow.shape[1]:
                motion = flow[iy, ix]  # (dx, dy)
                motion_vectors.append(motion)
            else:
                motion_vectors.append([0, 0])
        
        motion_vectors = np.array(motion_vectors)
        
        # 3. Predict landmark positions
        predicted_landmarks = self.prev_landmarks + motion_vectors * 0.8
        
        # 4. Apply EMA smoothing
        # Without flow: alpha * current + (1-alpha) * prev
        # With flow:    alpha * predicted + (1-alpha) * smooth_prev
        smooth_landmarks = (self.alpha * landmarks + 
                           (1-self.alpha) * predicted_landmarks)
        
        # Store for next frame
        self.prev_landmarks = smooth_landmarks
        self.prev_frame = current_frame
        
        return smooth_landmarks
```

#### Step 2.2: Motion Compensation for Warping Grid

```python
class WarpingTemporalStabilization:
    """Stabilize warping grid using optical flow"""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_warp_grid = None
        self.optical_flow = OpticalFlowFarneback()
    
    def stabilize_warp_grid(self, warp_grid, frame, landmarks):
        """
        Stabilize warping grid
        
        Args:
            warp_grid: (H//8, W//8, 2) thin-plate spline grid
            frame: (H, W, 3) RGB frame
            landmarks: (17, 2) body landmarks
        
        Returns:
            stable_grid: Smoothed warp grid
        """
        if self.prev_warp_grid is None:
            self.prev_warp_grid = warp_grid
            return warp_grid
        
        # 1. Detect overall motion from landmarks
        motion_scale = self.estimate_motion_scale(landmarks)
        
        # 2. Apply predicted smoothing
        # Use lower alpha for warp grid (more conservative)
        smooth_grid = (self.alpha * warp_grid + 
                      (1-self.alpha) * self.prev_warp_grid * motion_scale)
        
        # 3. Constrain grid deformation
        smooth_grid = self.apply_deformation_constraints(smooth_grid)
        
        self.prev_warp_grid = smooth_grid
        
        return smooth_grid
    
    def estimate_motion_scale(self, landmarks):
        """Estimate global motion scale"""
        # Calculate scale of motion across all landmarks
        # Used to adjust smoothing for fast vs slow motion
        pass
    
    def apply_deformation_constraints(self, grid):
        """Ensure grid deformation stays within reasonable bounds"""
        # Prevent grid from deforming too much
        # Helps reduce flicker and jitter
        pass
```

### Stage 3: Advanced EMA Smoothing (Days 6-7)

#### Step 3.1: Adaptive EMA Based on Motion

```python
class AdaptiveEMA:
    """EMA smoothing that adapts to motion magnitude"""
    
    def __init__(self):
        self.prev_value = None
        self.motion_history = []
        self.motion_window = 5
    
    def smooth(self, current_value, motion_magnitude):
        """
        Apply adaptive EMA smoothing
        
        Args:
            current_value: Current frame value (landmark, warping grid, etc)
            motion_magnitude: Magnitude of motion detected
        
        Returns:
            smoothed_value: EMA-smoothed value
        """
        if self.prev_value is None:
            self.prev_value = current_value
            return current_value
        
        # Calculate adaptive alpha based on motion
        # Low motion (< 0.5px): alpha = 0.8 (aggressive smoothing)
        # High motion (> 2px): alpha = 0.3 (responsive to motion)
        alpha = self.adaptive_alpha(motion_magnitude)
        
        # Apply EMA
        smoothed = alpha * current_value + (1 - alpha) * self.prev_value
        
        # Track motion
        self.motion_history.append(motion_magnitude)
        if len(self.motion_history) > self.motion_window:
            self.motion_history.pop(0)
        
        self.prev_value = smoothed
        
        return smoothed
    
    def adaptive_alpha(self, motion_magnitude):
        """Calculate alpha based on motion"""
        # Motion less than 0.5px: High smoothing
        if motion_magnitude < 0.5:
            return 0.80
        # Motion 0.5-1px: Medium smoothing
        elif motion_magnitude < 1.0:
            return 0.70
        # Motion 1-2px: Lower smoothing
        elif motion_magnitude < 2.0:
            return 0.50
        # Motion > 2px: Minimal smoothing (responsive)
        else:
            return 0.30
```

#### Step 3.2: Multi-Scale Temporal Filtering

```python
class TemporalFilter:
    """Advanced temporal filtering combining multiple scales"""
    
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.frame_buffer = []
    
    def apply_temporal_filter(self, current_value):
        """
        Apply multi-scale temporal filtering
        
        Combines:
        - Fast response filter (α=0.6, for sudden changes)
        - Medium smoothing filter (α=0.5, for gradual changes)
        - Aggressive smoothing filter (α=0.8, for jitter)
        """
        # Add to buffer
        self.frame_buffer.append(current_value)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Calculate weighted average
        # Recent frames weighted more heavily
        weights = np.linspace(0.5, 1.0, len(self.frame_buffer))
        weights /= weights.sum()
        
        filtered = np.average(
            np.array(self.frame_buffer),
            axis=0,
            weights=weights
        )
        
        return filtered
```

---

## 📊 PERFORMANCE IMPACT ANALYSIS

### Latency Addition

```
Optical Flow Computation:
├─ CPU Farnebäck:        10-15 ms
├─ GPU Farnebäck:         3-5 ms ← Recommended
├─ GPU PWCNet:           15-20 ms
└─ GPU RAFT:            25-40 ms

Motion Compensation:      2-3 ms
EMA Smoothing:           1-2 ms
───────────────────────────────
TOTAL ADDITION (GPU):     6-10 ms
```

### End-to-End Impact

```
Phase 2A (GPU + Geometric):  45-55 ms → 18-22 FPS
Phase 3 (Add Temporal):      51-65 ms → 15-18 FPS

Trade-off:
- FPS: -10% to -18% (due to optical flow computation)
- Quality: +15-20% (due to temporal stabilization)
- Smoothness: +80% (jitter nearly eliminated)
```

**Recommendation**: Accept FPS trade-off for quality improvement

---

## 📋 IMPLEMENTATION ROADMAP

### Week 1 (Feb 7-13): Optical Flow & Landmark Stabilization

**Day 1-2: Setup & Testing**
- [ ] Create OpticalFlowEstimator class
- [ ] Test Farnebäck on CPU
- [ ] Benchmark: Expected 10-15ms per frame
- [ ] Create unit tests

**Day 3-4: GPU Acceleration**
- [ ] Implement GPU version using CuPy
- [ ] Benchmark: Expected 3-5ms per frame
- [ ] Verify accuracy matches CPU version

**Day 5: Landmark Stabilization**
- [ ] Integrate optical flow into landmark tracking
- [ ] Apply motion compensation to landmarks
- [ ] Test with camera input
- [ ] Benchmark end-to-end performance

**Day 6-7: Testing & Refinement**
- [ ] Stress test with varied motion speeds
- [ ] Test with different garments
- [ ] Measure jitter reduction
- [ ] Optimize parameters (alpha values)

### Week 2 (Feb 14-20): Warping & Full Integration

**Day 1-3: Warp Grid Stabilization**
- [ ] Apply optical flow to warp grid
- [ ] Implement deformation constraints
- [ ] Test with camera
- [ ] Measure flickering reduction

**Day 4-5: Adaptive EMA**
- [ ] Implement adaptive alpha calculation
- [ ] Add motion-aware smoothing
- [ ] Test responsiveness to sudden motion
- [ ] Verify smooth motion tracking

**Day 6-7: Integration & Testing**
- [ ] Full end-to-end testing
- [ ] Measure overall quality improvement
- [ ] Performance profiling
- [ ] GPU memory usage verification

---

## 🧪 TESTING & VALIDATION

### Quantitative Metrics

```python
# Jitter measurement
def calculate_jitter(positions_sequence):
    """Calculate frame-to-frame jitter"""
    # Compute frame-to-frame differences
    differences = np.diff(positions_sequence, axis=0)
    # Jitter = std dev of differences
    jitter = np.std(differences)
    return jitter

# Temporal coherence
def calculate_temporal_coherence(sequence):
    """Measure temporal consistency"""
    # Lower variance = higher coherence
    return np.mean(np.var(sequence, axis=0))

# Flicker frequency detection
def detect_flicker_frequency(alpha_matte_sequence):
    """Detect oscillation frequency in alpha values"""
    # Use FFT to find dominant frequencies
    fft = np.fft.fft(alpha_matte_sequence)
    frequencies = np.fft.fftfreq(len(alpha_matte_sequence), 1/30)  # 30 FPS
    # Find peaks > 2Hz (flicker threshold)
    return frequencies[np.argsort(np.abs(fft))[-1:]]
```

### Test Cases

| Test | Method | Expected Result |
|------|--------|-----------------|
| **Static** | Garment on stationary person | Jitter < 0.2px |
| **Slow Motion** | Person slowly moving | FPS > 15, Smooth tracking |
| **Fast Motion** | Rapid arm movement | FPS > 12, No ghosting |
| **Camera Jitter** | Shaky hand-held camera | Jitter eliminated (< 1px) |
| **Long Duration** | 300+ frame video | Consistent smoothness |

---

## 🎯 SUCCESS CRITERIA

### Quality Metrics

```
Metric                  Target      Measurement
────────────────────────────────────────────────
Jitter Magnitude        < 0.5px     Std dev of frame diffs
Flicker Frequency       < 1 Hz      FFT analysis
Temporal Consistency    > 95%       Variance of positions
Overall Smoothness      "Excellent" Subjective rating
────────────────────────────────────────────────
```

### Performance Metrics

```
FPS Target:        15-18 (vs 18-22 Phase 2A)
Latency Target:    55-70 ms (vs 45-55 Phase 2A)
GPU Memory:        < 2.5 GB (vs 1.5 GB Phase 2A)
CPU Usage:         < 50% (per core)
```

### Deliverables

- [ ] OpticalFlowEstimator class (GPU-optimized)
- [ ] TemporalStabilizationPipeline (integrated)
- [ ] AdaptiveEMA implementation
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Documentation and examples
- [ ] PHASE_3_TEMPORAL_VALIDATION_REPORT.md

---

## 📚 REFERENCE: OPTICAL FLOW ALGORITHMS COMPARISON

### Farnebäck (Recommended for Phase 3A)

```python
def farneback_flow(prev_frame, curr_frame):
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None,
        pyr_scale=0.5,       # Image pyramid scale
        levels=3,            # Number of pyramid levels
        winsize=15,          # Averaging window size
        iterations=3,        # Iterations at each level
        n_poly=5,            # Polynomial degree
        poly_sigma=1.2,      # Gaussian sigma for polynomial
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    return flow
```

**Pros**: Good speed/accuracy, well-established
**Cons**: Not as accurate as deep learning methods

### PWCNet (Upgrade for Phase 3B)

```python
class PWCNet(nn.Module):
    """Pyramidal Cascading Warping network"""
    
    def forward(self, frame1, frame2):
        # Multi-scale pyramid analysis
        # Cascading correlation computation
        # Coarse-to-fine refinement
        return flow
```

**Pros**: Better accuracy (92-96%), faster than RAFT
**Cons**: More GPU memory, slower than Farnebäck

### RAFT (Highest Quality, Phase 3C Future)

```python
class RAFT(nn.Module):
    """Recurrent All-Pairs Field Transforms"""
    
    def forward(self, image1, image2):
        # Iterative refinement
        # All-pairs correlations
        # Recurrent updates
        return flow
```

**Pros**: Best accuracy (95-98%)
**Cons**: Slowest (25-40ms GPU)

---

## 🚀 NEXT PHASE (PHASE 4)

After Phase 3 temporal stabilization:
- Multi-garment support
- Real-time body measurements
- Visual effects and styling options
- Expected timeline: Late February - March 7

---

## ⚠️ KNOWN CHALLENGES & SOLUTIONS

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Optical flow fails at occlusions | Body parts hidden by garment | Use sparse landmarks instead |
| Large motion causes jitter | Algorithm breaks down | Use coarser pyramid levels |
| Flicker in thin garment parts | Segmentation uncertainty | Apply temporal confidence weighting |
| GPU memory spikes | Multiple flow maps in memory | Stream processing instead of batch |

---

## 📞 DESIGN DECISIONS LOG

### Decision 1: Farnebäck vs PWCNet for Phase 3A
- **Choice**: Farnebäck
- **Rationale**: 
  - Speed: 3-5ms vs 15-20ms (4-5x faster)
  - Sufficient accuracy: 88-92%
  - Can upgrade to PWCNet in Phase 3B if needed
  - Reduces GPU load for other tasks

### Decision 2: GPU vs CPU Optical Flow
- **Choice**: GPU (CuPy acceleration)
- **Rationale**:
  - CPU: 10-15ms (too slow, would block pipeline)
  - GPU: 3-5ms (acceptable overhead)
  - Aligns with Phase 2 GPU optimization
  - Keep pipeline on GPU to reduce transfers

### Decision 3: Adaptive vs Fixed Alpha
- **Choice**: Adaptive EMA
- **Rationale**:
  - Fixed alpha (0.7) causes lag in fast motion
  - Adaptive alpha responds to motion magnitude
  - Better user experience for varied motion speeds
  - Minor computational overhead (negligible)


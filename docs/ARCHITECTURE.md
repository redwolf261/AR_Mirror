# AR Sizing System Architecture

## System Data Flow

```
[Camera Input] 
    ↓
[Frame Preprocessor]
    ↓ (640x480 RGB)
[Pose Detector] (MediaPipe Pose Lite)
    ↓ (33 landmarks + confidence)
[Measurement Estimator]
    ↓ (shoulder_width, chest_width, torso_length in cm)
[Size Matcher]
    ↓ (fit_decision: TIGHT/GOOD/LOOSE)
[AR Overlay Renderer]
    ↓
[Display + Logger]
```

## Module Specifications

### 1. Frame Preprocessor
**Input:** Raw camera frame (varies)
**Output:** Normalized 640x480 RGB frame
**Operations:**
- Resize to fixed resolution
- Apply CLAHE if brightness < threshold
- Optional histogram equalization for poor lighting

### 2. Pose Detector
**Input:** 640x480 RGB frame
**Output:** 33 MediaPipe landmarks (x, y, z, visibility)
**Library:** MediaPipe Pose Lite (model size: ~3.5MB)
**Fallback:** If confidence < 0.6, request re-capture

### 3. Measurement Estimator
**Input:** Key landmarks (11, 12, 23, 24, 0)
**Output:** Body measurements in cm
**Key landmarks used:**
- 11: Left shoulder
- 12: Right shoulder
- 23: Left hip
- 24: Right hip
- 0: Nose (for distance calibration)

**Measurements extracted:**
- shoulder_width: Distance between landmarks 11-12
- chest_width: Distance between points at 40% down from shoulders
- torso_length: Vertical distance from midpoint(11,12) to midpoint(23,24)

### 4. Size Matcher
**Input:** Body measurements + garment SKU
**Output:** Fit decision (TIGHT/GOOD/LOOSE)
**Logic:**
- Load garment dimensions from lookup table
- Compare body vs garment with tolerance bands
- Apply ease factors (chest: +4cm, shoulder: +2cm)

### 5. AR Overlay Renderer
**Input:** Original frame + fit decision
**Output:** Frame with simple overlay
**Rendering:**
- Draw bounding box around torso region
- Color-code: Red (TIGHT), Green (GOOD), Yellow (LOOSE)
- Display measurement values as text

### 6. Logger
**Input:** All pipeline data
**Output:** JSON log entry
**Purpose:** Build long-term data moat

## Hardware Requirements

**Minimum:**
- Android 7.0+
- 2GB RAM
- Rear camera 8MP+
- ARCore not required

**Performance targets:**
- Frame processing: 15-20 FPS
- End-to-end latency: < 100ms per frame
- Model inference: < 50ms on CPU

## Storage Requirements

**On-device:**
- MediaPipe model: 3.5MB
- Garment database: < 500KB (JSON)
- Logs buffer: 10MB rolling

## Network Requirements

**Offline-first:**
- All inference on-device
- Log sync when WiFi available
- Garment DB updates: weekly delta sync

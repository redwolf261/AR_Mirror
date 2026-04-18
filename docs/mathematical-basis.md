# Mathematical Basis for Measurement Inference

## Problem Statement

Given 2D camera projections of body landmarks, estimate real-world body dimensions in centimeters without:
- Depth sensors
- Known distance to camera
- Calibration targets

## Core Approach: Reference-Based Scaling

### Assumption
Average human head height: 23cm (across Indian population, slight variance acceptable)

### Calibration Formula

Given MediaPipe landmarks in normalized coordinates [0, 1]:

1. **Pixel Distance Calculation**
```
pixel_distance(p1, p2) = sqrt((p1.x - p2.x)² + (p1.y - p2.y)²) × frame_width
```

2. **Head Height Estimation**
```
head_height_pixels = |nose.y - midpoint(shoulders).y| × frame_height
```

3. **Pixels-to-CM Scale Factor**
```
scale_factor = 23.0 / head_height_pixels
```

Where 23.0 is the reference head height in cm.

## Body Measurement Inference

### 1. Shoulder Width
```
shoulder_landmarks = [11, 12]  # Left, Right
shoulder_width_pixels = euclidean_distance(lm[11], lm[12]) × frame_width
shoulder_width_cm = shoulder_width_pixels × scale_factor
```

**Expected range:** 38-48cm for adults

### 2. Chest Width
Chest is approximated at 40% down the torso from shoulders:

```
left_shoulder = lm[11]
right_shoulder = lm[12]
left_hip = lm[23]
right_hip = lm[24]

left_chest_point = left_shoulder + 0.4 × (left_hip - left_shoulder)
right_chest_point = right_shoulder + 0.4 × (right_hip - right_shoulder)

chest_width_pixels = euclidean_distance(left_chest_point, right_chest_point) × frame_width
chest_width_cm = chest_width_pixels × scale_factor
```

**Expected range:** 40-55cm for adults

### 3. Torso Length
Vertical distance from shoulder midpoint to hip midpoint:

```
shoulder_mid = (lm[11] + lm[12]) / 2
hip_mid = (lm[23] + lm[24]) / 2

torso_length_pixels = |shoulder_mid.y - hip_mid.y| × frame_height
torso_length_cm = torso_length_pixels × scale_factor
```

**Expected range:** 50-70cm for adults

## Error Sources and Mitigation

### 1. Distance Variation
**Problem:** Person too close/far affects perspective

**Mitigation:**
- Check if head_height_pixels is within [80, 180] pixels for 640x480
- If outside range, display "Move closer/farther" prompt
- Reject frames outside valid range

### 2. Pose Angle
**Problem:** Shoulders not parallel to camera

**Mitigation:**
- Calculate shoulder tilt: |lm[11].y - lm[12].y|
- Reject if tilt > 0.05 (normalized coordinates)
- Display "Stand straight" prompt

### 3. Occlusion
**Problem:** Landmarks partially visible

**Mitigation:**
- Check visibility score for landmarks 11, 12, 23, 24
- Require all > 0.7
- Skip frame if below threshold

### 4. Lighting Variance
**Problem:** Poor detection in low light

**Mitigation:**
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma correction: adjusted = 255 × (pixel / 255)^(1/2.2)
- Check mean brightness, reject if < 40 or > 220

## Size Matching Logic

### Fit Decision Algorithm

For garment with dimensions (G_shoulder, G_chest, G_length):

```
ease_shoulder = 2.0  # cm
ease_chest = 4.0     # cm
ease_length = 3.0    # cm

fit_shoulder = categorize(B_shoulder, G_shoulder, ease_shoulder)
fit_chest = categorize(B_chest, G_chest, ease_chest)
fit_length = categorize(B_torso, G_length, ease_length)

def categorize(body_measure, garment_measure, ease):
    diff = garment_measure - body_measure
    
    if diff < ease:
        return TIGHT
    elif ease <= diff <= (ease + 4.0):
        return GOOD
    else:
        return LOOSE

final_fit = mode([fit_shoulder, fit_chest, fit_length])
```

### Confidence Score

```
conf_pose = min(visibility_scores for lm in [11,12,23,24])
conf_range = 1.0 if valid_distance else 0.5
conf_angle = 1.0 if valid_pose else 0.5

total_confidence = conf_pose × conf_range × conf_angle
```

Reject if total_confidence < 0.6

## Validation Strategy

### Sanity Checks
```
assert 35 < shoulder_width < 55, "Invalid shoulder measurement"
assert 35 < chest_width < 60, "Invalid chest measurement"
assert 45 < torso_length < 75, "Invalid torso measurement"
```

### Temporal Smoothing
For real-time video, apply exponential moving average:

```
smoothed[t] = alpha × raw[t] + (1 - alpha) × smoothed[t-1]
alpha = 0.3
```

Only for display, not for decision making.

## Calibration Improvement Path

Post-MVP, collect (body_measures, actual_size, fit_feedback) tuples:

```
true_shoulder = reported_size_worn × size_chart_ratio
error = predicted_shoulder - true_shoulder
```

Build lookup table of bias corrections per height/weight bracket.
No retraining required, pure deterministic adjustment.

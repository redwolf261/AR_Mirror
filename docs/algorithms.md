# Algorithmic Steps for Each Module

## Module 1: Frame Preprocessor

```
INPUT: raw_frame (H×W×3 BGR)
OUTPUT: processed_frame (640×480×3 BGR), lighting_valid (bool)

STEP 1: Resize
  IF frame.shape != (480, 640, 3):
    frame = resize(frame, (640, 480))

STEP 2: Check Lighting
  gray = rgb_to_gray(frame)
  mean_brightness = mean(gray)
  
  IF mean_brightness < 40 OR mean_brightness > 220:
    RETURN frame, False

STEP 3: Enhance if Needed
  IF mean_brightness < 80:
    frame = apply_clahe(frame, clip_limit=2.0, tile_size=(8,8))

RETURN frame, True
```

---

## Module 2: Pose Detector

```
INPUT: frame (640×480×3 BGR)
OUTPUT: landmarks (dict) OR None

STEP 1: Convert Color Space
  rgb_frame = bgr_to_rgb(frame)

STEP 2: Run MediaPipe
  results = mediapipe_pose.process(rgb_frame)
  
  IF results.pose_landmarks IS None:
    RETURN None

STEP 3: Extract Required Landmarks
  required = [0, 11, 12, 23, 24]
  landmarks = {}
  
  FOR idx IN required:
    lm = results.pose_landmarks.landmark[idx]
    
    IF lm.visibility < 0.7:
      RETURN None
    
    landmarks[idx] = {
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: lm.visibility
    }

RETURN landmarks
```

---

## Module 3: Measurement Estimator

```
INPUT: landmarks (dict), frame_size (640, 480)
OUTPUT: measurements (BodyMeasurements) OR None

STEP 1: Validate Pose Angle
  shoulder_left = landmarks[11]
  shoulder_right = landmarks[12]
  tilt = |shoulder_left.y - shoulder_right.y|
  
  IF tilt > 0.05:
    RETURN None

STEP 2: Compute Scale Factor
  nose = landmarks[0]
  shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
  head_height_pixels = |nose.y - shoulder_mid_y| * 480
  
  IF head_height_pixels < 80 OR head_height_pixels > 180:
    RETURN None
  
  scale_factor = 23.0 / head_height_pixels

STEP 3: Compute Shoulder Width
  dx = shoulder_right.x - shoulder_left.x
  dy = shoulder_right.y - shoulder_left.y
  pixel_dist = sqrt(dx² + dy²) * 640
  shoulder_width = pixel_dist * scale_factor

STEP 4: Compute Chest Width
  left_hip = landmarks[23]
  right_hip = landmarks[24]
  
  chest_left_x = shoulder_left.x + 0.4 * (left_hip.x - shoulder_left.x)
  chest_left_y = shoulder_left.y + 0.4 * (left_hip.y - shoulder_left.y)
  
  chest_right_x = shoulder_right.x + 0.4 * (right_hip.x - shoulder_right.x)
  chest_right_y = shoulder_right.y + 0.4 * (right_hip.y - shoulder_right.y)
  
  dx = chest_right_x - chest_left_x
  dy = chest_right_y - chest_left_y
  pixel_dist = sqrt(dx² + dy²) * 640
  chest_width = pixel_dist * scale_factor

STEP 5: Compute Torso Length
  hip_mid_y = (left_hip.y + right_hip.y) / 2
  pixel_dist = |shoulder_mid_y - hip_mid_y| * 480
  torso_length = pixel_dist * scale_factor

STEP 6: Validate Measurements
  IF NOT (35 < shoulder_width < 55):
    RETURN None
  IF NOT (35 < chest_width < 60):
    RETURN None
  IF NOT (45 < torso_length < 75):
    RETURN None

STEP 7: Compute Confidence
  confidences = [lm.visibility FOR lm IN landmarks.values()]
  overall_confidence = min(confidences)

RETURN BodyMeasurements(
  shoulder_width,
  chest_width,
  torso_length,
  overall_confidence,
  timestamp=now()
)
```

---

## Module 4: Size Matcher

```
INPUT: measurements (BodyMeasurements), sku (string)
OUTPUT: fit_result (FitResult) OR None

STEP 1: Load Garment Specs
  IF sku NOT IN garment_database:
    RETURN None
  
  garment = garment_database[sku]

STEP 2: Categorize Shoulder Fit
  diff = garment.shoulder_cm - measurements.shoulder_width_cm
  ease = 2.0
  tolerance = 4.0
  
  IF diff < ease:
    fit_shoulder = TIGHT
  ELSE IF ease <= diff <= (ease + tolerance):
    fit_shoulder = GOOD
  ELSE:
    fit_shoulder = LOOSE

STEP 3: Categorize Chest Fit
  diff = garment.chest_cm - measurements.chest_width_cm
  ease = 4.0
  
  IF diff < ease:
    fit_chest = TIGHT
  ELSE IF ease <= diff <= (ease + tolerance):
    fit_chest = GOOD
  ELSE:
    fit_chest = LOOSE

STEP 4: Categorize Length Fit
  diff = garment.length_cm - measurements.torso_length_cm
  ease = 3.0
  
  IF diff < ease:
    fit_length = TIGHT
  ELSE IF ease <= diff <= (ease + tolerance):
    fit_length = GOOD
  ELSE:
    fit_length = LOOSE

STEP 5: Aggregate Decision
  fits = [fit_shoulder, fit_chest, fit_length]
  count_tight = count(fits, TIGHT)
  count_good = count(fits, GOOD)
  count_loose = count(fits, LOOSE)
  
  final_decision = argmax(count_tight, count_good, count_loose)

RETURN FitResult(
  decision=final_decision,
  measurements=measurements,
  garment=garment,
  component_fits={
    shoulder: fit_shoulder,
    chest: fit_chest,
    length: fit_length
  },
  confidence=measurements.confidence
)
```

---

## Module 5: AR Overlay Renderer

```
INPUT: frame (640×480×3), landmarks (dict), fit_result (FitResult)
OUTPUT: rendered_frame (640×480×3)

STEP 1: Copy Frame
  output = frame.copy()

STEP 2: Draw Bounding Box
  left_shoulder = landmarks[11]
  right_shoulder = landmarks[12]
  left_hip = landmarks[23]
  right_hip = landmarks[24]
  
  x1 = min(left_shoulder.x, left_hip.x) * 640 - 20
  y1 = left_shoulder.y * 480 - 20
  x2 = max(right_shoulder.x, right_hip.x) * 640 + 20
  y2 = right_hip.y * 480 + 20
  
  IF fit_result.decision == TIGHT:
    color = (0, 0, 255)
  ELSE IF fit_result.decision == GOOD:
    color = (0, 255, 0)
  ELSE:
    color = (0, 255, 255)
  
  draw_rectangle(output, (x1, y1), (x2, y2), color, thickness=3)

STEP 3: Draw Decision Text
  text = fit_result.decision.value
  draw_text(output, text, (x1, y1-10), font_size=1.2, color=color)

STEP 4: Draw Measurements Overlay
  lines = [
    f"Shoulder: {measurements.shoulder_width_cm:.1f}cm ({component_fits.shoulder})",
    f"Chest: {measurements.chest_width_cm:.1f}cm ({component_fits.chest})",
    f"Torso: {measurements.torso_length_cm:.1f}cm ({component_fits.length})",
    f"Size: {garment.size_label} | Confidence: {confidence:.2f}"
  ]
  
  y = 30
  FOR line IN lines:
    draw_text(output, line, (10, y), font_size=0.5, color=(255,255,255))
    y += 25

RETURN output
```

---

## Module 6: Data Logger

```
INPUT: event_type (string), data (dict)
OUTPUT: None (side effect: write to disk)

STEP 1: Create Log Entry
  entry = {
    timestamp: time.now(),
    event_type: event_type,
    data: data
  }

STEP 2: Append to Log File
  log_file = f"logs/sizing_log_{session_id}.jsonl"
  
  WITH open(log_file, 'a') AS f:
    write_json_line(f, entry)

STEP 3: Check Buffer Size
  IF file_size(log_file) > 10MB:
    rotate_log_file()
```

---

## Full Pipeline Integration

```
INPUT: raw_frame, current_sku
OUTPUT: display_frame, fit_result OR error_message

STEP 1: Preprocess
  processed_frame, lighting_ok = preprocess(raw_frame)
  
  IF NOT lighting_ok:
    RETURN error_frame("Poor lighting"), None

STEP 2: Detect Pose
  landmarks = detect_pose(processed_frame)
  
  IF landmarks IS None:
    RETURN error_frame("Stand straight, face camera"), None

STEP 3: Estimate Measurements
  measurements = estimate_measurements(landmarks, frame_size)
  
  IF measurements IS None:
    RETURN error_frame("Move closer/farther"), None

STEP 4: Match Size
  IF current_sku IS None:
    RETURN error_frame("No garment selected"), None
  
  fit_result = match_size(measurements, current_sku)
  
  IF fit_result IS None:
    RETURN error_frame("Garment not in database"), None

STEP 5: Render
  display_frame = render_overlay(processed_frame, landmarks, fit_result)

STEP 6: Log
  log_fit_result(fit_result)

RETURN display_frame, fit_result
```

---

## Performance Optimization Points

**Parallel Processing:**
- Frame preprocessing can overlap with previous frame's inference
- MediaPipe runs on separate thread

**Early Exit:**
- If lighting check fails, skip all downstream processing
- If pose confidence < threshold, skip measurement estimation

**Caching:**
- Garment database loaded once at startup
- MediaPipe model persistent across frames

**Frame Skipping:**
- For display smoothing, render every frame
- For fit decision, require 3 consecutive consistent results

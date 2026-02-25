# Failure Modes and Fallback Logic

## Critical Failure Scenarios

### 1. Pose Detection Failure

**Triggers:**
- Landmarks not detected
- Visibility score < 0.7 for required landmarks
- MediaPipe returns None

**Fallback:**
```
IF pose_detection_failed:
    display_message("Stand straight, face camera")
    skip_frame()
    increment_failure_counter()
    
    IF failure_counter > 30:
        suggest_restart()
```

**Root causes:**
- Person not in frame
- Occlusion (hands, objects)
- Side profile instead of frontal
- Poor lighting

**Mitigation:**
- Display real-time skeleton overlay during calibration phase
- Visual indicators for correct pose
- Skip frames gracefully, continue processing

---

### 2. Invalid Distance

**Triggers:**
- Head height pixels < 80 or > 180
- Scale factor computation returns None

**Fallback:**
```
IF invalid_distance:
    compute_current_distance_hint()
    
    IF head_pixels < 80:
        display_message("Move farther from camera")
    ELSE:
        display_message("Move closer to camera")
    
    skip_frame()
```

**Visual feedback:**
- Show head height bar on screen
- Target zone indicator (green when in range)

---

### 3. Invalid Pose Angle

**Triggers:**
- Shoulder tilt > 0.05 (normalized)
- One shoulder significantly higher than other

**Fallback:**
```
IF invalid_pose_angle:
    display_message("Stand straight, level shoulders")
    draw_tilt_indicator()
    skip_frame()
```

**Visual aid:**
- Horizontal reference line
- Shoulder angle indicator

---

### 4. Poor Lighting

**Triggers:**
- Mean brightness < 40 or > 220
- High contrast variance indicating harsh shadows

**Fallback:**
```
IF poor_lighting:
    apply_clahe()
    IF still_poor_after_enhancement:
        display_message("Improve lighting or move to better lit area")
        skip_frame()
```

**Adaptive strategy:**
- Attempt CLAHE enhancement first
- If multiple consecutive failures, suggest environment change
- Log lighting conditions for offline analysis

---

### 5. Measurement Out of Range

**Triggers:**
- Shoulder width < 35cm or > 55cm
- Chest width < 35cm or > 60cm
- Torso length < 45cm or > 75cm

**Fallback:**
```
IF measurement_out_of_range:
    log_anomaly(measurements, landmarks)
    display_message("Measurement error - retry")
    skip_frame()
    
    IF repeated_anomaly:
        flag_for_manual_review()
```

**Diagnostic:**
- Log full landmark data for debugging
- Enable debug mode showing intermediate calculations
- May indicate MediaPipe model failure or edge case anatomy

---

### 6. Garment Not Found

**Triggers:**
- SKU not in database
- Database file missing/corrupted

**Fallback:**
```
IF garment_not_found:
    display_message("Product not available for sizing")
    log_missing_sku(sku)
    allow_manual_size_selection()
```

**Recovery:**
- Check for database update available
- Trigger background sync if online
- Provide fallback size chart view

---

### 7. Low Confidence Result

**Triggers:**
- Overall confidence < 0.6
- Conflicting component fits (e.g., TIGHT chest but LOOSE shoulders)

**Fallback:**
```
IF low_confidence:
    display_message("Low confidence - try again")
    show_confidence_score()
    
    IF user_requests_override:
        show_all_measurements()
        allow_manual_decision()
```

**User empowerment:**
- Show raw measurements alongside decision
- Allow user to see component-level fits
- Option to manually select size if system uncertain

---

## Graceful Degradation Strategy

### Frame Skip vs. System Halt

**Policy:**
```
consecutive_failures = 0

FOR each frame:
    result = process_frame()
    
    IF result.failed:
        consecutive_failures += 1
        
        IF consecutive_failures < 100:
            continue_with_next_frame()
        ELSE:
            display_critical_error()
            suggest_app_restart()
    ELSE:
        consecutive_failures = 0
```

### Temporal Smoothing for Display Only

```
measurements_buffer = RingBuffer(size=10)

FOR each valid measurement:
    measurements_buffer.add(measurement)
    
    display_value = exponential_moving_average(
        measurements_buffer,
        alpha=0.3
    )
```

**Critical:** Smoothing is display-only. Fit decisions use raw measurements.

---

## Network Connectivity Handling

### Offline-First Architecture

```
IF garment_db_missing:
    attempt_load_backup_db()
    IF backup_missing:
        download_minimal_db()
        IF download_failed:
            display_error("Cannot function without garment database")
            exit()

IF network_available:
    queue_log_sync()
    check_for_db_updates()
ELSE:
    buffer_logs_locally()
    continue_operation()
```

**Buffer limits:**
- Max 10MB log files before rotation
- Keep last 5 rotated logs
- Sync when WiFi detected

---

## Hardware Capability Detection

### Runtime Performance Check

```
ON_STARTUP:
    run_benchmark_frame()
    measure_inference_time()
    
    IF inference_time > 100ms:
        reduce_model_complexity()
        lower_frame_rate()
    
    IF inference_time > 300ms:
        display_warning("Device may be too slow")
        offer_reduced_mode()
```

### Reduced Mode

```
reduced_mode_changes:
    - Skip CLAHE preprocessing
    - Process every 3rd frame instead of every frame
    - Disable temporal smoothing
    - Reduce display resolution to 480x360
```

---

## User Recovery Actions

### Calibration Mode

```
IF user_requests_calibration:
    enter_calibration_mode()
    
    WHILE in_calibration:
        display_live_landmark_overlay()
        display_distance_indicator()
        display_pose_angle_indicator()
        display_current_measurements()
        
        wait_for_user_confirmation()
```

### Debug Mode

```
IF user_enables_debug:
    display_on_screen:
        - Raw landmark coordinates
        - Scale factor
        - Head height in pixels
        - Individual component fits
        - Confidence breakdown
        - Frame processing time
```

---

## Logging for Failure Analysis

### Failure Event Structure

```json
{
    "timestamp": 1704567890.123,
    "event_type": "failure",
    "data": {
        "reason": "pose_detection_failed",
        "frame_info": {
            "mean_brightness": 35.2,
            "frame_number": 4567,
            "consecutive_failures": 12
        },
        "context": {
            "current_sku": "SKU-001",
            "session_duration_sec": 45.3
        }
    }
}
```

### Aggregate Metrics for Improvement

Track:
- Failure rate by reason
- Most common failure mode
- Time-to-successful-measurement
- Lighting distribution of failures
- Distance distribution of failures

Use to:
- Adjust threshold parameters
- Improve user messaging
- Identify hardware limitations
- Guide V2 feature priority

---

## Emergency Reset

```
IF user_requests_reset:
    clear_measurement_buffer()
    reset_failure_counters()
    reload_garment_database()
    restart_mediapipe_session()
    clear_cached_frames()
```

Triggered by:
- Manual user action
- 200+ consecutive failures
- Memory pressure detected
- Unexpected exception caught

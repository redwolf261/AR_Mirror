# Data Logging Schema for Long-Term Competitive Moat

## Core Philosophy

Every interaction generates training data for deterministic improvement. No ML retraining needed—collect structured tuples that enable:
- Bias correction lookup tables
- Size chart optimization per demographic
- Return prediction models
- Vendor negotiation leverage

## Primary Log Types

### 1. Fit Measurement Log

**File:** `fit_measurements.jsonl`

```json
{
    "event_id": "uuid-v4",
    "timestamp": 1704567890.123,
    "session_id": "session-uuid",
    "user_id_hash": "sha256-hash",
    "sku": "SKU-001",
    "garment_metadata": {
        "size_label": "M",
        "brand": "BasicWear",
        "category": "t-shirt",
        "shoulder_cm": 44.0,
        "chest_cm": 50.0,
        "length_cm": 65.0
    },
    "body_measurements": {
        "shoulder_width_cm": 42.3,
        "chest_width_cm": 46.8,
        "torso_length_cm": 63.5,
        "confidence": 0.87
    },
    "fit_decision": {
        "overall": "GOOD",
        "components": {
            "shoulder": "GOOD",
            "chest": "GOOD",
            "length": "GOOD"
        }
    },
    "frame_metadata": {
        "distance_proxy": 135,
        "pose_confidence": 0.89,
        "lighting_mean": 142.3
    }
}
```

**Purpose:** Build measurement → true size mappings

---

### 2. User Feedback Log

**File:** `user_feedback.jsonl`

Captured after purchase/return:

```json
{
    "event_id": "uuid-v4",
    "timestamp": 1704654290.456,
    "session_id": "session-uuid",
    "user_id_hash": "sha256-hash",
    "sku": "SKU-001",
    "predicted_fit": "GOOD",
    "actual_outcome": {
        "purchased": true,
        "size_selected": "M",
        "returned": false,
        "return_reason": null,
        "fit_rating": 4,
        "fit_comment": "slightly_loose"
    },
    "time_since_measurement_sec": 86400
}
```

**Purpose:** Ground truth for system accuracy, return prediction

---

### 3. Failure Event Log

**File:** `failures.jsonl`

```json
{
    "event_id": "uuid-v4",
    "timestamp": 1704567900.789,
    "session_id": "session-uuid",
    "failure_type": "pose_detection_failed",
    "context": {
        "consecutive_failures": 8,
        "lighting_mean": 28.4,
        "frame_number": 234,
        "sku_active": "SKU-003"
    },
    "device_info": {
        "model": "Redmi_9A",
        "android_version": "10",
        "ram_mb": 2048,
        "camera_mp": 13
    }
}
```

**Purpose:** Identify device-specific issues, optimize thresholds

---

### 4. Session Analytics Log

**File:** `sessions.jsonl`

```json
{
    "session_id": "session-uuid",
    "user_id_hash": "sha256-hash",
    "start_timestamp": 1704567800.0,
    "end_timestamp": 1704567950.0,
    "duration_sec": 150,
    "skus_tried": ["SKU-001", "SKU-002", "SKU-003"],
    "successful_measurements": 3,
    "failed_measurements": 8,
    "failure_breakdown": {
        "pose_detection_failed": 5,
        "poor_lighting": 2,
        "invalid_distance": 1
    },
    "device_info": {
        "model": "Redmi_9A",
        "android_version": "10"
    }
}
```

**Purpose:** UX optimization, device performance tracking

---

### 5. Size Chart Optimization Log

**File:** `size_chart_adjustments.jsonl`

Periodic aggregation output:

```json
{
    "timestamp": 1704672000.0,
    "sku": "SKU-001",
    "size_label": "M",
    "original_specs": {
        "shoulder_cm": 44.0,
        "chest_cm": 50.0,
        "length_cm": 65.0
    },
    "measured_wearers": 47,
    "outcome_distribution": {
        "returned_too_tight": 3,
        "returned_too_loose": 2,
        "kept_good_fit": 38,
        "kept_acceptable": 4
    },
    "body_measurement_stats": {
        "shoulder_mean": 41.2,
        "shoulder_std": 2.1,
        "chest_mean": 45.8,
        "chest_std": 2.9,
        "torso_mean": 62.3,
        "torso_std": 3.4
    },
    "recommendation": {
        "verdict": "size_chart_accurate",
        "suggested_adjustment": null
    }
}
```

**Purpose:** Evidence for vendor renegotiation, QA alerts

---

## Aggregated Analytical Datasets

### Dataset 1: Measurement Bias Correction Table

Generated weekly from fit + feedback logs:

```json
{
    "demographic_segment": {
        "height_bracket": "160-170cm",
        "weight_bracket": "60-70kg"
    },
    "measurement_biases": {
        "shoulder_bias_cm": -1.2,
        "chest_bias_cm": 0.8,
        "torso_bias_cm": -0.5
    },
    "sample_size": 234,
    "confidence_interval_95": {
        "shoulder": [-1.5, -0.9],
        "chest": [0.5, 1.1],
        "torso": [-0.8, -0.2]
    }
}
```

**Usage in code:**
```python
corrected_shoulder = raw_shoulder + lookup_bias(height, weight, 'shoulder')
```

No model retraining. Pure lookup table.

---

### Dataset 2: Return Prediction Features

```json
{
    "sku": "SKU-001",
    "size_label": "M",
    "return_predictors": {
        "chest_margin_cm": -1.5,
        "shoulder_margin_cm": 0.5,
        "overall_fit": "TIGHT",
        "confidence": 0.72,
        "lighting_mean": 95.3
    },
    "return_probability": 0.34,
    "actual_returned": true,
    "features_for_model": [
        -1.5, 0.5, 1, 0.72, 95.3
    ]
}
```

Train simple logistic regression offline:
```
return ~ chest_margin + shoulder_margin + confidence + fit_encoded
```

Deploy as lightweight scoring function. No online ML.

---

### Dataset 3: Device Performance Profile

```json
{
    "device_model": "Redmi_9A",
    "total_sessions": 1523,
    "avg_measurement_time_ms": 87.3,
    "failure_rate": 0.23,
    "common_failures": [
        {"type": "poor_lighting", "rate": 0.15},
        {"type": "pose_detection_failed", "rate": 0.08}
    ],
    "recommended_settings": {
        "enable_clahe": true,
        "frame_skip": 2,
        "reduce_resolution": false
    }
}
```

Auto-tune app on device detection.

---

## Privacy and Compliance

### PII Handling

**User ID:** SHA-256 hash of device ID + salt
```python
user_id_hash = hashlib.sha256(f"{device_id}{SALT}".encode()).hexdigest()
```

**No storage:**
- Raw camera frames
- Face landmarks
- Biometric identifiers

**Stored only:**
- Body measurements (cm values)
- Fit decisions
- Purchase/return outcomes

### Data Retention

```
fit_measurements.jsonl: 2 years
user_feedback.jsonl: 2 years
failures.jsonl: 90 days
sessions.jsonl: 90 days
```

### Anonymization for ML

Before any model training:
```python
strip_fields = ['user_id_hash', 'device_info', 'session_id']
anonymized_data = raw_data.drop(columns=strip_fields)
```

---

## Offline Analysis Workflows

### Weekly: Bias Correction Update

```python
feedback_data = load_logs('user_feedback.jsonl', last_7_days)
measurement_data = load_logs('fit_measurements.jsonl', last_7_days)

merged = join(feedback_data, measurement_data, on='session_id')

for segment in demographic_segments:
    subset = filter(merged, segment)
    bias = compute_bias(subset)
    update_lookup_table(segment, bias)
```

Deploy updated lookup table via app update.

### Monthly: Size Chart Analysis

```python
for sku in catalog:
    outcomes = get_outcomes(sku, last_30_days)
    return_rate = outcomes.returned.mean()
    
    if return_rate > 0.15:
        flag_for_vendor_review(sku)
        
    if return_rate > 0.25:
        alert_operations(sku, "high_return_risk")
```

### Quarterly: Return Prediction Model

```python
features = build_feature_matrix(fit_measurements, user_feedback)
X = features[['chest_margin', 'shoulder_margin', 'confidence', 'fit_encoded']]
y = features['returned']

model = LogisticRegression()
model.fit(X, y)

export_model_as_lookup_table(model, 'return_score.json')
```

Deploy as JSON scoring function—no runtime ML lib needed.

---

## Moat Accumulation Strategy

### Months 1-3: Data Collection
- Focus on volume: every interaction logged
- Build baseline measurement distributions
- Identify common failure modes

### Months 4-6: First Bias Corrections
- Compute demographic-specific biases
- Deploy lookup tables
- Measure accuracy improvement

### Months 7-12: Return Prediction
- Train return models per category
- Integrate return warnings in UI
- Validate against actual returns

### Year 2+: Vendor Leverage
- Present size chart inconsistency evidence
- Negotiate standardization
- Build brand-specific correction profiles

**End state:** Proprietary measurement-to-fit mapping that competitors cannot replicate without equivalent data.

---

## Implementation Checklist

```python
class DataLogger:
    def log_fit_measurement(self, data: FitMeasurementData):
        pass
    
    def log_user_feedback(self, data: UserFeedbackData):
        pass
    
    def log_failure(self, failure_type: str, context: dict):
        pass
    
    def log_session_end(self, session: SessionData):
        pass
    
    def flush_to_storage(self):
        pass
    
    def sync_to_server(self):
        pass
```

All logging async, non-blocking, with in-memory buffer.

# Failure Taxonomy for AR Mirror Virtual Try-On

## Purpose

This document defines **retailer-relevant failure modes** for the AR Mirror system. This is a **retail revenue protection system**, not an academic exercise.

**Core Principle**: A failure mode only matters if it:
- Breaks user trust **OR**
- Causes incorrect purchase decisions **OR**
- Prevents usage in common real-world settings

---

## Failure Mode Classification

Each failure mode is evaluated on:

1. **Definition**: One-sentence description
2. **Trigger Condition**: What causes it
3. **Detection Strategy**: How to measure it (even if imperfect)
4. **Frequency**: Low / Medium / High
5. **User Trust Impact**: Low / Medium / High / Catastrophic
6. **Business Impact**: Revenue/retention implications
7. **Acceptable?**: Yes / No / Temporarily
8. **Module Ownership**: Which module is responsible

---

## Priority 0: Trust-Breaking Failures

### 1. Face/Neck Contamination

**Definition**: Any garment pixels visible above chin line.

**Trigger Condition**:
- Semantic parsing misclassifies face as upper-clothes
- Geometric constraint fails (no pose landmarks)
- Collar extends into face region

**Detection Strategy**:
```python
face_contamination = cv2.bitwise_and(garment_mask, face_mask)
violation_rate = (face_contamination.sum() / face_mask.sum()) > 0.01
```

**Frequency**: Low (after Option 1 geometric constraints)

**User Trust Impact**: **CATASTROPHIC**

**Business Impact**:
- Immediate app abandonment
- Zero conversion probability
- Negative social media sharing

**Acceptable?**: **NO** - This is a P0 invariant

**Module Ownership**: 
- Primary: **Occlusion Policy** (must enforce face dominance)
- Secondary: **Geometry** (torso constraint prevents this)

**Current Mitigation**:
- ✅ Geometric torso constraint (eliminates ~60% of cases)
- ✅ Face dominance rule (LAYER_DOMINANCE.md Rule 1)
- ✅ Runtime validation (OcclusionLayer.validate_ordering())

**Measurement Target**: 0.0% violation rate (zero tolerance)

---

## Priority 1: High-Frequency Trust Breakers

### 2. Hairline/Collar Conflict

**Definition**: Collar visually cuts through hair mass unnaturally.

**Trigger Condition**:
- Long hair extends below collar line
- Parsing resolution too coarse (119×119) to capture hair boundary
- Temporal jitter causes hair-garment boundary flicker

**Detection Strategy**:
```python
hair_garment_boundary = cv2.bitwise_and(
    cv2.dilate(hair_mask, kernel),
    garment_mask
)
boundary_jitter = IoU_delta(hair_garment_t, hair_garment_t-1)
violation = boundary_jitter > 0.15
```

**Frequency**: **High** (long hair is common, especially in fashion retail)

**User Trust Impact**: **High**

**Business Impact**:
- "Looks fake" perception
- Reduced time-on-app
- Lower conversion for high-collar garments

**Acceptable?**: **Temporarily** (until hair-length classification added)

**Module Ownership**:
- Primary: **Parsing** (coarse resolution is root cause)
- Secondary: **Policy** (hair dominance rule helps but doesn't solve)

**Current Mitigation**:
- ✅ Hair dominance rule (LAYER_DOMINANCE.md Rule 2)
- ✅ Temporal smoothing (reduces jitter)
- ⚠️ Resolution mismatch remains (119×119 → 640×480)

**Measurement Target**: 
- Boundary jitter < 0.1 (IoU delta)
- Violation rate < 15%

**Future Fix**: Hair-length classification → short hair can be dominated by high collars

---

### 3. Side-Profile Torso Clipping

**Definition**: Garment shrinks or clips on rotated torso (>30° yaw).

**Trigger Condition**:
- User rotates torso in mirror
- Axis-aligned rectangle clips rotated shoulders
- Shoulder landmarks no longer horizontally aligned

**Detection Strategy**:
```python
shoulder_rotation = abs(left_shoulder.y - right_shoulder.y) / frame_height
garment_area_delta = (area_t - area_t-1) / area_t-1
violation = (shoulder_rotation > 0.1) and (garment_area_delta < -0.3)
```

**Frequency**: **Medium** (mirror behavior - users turn sideways)

**User Trust Impact**: **High**

**Business Impact**:
- "Looks wrong from the side" → no purchase
- Especially critical for fitted garments
- Reduces confidence in size accuracy

**Acceptable?**: **Temporarily** (documented in KNOWN_GEOMETRIC_LIMITATIONS.md)

**Module Ownership**:
- Primary: **Geometry** (axis-aligned rectangle limitation)
- Secondary: **Parsing** (could provide better torso boundary)

**Current Mitigation**:
- ⚠️ Documented as known limitation
- ⚠️ Safety margins (10% horizontal) absorb small rotations

**Measurement Target**:
- Garment area drop < 30% for rotations < 45°
- Violation rate < 20%

**Future Fix**: Replace rectangle with convex polygon or trapezoid aligned to shoulder slope

---

## Priority 2: Medium-Impact Quality Issues

### 4. Loose Garment Suppression

**Definition**: Oversized shirts/dresses rendered too tight due to torso constraint.

**Trigger Condition**:
- Garment fabric extends beyond geometric torso boundary
- Geometric constraint clips loose fabric
- Semantic parsing says "upper-clothes" but geometry says "outside torso"

**Detection Strategy**:
```python
semantic_garment = parsing_masks['upper_body']
geometric_garment = cv2.bitwise_and(semantic_garment, torso_geometry)
suppression_ratio = 1 - (geometric_garment.sum() / semantic_garment.sum())
violation = suppression_ratio > 0.5
```

**Frequency**: **Medium** (fashion trend dependent - oversized is popular)

**User Trust Impact**: **Medium**

**Business Impact**:
- Size mistrust → returns
- "Looks tighter than it should"
- Affects oversized/loose-fit category sales

**Acceptable?**: **Yes** (documented trade-off in KNOWN_GEOMETRIC_LIMITATIONS.md)

**Module Ownership**:
- Primary: **Geometry** (torso constraint is intentionally conservative)
- Secondary: **Policy** (could add "loose garment" mode in future)

**Current Mitigation**:
- ✅ Documented as acceptable trade-off
- ✅ Prevents worse failure (garment on arms/neck)

**Measurement Target**:
- Suppression ratio < 40% (acceptable)
- Violation rate < 25%

**Design Decision**: Prefer under-rendering (safe) over over-rendering (breaks trust)

---

### 5. Arms Raised Shrinkage

**Definition**: Garment shrinks when arms lifted above shoulders.

**Trigger Condition**:
- User raises arms (trying on, adjusting hair)
- Shoulder landmarks move upward
- Torso rectangle shrinks vertically

**Detection Strategy**:
```python
shoulder_height = min(left_shoulder.y, right_shoulder.y)
baseline_shoulder_height = 0.3  # Typical for arms-down
arms_raised = shoulder_height < baseline_shoulder_height - 0.1
garment_shrinkage = (area_t - area_baseline) / area_baseline < -0.4
violation = arms_raised and garment_shrinkage
```

**Frequency**: **Medium**

**User Trust Impact**: **Medium**

**Business Impact**:
- UX polish issue, not core trust
- Affects dynamic try-on experience
- Low impact on purchase decision

**Acceptable?**: **Temporarily** (documented in KNOWN_GEOMETRIC_LIMITATIONS.md)

**Module Ownership**:
- Primary: **Geometry** (shoulder position used for torso top boundary)
- Secondary: None

**Current Mitigation**:
- ⚠️ Documented as known limitation

**Measurement Target**:
- Garment shrinkage < 40% when arms raised
- Violation rate < 30%

**Future Fix**: Use shoulder-to-hip distance instead of absolute shoulder position

---

### 6. Foreground Object Occlusion Failure

**Definition**: Garment rendered over bags, hands, phones held in front of torso.

**Trigger Condition**:
- Object in front of torso (no depth information)
- Garment rendered on object surface
- No object detection or depth estimation

**Detection Strategy**:
```python
# Hard to detect without depth/object signals
# Proxy: Sudden garment mask appearance in unexpected regions
garment_delta = cv2.absdiff(garment_mask_t, garment_mask_t-1)
unexpected_appearance = (garment_delta > 128).sum() / frame_area > 0.05
# This is a weak proxy - needs depth or object detection
```

**Frequency**: **Medium**

**User Trust Impact**: **Medium**

**Business Impact**:
- Moderate annoyance
- "Looks glitchy" when holding objects
- Doesn't prevent core try-on usage

**Acceptable?**: **Temporarily** (requires depth/object detection - future work)

**Module Ownership**:
- Primary: **Geometry** (needs depth information)
- Secondary: **Parsing** (could detect hands/objects)

**Current Mitigation**:
- ❌ None (no depth information available)

**Measurement Target**:
- Violation rate < 20% (estimated)

**Future Fix**: Add depth estimation or object segmentation

---

## Priority 3: Low-Impact Edge Cases

### 7. Multi-Person Misbinding

**Definition**: Garment attaches to wrong person in multi-person frame.

**Trigger Condition**:
- Multiple people in frame
- Pose detector picks first person (not target user)
- Garment constrained to wrong person's torso

**Detection Strategy**:
```python
num_people = len(pose_detector.detect_all_people(frame))
person_id_mismatch = (target_person_id != selected_person_id)
violation = (num_people > 1) and person_id_mismatch
```

**Frequency**: **Low** (AR mirror setup typically single-person)

**User Trust Impact**: **High** (when it happens)

**Business Impact**:
- Low overall (rare in mirror setup)
- High impact in specific cases (shopping with friends)

**Acceptable?**: **Temporarily** (documented in KNOWN_GEOMETRIC_LIMITATIONS.md)

**Module Ownership**:
- Primary: **Geometry** (single-person assumption)
- Secondary: None

**Current Mitigation**:
- ⚠️ Documented as known limitation
- ⚠️ Mirror setup reduces frequency

**Measurement Target**:
- Violation rate < 5% (rare)

**Future Fix**: Add person selection logic or multi-person support

---

## Measurement Framework

### What We Will Measure

For each failure mode, track:

1. **Violation Rate**: % of frames where failure occurs
2. **Severity**: Impact magnitude when it occurs
3. **Duration**: How long failure persists (frames)
4. **Recovery**: Can system self-correct?

### Core Metrics (Cross-Cutting)

| Metric | Definition | Target | Priority |
|--------|------------|--------|----------|
| Occlusion Violation Rate | % frames with garment-face overlap | 0.0% | P0 |
| Boundary Jitter | IoU delta frame-to-frame | < 0.1 | P1 |
| Garment Area Stability | % area change frame-to-frame | < 20% | P1 |
| Side-Profile Garment Loss | % area drop at 45° rotation | < 30% | P1 |
| Temporal Consistency | Mask correlation t to t-1 | > 0.85 | P2 |

### Detection Implementation

```python
class FailureTaxonomyBenchmark:
    def measure_face_contamination(self, garment, face) -> float:
        """P0: Face/neck contamination"""
        overlap = cv2.bitwise_and(garment, face)
        return overlap.sum() / max(face.sum(), 1)
    
    def measure_boundary_jitter(self, mask_t, mask_t_prev) -> float:
        """P1: Hairline/collar conflict"""
        intersection = cv2.bitwise_and(mask_t, mask_t_prev).sum()
        union = cv2.bitwise_or(mask_t, mask_t_prev).sum()
        iou = intersection / max(union, 1)
        return 1 - iou  # Jitter = 1 - IoU
    
    def measure_garment_stability(self, area_t, area_t_prev) -> float:
        """P1: Side-profile clipping, arms raised"""
        delta = abs(area_t - area_t_prev) / max(area_t_prev, 1)
        return delta
    
    def measure_suppression_ratio(self, semantic, geometric) -> float:
        """P2: Loose garment suppression"""
        return 1 - (geometric.sum() / max(semantic.sum(), 1))
```

---

## Module Ownership Matrix

| Failure Mode | Primary Module | Secondary Module | Fix Location |
|--------------|----------------|------------------|--------------|
| Face/Neck Contamination | Occlusion Policy | Geometry | Policy enforcement |
| Hairline/Collar Conflict | Parsing | Policy | Resolution upgrade |
| Side-Profile Clipping | Geometry | Parsing | Polygon torso |
| Loose Garment Suppression | Geometry | Policy | Acceptable trade-off |
| Arms Raised Shrinkage | Geometry | - | Distance-based torso |
| Foreground Object Failure | Geometry | Parsing | Depth estimation |
| Multi-Person Misbinding | Geometry | - | Person selection |

**Key Insight**: Most failures are **Geometry** or **Parsing** issues, not Policy or Compositor.

---

## Failure Mode Prioritization

### P0: Zero Tolerance
- Face/Neck Contamination

### P1: High Business Impact
- Hairline/Collar Conflict
- Side-Profile Torso Clipping

### P2: Quality Polish
- Loose Garment Suppression
- Arms Raised Shrinkage
- Foreground Object Failure

### P3: Edge Cases
- Multi-Person Misbinding

---

## Testing Requirements

### Minimum Test Coverage

For each P0/P1 failure mode:
- [ ] Golden artifact showing failure case
- [ ] Detection metric implementation
- [ ] Baseline measurement (current violation rate)
- [ ] Target threshold defined

### Regression Prevention

- [ ] Add failure mode tests to CI/CD
- [ ] Alert on violation rate increase > 5%
- [ ] Track metrics over time (trend analysis)

---

## What We Will NOT Measure

Explicitly excluded (academic, not retail-relevant):

- ❌ Pixel-perfect mask accuracy
- ❌ Semantic segmentation mIoU
- ❌ Single-frame glitches (< 3 frames)
- ❌ Rare pose edge cases (< 1% frequency)
- ❌ Perfect boundary smoothness

**Rationale**: These don't affect purchase decisions or user trust.

---

## Next Steps (After Option 2)

1. **Implement detection metrics** in `benchmarks/failure_taxonomy_benchmark.py`
2. **Measure baseline** violation rates with current system
3. **Set thresholds** for acceptable failure rates
4. **Only then**: Design fixes (Option 3 module architecture)

---

## Design Review Checkpoint

Before proceeding to Option 3, validate:

- [ ] Each failure mode maps to exactly one primary module
- [ ] Detection strategy is implementable (even if imperfect)
- [ ] Business impact is clearly stated
- [ ] Acceptable trade-offs are explicitly documented
- [ ] No academic edge cases included

---

## Summary

| Failure Mode | Frequency | Trust Impact | Acceptable? | Primary Module |
|--------------|-----------|--------------|-------------|----------------|
| Face/Neck Contamination | Low | Catastrophic | **NO** | Policy |
| Hairline/Collar Conflict | High | High | Temporarily | Parsing |
| Side-Profile Clipping | Medium | High | Temporarily | Geometry |
| Loose Garment Suppression | Medium | Medium | **Yes** | Geometry |
| Arms Raised Shrinkage | Medium | Medium | Temporarily | Geometry |
| Foreground Object Failure | Medium | Medium | Temporarily | Geometry |
| Multi-Person Misbinding | Low | High | Temporarily | Geometry |

**Total**: 7 failure modes (6-8 target range)

**Key Insight**: Geometry module is responsible for 5/7 failures → highest priority for future work.

**Business Principle**: Zero tolerance for face contamination, temporary acceptance of geometric limitations with clear documentation.

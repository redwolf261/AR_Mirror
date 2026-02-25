# Known Geometric Limitations

## Purpose

This document explicitly acknowledges the **limitations and edge cases** of the geometric torso constraint system. These are **known trade-offs**, not bugs.

The geometric constraint uses pose landmarks to define a torso boundary, then intersects semantic parsing masks with this geometry. This is a **design decision** with specific failure modes.

---

## Core Design Decision

### "Geometry Wins for Safety"

**Statement**: When geometric constraints (pose-derived torso) conflict with semantic parsing (LIP upper-clothes), **geometry wins**.

**Rationale**: We prefer **visual plausibility** over **semantic completeness**.

**Implication**:
- We may **under-render** garments (safe - garment too small)
- We will never **over-render** garments (breaks trust - garment on face/neck)

**Trade-off**: Acceptable loss of garment coverage in exchange for zero face/hair occlusion.

---

## Limitation 1: Axis-Aligned Rectangle Torso

### Current Implementation

The torso geometry is calculated as an **axis-aligned rectangle**:

```python
# Extract landmarks
left_shoulder = pose_landmarks[11]
right_shoulder = pose_landmarks[12]
left_hip = pose_landmarks[23]
right_hip = pose_landmarks[24]

# Create rectangle
top_y = min(left_shoulder.y, right_shoulder.y)
bottom_y = max(left_hip.y, right_hip.y)
left_x = left_shoulder.x
right_x = right_shoulder.x

torso_mask = cv2.rectangle(...)
```

### Failure Modes

#### 1.1 Side Profile / Rotated Torso

**Problem**: When user rotates torso, shoulders are no longer horizontally aligned.

**Current Behavior**: Rectangle clips visible shoulder region.

**Example**:
```
Front view:     Side view (45°):
  |-----|         |---|  ← Rectangle clips rotated shoulder
  |     |         |   |
  |     |         |   |
  |-----|         |---|
```

**Impact**: Garment region under-rendered on rotated shoulder.

**Mitigation (Future)**: Replace rectangle with convex polygon or trapezoid aligned to shoulder slope.

---

#### 1.2 Forward/Backward Lean

**Problem**: When user leans forward, torso extends beyond hip landmarks.

**Current Behavior**: Rectangle bottom boundary is too high, clips lower torso.

**Example**:
```
Upright:        Forward lean:
  |-----|         |-----|
  |     |         |     |
  |     |         |     | ← Torso extends below
  |-----|         |-----|    rectangle boundary
                    ↓↓↓
```

**Impact**: Garment region under-rendered on lower torso during lean.

**Mitigation (Future)**: Add vertical margin based on torso angle, or use spine landmarks.

---

#### 1.3 Cropped Frames / Partial Body

**Problem**: When hips are out of frame, hip landmarks are unavailable.

**Current Behavior**: Torso geometry calculation fails, constraint is skipped.

**Example**:
```
Full frame:     Cropped (chest-up):
  |-----|         |-----|
  |     |         |     |
  |     |         -------  ← No hip landmarks
  |-----|         (skipped)
```

**Impact**: Falls back to unconstrained semantic parsing (acceptable degradation).

**Mitigation**: Already handled via try-except in `_apply_geometric_constraints()`.

---

## Limitation 2: Landmark Drift vs Parsing Disagreement

### The Two-Authority Problem

The system now has **two sources of truth**:

1. **Semantic parsing** (ONNX LIP model): "This is upper-clothes"
2. **Geometric constraint** (MediaPipe pose): "This is torso region"

When they disagree, **geometry wins**.

### Failure Modes

#### 2.1 Pose Estimation Error

**Problem**: MediaPipe pose mislocates shoulder/hip landmarks.

**Current Behavior**: Garment constrained to incorrect torso region.

**Example**:
- Pose says: "Shoulders at y=100"
- Reality: Shoulders at y=120
- Result: Garment region too high (clips actual torso)

**Impact**: Garment under-rendered due to pose error, not semantic error.

**Mitigation**: Pose errors are typically small (<5% of frame). Margins (10% horizontal, 5% vertical) absorb most errors.

---

#### 2.2 Loose Clothing Edge Case

**Problem**: Loose clothing (oversized shirts, dresses) extends beyond torso geometry.

**Current Behavior**: Garment constrained to torso, clips loose fabric.

**Example**:
```
Tight shirt:    Loose dress:
  |-----|         |-------|  ← Fabric extends beyond
  | fit |         |  clip |     torso boundary
  |-----|         |-------|
```

**Impact**: Loose garments appear tighter than they should.

**Trade-off**: Acceptable - prevents garments from floating on arms/neck.

---

#### 2.3 Arms Raised / Non-Standard Poses

**Problem**: When arms are raised, shoulder landmarks move upward.

**Current Behavior**: Torso rectangle shrinks vertically.

**Example**:
```
Arms down:      Arms raised:
  |-----|         |---|    ← Shoulders higher
  |     |         |   |       torso smaller
  |     |         |   |
  |-----|         |---|
```

**Impact**: Garment region under-rendered when arms raised.

**Mitigation (Future)**: Use shoulder-to-hip distance instead of absolute positions.

---

## Limitation 3: No Depth Information

### Current Implementation

The torso geometry is **2D only** (image plane).

### Failure Modes

#### 3.1 Depth Ambiguity

**Problem**: Cannot distinguish between:
- User close to camera (large torso)
- User far from camera (small torso)

**Current Behavior**: Torso size varies with distance.

**Impact**: Garment region scales with user distance (acceptable for AR mirror).

**Mitigation**: Not needed for fixed-distance AR mirror setup.

---

#### 3.2 Occlusion by Objects

**Problem**: Objects in front of torso (hands, bags) are not detected.

**Current Behavior**: Garment rendered on occluding objects.

**Example**:
- User holds bag in front of chest
- Garment rendered on bag surface

**Impact**: Visual artifact when objects occlude torso.

**Mitigation (Future)**: Add depth estimation or object detection.

---

## Limitation 4: Single-Person Assumption

### Current Implementation

The system assumes **one person** in frame.

### Failure Modes

#### 4.1 Multi-Person Frames

**Problem**: Multiple people in frame, pose detector picks first person.

**Current Behavior**: Garment constrained to first person's torso.

**Impact**: If target user is not first person, constraint fails.

**Mitigation (Future)**: Add person selection logic or multi-person support.

---

## Testing Requirements

Any change to geometric constraints MUST test these edge cases:

1. **Side profile** (45° rotation)
2. **Forward lean** (30° torso angle)
3. **Cropped frame** (hips out of view)
4. **Arms raised** (shoulders above normal)
5. **Loose clothing** (fabric beyond torso)

---

## Acceptable Degradation Modes

The system is designed to **fail safely**:

| Failure Mode | Behavior | Acceptable? |
|--------------|----------|-------------|
| Pose landmarks unavailable | Skip constraint, use semantic only | ✅ Yes |
| Torso too small | Under-render garment | ✅ Yes (safe) |
| Torso too large | Over-render garment | ❌ No (breaks trust) |
| Garment clips loose fabric | Tighter appearance | ✅ Yes (trade-off) |
| Multi-person frame | Wrong person selected | ⚠️ Acceptable (rare) |

**Key Principle**: Always prefer **under-rendering** (safe) over **over-rendering** (breaks trust).

---

## Future Improvements (Not Implemented)

### Phase 2: Better Geometry

- Replace rectangle with **convex polygon** (shoulders + hips)
- Add **trapezoid** aligned to shoulder slope
- Use **spine landmarks** for lean detection

### Phase 3: Depth Awareness

- Add **depth estimation** for occlusion handling
- Detect **objects in front of torso**
- Support **variable user distance**

### Phase 4: Multi-Person Support

- Add **person selection** logic
- Support **multiple garments** on multiple people
- Handle **person tracking** across frames

---

## Modification Protocol

To change geometric constraint logic:

1. **Test all edge cases** listed in this document
2. **Verify degradation modes** remain acceptable
3. **Update this document** with new limitations
4. **Add regression tests** for new edge cases

**Do not modify geometric constraints without updating this document.**

---

## Summary

| Limitation | Impact | Mitigation | Priority |
|------------|--------|------------|----------|
| Axis-aligned rectangle | Clips rotated torso | Use convex polygon | Medium |
| Forward/backward lean | Clips lower torso | Add vertical margin | Low |
| Cropped frames | Skips constraint | Graceful degradation | ✅ Done |
| Pose estimation error | Wrong torso region | Margins absorb errors | ✅ Done |
| Loose clothing | Clips fabric | Acceptable trade-off | ✅ Accepted |
| Arms raised | Shrinks torso | Use shoulder-hip distance | Low |
| No depth info | Distance scaling | Not needed for AR mirror | ✅ Accepted |
| Multi-person | Wrong person | Add person selection | Medium |

**Current Status**: All critical limitations documented and mitigated or accepted.

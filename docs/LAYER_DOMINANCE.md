# Layer Dominance Rules

## Purpose

This document defines the **semantic ownership rules** for occlusion handling in the AR Mirror virtual try-on system. These rules are **design contracts** that must not change without explicit review.

The `OcclusionLayer` enum defines **Z-order** (rendering sequence), but this document defines **dominance** (which layer wins when signals conflict).

---

## Core Dominance Rules

### Rule 1: Face Always Dominates Garment

**Statement**: Face regions ALWAYS occlude garments, regardless of semantic parsing output.

**Rationale**: User trust breaks immediately if garments cover the face. This is non-negotiable for virtual try-on.

**Implementation**: 
```python
# In create_occlusion_aware_composite()
occlusion_mask = cv2.bitwise_or(hair, face)
garment_alpha = garment_mask * (1 - occlusion_mask)
```

**Edge Cases**:
- High collars near chin: Face wins
- Scarves near mouth: Face wins
- Parsing misclassifies face as torso: Face still wins (geometry protects)

---

### Rule 2: Hair Always Dominates Garment

**Statement**: Hair regions ALWAYS occlude garments, even if parsing says "upper-clothes".

**Rationale**: Hair covering garments is natural. Garments covering hair looks artificial.

**Exception (Reserved for Future)**:
- Short hair above collar line may be dominated by high-collar garments
- NOT IMPLEMENTED YET - requires hair length classification

**Implementation**:
```python
# Same as Rule 1 - hair included in occlusion mask
occlusion_mask = cv2.bitwise_or(hair, face)
```

---

### Rule 3: Geometry Can Suppress Semantics, Never Invert

**Statement**: When geometric constraints (pose-derived torso) conflict with semantic parsing (LIP upper-clothes), **geometry wins for safety**.

**Rationale**: We prefer **visual plausibility** over **semantic completeness**.

**Design Decision**:
> "When semantic and geometric signals disagree, geometry wins for safety."

**Example**:
- Parsing says "upper-clothes" extends to arms
- Pose says "torso is only shoulders-to-hips"
- Result: Garment constrained to torso (geometry wins)

**Implication**: 
- We may **under-render** garments (safe)
- We will never **over-render** garments (breaks trust)

**Implementation**:
```python
# In _apply_geometric_constraints()
masks['upper_body'] = cv2.bitwise_and(
    masks['upper_body'],  # Semantic signal
    torso_geometry        # Geometric signal (wins)
)
```

---

### Rule 4: Garment Dominates Torso Skin Only

**Statement**: Garments only occlude exposed skin regions (neck, arms, torso).

**Current Status**: Torso skin layer not yet implemented (reserved as `OcclusionLayer.TORSO_SKIN`).

**Future Use Cases**:
- Rendering visible skin for tank tops
- Showing arms for sleeveless garments
- Neck visibility for low necklines

---

## Reserved Abstractions (Future Complexity)

### Accessories Layer (Not Yet Implemented)

**Challenge**: Where do scarves, necklaces, glasses fit in the dominance hierarchy?

**Proposed Rules** (NOT FINAL):
- Necklaces: Between GARMENT and FACE
- Scarves: Between GARMENT and HAIR (context-dependent)
- Glasses: Always dominate everything (FACE + 1)

**Action Required**: Define `OcclusionLayer.ACCESSORIES` when needed.

---

### Multi-Garment Layering (Not Yet Implemented)

**Challenge**: How do we handle jackets over shirts?

**Proposed Rules** (NOT FINAL):
- Outerwear: New layer GARMENT_OUTER
- Base layer: Existing GARMENT becomes GARMENT_INNER
- Dominance: GARMENT_OUTER > GARMENT_INNER

**Action Required**: Split GARMENT layer when multi-garment support added.

---

### Short Hair vs High Collars (Not Yet Implemented)

**Challenge**: Short hair above collar line should be dominated by high-collar garments.

**Proposed Rules** (NOT FINAL):
- Classify hair as SHORT or LONG
- SHORT hair: Can be dominated by collar if `hair_bottom_y < collar_top_y`
- LONG hair: Always dominates

**Action Required**: Add hair length classification.

---

## Conflict Resolution Priority

When multiple rules apply, resolve in this order:

1. **Face dominance** (highest priority)
2. **Hair dominance**
3. **Geometric safety** (suppress over-rendering)
4. **Semantic parsing** (lowest priority)

**Example Conflict**:
- Parsing says: "upper-clothes covers face"
- Geometry says: "torso is below face"
- Face rule says: "face always wins"

**Resolution**: Face wins → Garment suppressed in face region.

---

## Design Invariants

These invariants MUST hold at all times:

1. **No garment pixels on face** (except parsing errors, which geometry mitigates)
2. **No garment pixels on hair** (except short-hair edge case, not yet implemented)
3. **Garment constrained to torso geometry** (when pose available)
4. **Layer ordering validated at runtime** (`OcclusionLayer.validate_ordering()`)

---

## Testing Requirements

Any change to dominance rules MUST include:

1. **Golden artifact test** showing before/after
2. **Edge case validation** (side profile, head tilt, long hair)
3. **Regression test** ensuring no face/hair coverage

---

## Modification Protocol

To change any rule in this document:

1. **Propose change** with visual examples
2. **Review edge cases** (accessories, multi-garment, hair length)
3. **Update `OcclusionLayer` enum** if needed
4. **Update tests** to validate new rule
5. **Document in this file** with rationale

**Do not modify dominance rules in code without updating this document.**

---

## Summary

| Layer | Dominates | Dominated By | Notes |
|-------|-----------|--------------|-------|
| HAIR | GARMENT, TORSO_SKIN | FACE (future: short hair exception) | Always on top |
| FACE | GARMENT, TORSO_SKIN, HAIR | None | Highest priority |
| GARMENT | TORSO_SKIN, BACKGROUND | FACE, HAIR, Geometry | Constrained by geometry |
| TORSO_SKIN | BACKGROUND | GARMENT, FACE, HAIR | Not yet implemented |
| BACKGROUND | None | Everything | Base layer |

**Key Principle**: When in doubt, prefer **visual plausibility** over **semantic completeness**.

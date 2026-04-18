# Module Architecture Sketch (Paper Design)

## Purpose

This document sketches the **future module architecture** for the AR Mirror semantic parsing system. This is a **design constraint**, not an implementation plan.

**Status**: PAPER ONLY - Do not implement yet.

**Goal**: Define clean boundaries before entanglement happens.

---

## Current Monolith Problem

The `SemanticParser` class currently does too much:

```python
class SemanticParser:
    # Parsing
    def parse(frame, pose_landmarks) -> masks
    
    # Geometry
    def _calculate_torso_geometry(pose_landmarks) -> torso_mask
    def _apply_geometric_constraints(masks, torso_mask) -> constrained_masks
    
    # Occlusion policy
    # (Implicit in create_occlusion_aware_composite)
    
    # Composition
    # (In create_occlusion_aware_composite function)
```

**Problem**: Future features (accessories, multi-garment, size-fitting) will entangle these concerns.

---

## Proposed Module Split

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        AR Mirror App                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Compositor (Dumb Executor)               │
│  • Receives: Layers with alpha masks                        │
│  • Outputs: Final composite frame                           │
│  • Logic: Pure alpha blending, no decisions                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Occlusion Policy       │  │   Garment Warper         │
│  • Receives: Masks       │  │  • Receives: Garment     │
│  • Outputs: Layer order  │  │  • Outputs: Warped       │
│  • Logic: Dominance      │  │  • Logic: Perspective    │
│    rules (LAYER_         │  │    transform             │
│    DOMINANCE.md)         │  └──────────────────────────┘
└──────────────────────────┘
                    ▲
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐  ┌──────────────────┐
│  Parsing Module  │  │ Geometry Module  │
│  • Receives:     │  │  • Receives:     │
│    Frame         │  │    Pose          │
│  • Outputs:      │  │  • Outputs:      │
│    Semantic      │  │    Validity      │
│    masks         │  │    regions       │
│  • Logic: ONNX   │  │  • Logic: Torso  │
│    inference     │  │    geometry      │
└──────────────────┘  └──────────────────┘
```

---

## Module Contracts

### 1. Parsing Module

**Responsibility**: Produce semantic body part masks from frame.

**Interface**:
```python
class ParsingModule:
    def parse(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            frame: Input BGR image (HxWx3)
        
        Returns:
            Dict of binary masks:
            {
                'hair': np.ndarray (HxW, uint8, 0-255),
                'face': np.ndarray (HxW, uint8, 0-255),
                'neck': np.ndarray (HxW, uint8, 0-255),
                'upper_body': np.ndarray (HxW, uint8, 0-255),
                'arms': np.ndarray (HxW, uint8, 0-255),
                'lower_body': np.ndarray (HxW, uint8, 0-255),
            }
        """
```

**Data Contract**:
- Input: BGR image, any resolution
- Output: Binary masks (0 or 255), same resolution as input
- Coordinate system: Image space (pixels)

**No knowledge of**: Geometry, occlusion policy, composition

---

### 2. Geometry Module

**Responsibility**: Produce validity regions from pose landmarks.

**Interface**:
```python
class GeometryModule:
    def calculate_validity_regions(
        self, 
        pose_landmarks: List[Landmark],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: (height, width)
        
        Returns:
            Dict of validity masks:
            {
                'torso': np.ndarray (HxW, uint8, 0-255),
                'arms': np.ndarray (HxW, uint8, 0-255),  # Future
                'legs': np.ndarray (HxW, uint8, 0-255),  # Future
            }
        """
```

**Data Contract**:
- Input: Normalized pose landmarks (0-1), frame dimensions
- Output: Binary validity masks (0 or 255)
- Coordinate system: Image space (pixels)

**No knowledge of**: Semantic parsing, occlusion policy, composition

---

### 3. Occlusion Policy Module

**Responsibility**: Define layer ordering and dominance rules.

**Interface**:
```python
class OcclusionPolicy:
    def apply_policy(
        self,
        semantic_masks: Dict[str, np.ndarray],
        validity_regions: Dict[str, np.ndarray]
    ) -> List[Layer]:
        """
        Args:
            semantic_masks: From ParsingModule
            validity_regions: From GeometryModule
        
        Returns:
            List of layers in rendering order (back to front):
            [
                Layer(name='background', mask=None, priority=0),
                Layer(name='torso_skin', mask=..., priority=1),
                Layer(name='garment', mask=..., priority=2),
                Layer(name='face', mask=..., priority=3),
                Layer(name='hair', mask=..., priority=4),
            ]
        """
```

**Data Contract**:
- Input: Semantic masks + validity regions
- Output: Ordered list of layers with alpha masks
- Logic: Implements LAYER_DOMINANCE.md rules

**No knowledge of**: How masks were produced, how they will be rendered

---

### 4. Compositor (Dumb Executor)

**Responsibility**: Render layers in order with alpha blending.

**Interface**:
```python
class Compositor:
    def composite(
        self,
        base_frame: np.ndarray,
        layers: List[Layer]
    ) -> np.ndarray:
        """
        Args:
            base_frame: Background image (HxWx3, BGR)
            layers: Ordered layers from OcclusionPolicy
        
        Returns:
            Composited frame (HxWx3, BGR, uint8)
        """
```

**Data Contract**:
- Input: Base frame + ordered layers
- Output: Final composite
- Logic: Pure alpha blending, no decisions

**No knowledge of**: Semantics, geometry, policy

---

## Data Flow

```
Frame ──────────────┐
                    ▼
              ┌──────────┐
              │ Parsing  │
              │  Module  │
              └──────────┘
                    │
                    │ semantic_masks
                    ▼
Pose ────────┐  ┌──────────────┐
             ▼  │              │
        ┌──────────┐           │
        │ Geometry │           │
        │  Module  │           │
        └──────────┘           │
             │                 │
             │ validity_regions│
             ▼                 ▼
        ┌─────────────────────────┐
        │   Occlusion Policy      │
        └─────────────────────────┘
                    │
                    │ layers
                    ▼
        ┌─────────────────────────┐
        │      Compositor         │
        └─────────────────────────┘
                    │
                    ▼
              Final Frame
```

---

## Migration Strategy (Future)

### Phase 1: Extract Geometry Module
1. Move `_calculate_torso_geometry()` to `GeometryModule`
2. Move `_apply_geometric_constraints()` to `OcclusionPolicy`
3. Test: No behavior change

### Phase 2: Extract Occlusion Policy
1. Create `OcclusionPolicy` class
2. Move layer dominance logic from `create_occlusion_aware_composite()`
3. Test: No behavior change

### Phase 3: Extract Compositor
1. Create `Compositor` class
2. Move pure blending logic
3. Test: No behavior change

### Phase 4: Refactor ParsingModule
1. Clean up `SemanticParser` to only do parsing
2. Remove temporal smoothing (move to policy or compositor)
3. Test: No behavior change

---

## Benefits of This Architecture

### 1. Clear Separation of Concerns
- Parsing: "What is this?"
- Geometry: "Where is this valid?"
- Policy: "What order should these render?"
- Compositor: "Render this order"

### 2. Testability
- Each module can be tested independently
- Mock interfaces for unit tests
- Golden artifacts per module

### 3. Extensibility
- Add accessories: New layer in policy
- Add multi-garment: Multiple garment layers
- Add depth: New geometry module output

### 4. Maintainability
- Changes to parsing don't affect composition
- Changes to geometry don't affect policy
- Clear contracts prevent entanglement

---

## What NOT to Do

❌ **Don't implement this now**
- Current monolith works
- Premature abstraction is expensive

❌ **Don't add features to current monolith**
- Will make migration harder
- Entanglement will increase

✅ **Do keep this design in mind**
- When adding features, ask: "Which module?"
- When refactoring, move toward this architecture
- When debugging, respect these boundaries

---

## Review Checklist

Before implementing this architecture:

- [ ] Option 1 complete (contracts documented)
- [ ] Option 2 complete (failure taxonomy defined)
- [ ] Multi-garment requirements defined
- [ ] Accessories requirements defined
- [ ] Performance budget established

**Do not implement until all checkboxes are complete.**

---

## Summary

This architecture separates:

| Module | Input | Output | Logic |
|--------|-------|--------|-------|
| Parsing | Frame | Semantic masks | ONNX inference |
| Geometry | Pose | Validity regions | Torso calculation |
| Policy | Masks + Regions | Ordered layers | Dominance rules |
| Compositor | Layers | Final frame | Alpha blending |

**Key Principle**: Each module has **one responsibility** and **no knowledge** of other modules' internals.

**Status**: Paper design only. Implement when requirements constrain it.

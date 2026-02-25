# Module Contracts for AR Mirror Perception System

## Purpose

This document defines **interface contracts** for the AR Mirror module architecture. Each module owns a **single authority**, not a single algorithm.

**Critical Principle**: Modules are separated by **responsibility** (what they answer), not by **implementation** (how they compute).

**Status**: INTERFACE CONTRACTS ONLY - No implementations yet.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        AR Mirror App                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Compositor    │
                    │   (Rendering    │
                    │    Authority)   │
                    └─────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ Occlusion Policy │  │  Garment Warper  │
        │     (Trust       │  │                  │
        │    Authority)    │  └──────────────────┘
        └──────────────────┘
                    ▲
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐  ┌──────────────────┐
│  Parsing Module  │  │ Pose & Geometry  │
│   (Semantic      │  │   (Anatomical    │
│   Authority)     │  │    Authority)    │
└──────────────────┘  └──────────────────┘
```

**Future Addition**:
```
┌──────────────────┐
│ Depth/Occlusion  │
│   (Spatial       │
│   Authority)     │
└──────────────────┘
```

---

## Module 1: Parsing Module (Semantic Authority)

### Authority

**Answers**: *"What category does this pixel belong to?"*

### Responsibilities

- Classify pixels into semantic categories (hair, face, neck, upper_body, arms, lower_body)
- Provide confidence/uncertainty for classifications
- Maintain taxonomy consistency

### Does NOT Handle

- ❌ Geometric validity (torso boundaries)
- ❌ Dominance rules (what beats what)
- ❌ User trust priorities
- ❌ Rendering decisions

### Interface Contract

```python
class ParsingModule:
    """Semantic Authority: Pixel classification"""
    
    def parse(self, frame: np.ndarray) -> ParsingResult:
        """
        Classify frame pixels into semantic categories
        
        Args:
            frame: Input BGR image (HxWx3, uint8)
        
        Returns:
            ParsingResult with:
            - masks: Dict[str, np.ndarray] (HxW, uint8, 0-255)
            - confidence: Dict[str, np.ndarray] (HxW, float32, 0-1)
            - metadata: ParsingMetadata
        
        Guarantees:
            - All masks same resolution as input
            - Confidence maps align with masks
            - Deterministic output (no randomness)
        """
```

### Input Guarantees

- Frame is valid BGR image (HxWx3, uint8)
- Frame dimensions ≥ 64x64 (minimum supported)

### Output Guarantees

```python
@dataclass
class ParsingResult:
    masks: Dict[str, np.ndarray]  # Binary masks (0 or 255)
    confidence: Dict[str, np.ndarray]  # Per-pixel confidence (0-1)
    metadata: ParsingMetadata
    
@dataclass
class ParsingMetadata:
    native_resolution: Tuple[int, int]  # Model's native resolution
    upscaled: bool  # Whether masks were upscaled
    taxonomy_version: str  # e.g., "LIP_v1"
    backend: str  # "ONNX" or "MediaPipe"
```

**Critical Addition**: Confidence/entropy propagation
- Policy needs this to make informed dominance decisions
- Hairline/collar conflict partially owned here

### Failure Ownership

| Failure Mode | Ownership | Responsibility |
|--------------|-----------|----------------|
| Hairline/Collar Conflict | **Primary** | Provide high-resolution masks + confidence |

### Performance Contract

- Parse time: < 100ms @ 640×480 (target)
- Memory: < 500MB (ONNX model + inference)
- Thread-safe: No (single-threaded inference)

### Semantic Tiers (Product-Level Extension)

**Purpose**: Different semantic regions have different trust requirements.

```python
@dataclass
class SemanticTiers:
    """Semantic regions grouped by trust criticality"""
    
    identity_critical: List[str] = ["face", "hair"]
    # Zero tolerance for errors, highest confidence required
    
    interaction_regions: List[str] = ["neck", "upper_body", "arms"]
    # Garment interaction zones, medium confidence acceptable
    
    non_critical: List[str] = ["background", "lower_body"]
    # Low trust impact, can degrade gracefully
```

**Usage**: Policy module uses tiers to make confidence-aware decisions:
- Identity-critical regions: Require confidence > 0.9
- Interaction regions: Require confidence > 0.7
- Non-critical: Accept confidence > 0.5

**Rationale**: Not all parsing errors are equal. Face misclassification is catastrophic, background noise is acceptable.

---

## Product-Level Contracts

### Garment Metadata Contract

**Purpose**: Different garments need different occlusion rules and error tolerances.

```python
@dataclass
class GarmentProfile:
    """Product-level garment metadata"""
    
    # Garment identification
    category: str  # "tshirt", "shirt", "hoodie", "kurta", "saree", "jacket"
    subcategory: str  # "formal_shirt", "casual_tshirt", "oversized_hoodie"
    
    # Physical properties
    neckline: str  # "crew", "v-neck", "collar", "high-collar", "boat-neck"
    fit: str  # "fitted", "regular", "loose", "oversized"
    rigidity: float  # 0.0 (flowy/saree) → 1.0 (rigid/jacket)
    
    # Occlusion behavior
    occlusion_sensitivity: float  # 0.0 (tolerant) → 1.0 (strict)
    # How strict should face/hair occlusion rules be?
    
    hair_overlap_acceptable: bool  # True for hoodies, False for t-shirts
    
    # Error tolerances (per-garment)
    max_collar_error: float  # Maximum acceptable collar-hair conflict
    max_fit_deviation: float  # How much geometric suppression is OK
    
    # Retailer overrides
    brand_id: Optional[str]  # Retailer-specific tuning
    custom_policy: Optional[Dict]  # Brand-specific dominance rules
```

**Integration Point**: Policy module receives `GarmentProfile` and adjusts rules:

```python
class OcclusionPolicyModule:
    def apply_policy(
        self,
        semantic: ParsingResult,
        anatomical: AnatomicalResult,
        garment: GarmentProfile  # NEW: Product-level input
    ) -> PolicyResult:
        # Adjust dominance rules based on garment type
        if garment.hair_overlap_acceptable:
            # Hoodie: relax hair dominance
            pass
        
        if garment.occlusion_sensitivity > 0.8:
            # Formal shirt: strict collar rules
            pass
```

**Garment-Specific Rules**:

| Garment | Acceptable Errors | Policy Adjustments |
|---------|-------------------|-------------------|
| T-shirt | High tolerance | Loose geometric constraint |
| Formal shirt | Very low collar error | Strict hair-collar boundary |
| Hoodie | Hair overlap OK | Relax hair dominance |
| Kurta | Loose fit expected | High geometric suppression OK |
| Saree | Geometry dominates | Semantic parsing low-trust |
| Jacket | Multi-layer | Enable GARMENT_OUTER layer |

**Status**: Contract defined, not yet implemented. Reserve for SaaS phase.

---

## Module 2: Pose & Geometry Module (Anatomical Authority)

### Authority

**Answers**: *"Where is the human body, physically?"*

### Internal Split

This module is internally split by sub-authority:

1. **Pose Estimation**: Landmark detection
2. **Validity Regions**: Geometric boundaries (torso, arms, legs)
3. **Person Selection**: Multi-person handling

### Responsibilities

- Detect pose landmarks (MediaPipe format)
- Calculate anatomical validity regions (torso polygon, arm regions)
- Select primary person in multi-person frames
- Provide pose confidence and quality metrics

### Does NOT Handle

- ❌ Semantic classification (what is hair/face)
- ❌ Garment properties
- ❌ Dominance rules
- ❌ Business logic

### Interface Contract

```python
class PoseGeometryModule:
    """Anatomical Authority: Body location and validity"""
    
    def analyze(self, frame: np.ndarray) -> AnatomicalResult:
        """
        Detect body pose and calculate validity regions
        
        Args:
            frame: Input BGR image (HxWx3, uint8)
        
        Returns:
            AnatomicalResult with:
            - pose: PoseData (landmarks, angles, confidence)
            - validity_regions: Dict[str, np.ndarray] (validity masks)
            - person_selection: PersonSelectionData
        
        Guarantees:
            - Validity regions are conservative (prefer under-coverage)
            - Pose landmarks in normalized coordinates (0-1)
            - Graceful degradation if pose unavailable
        """
```

### Input Guarantees

- Frame is valid BGR image (HxWx3, uint8)
- Frame contains at least partial human body

### Output Guarantees

```python
@dataclass
class AnatomicalResult:
    pose: Optional[PoseData]  # None if pose detection failed
    validity_regions: Dict[str, np.ndarray]  # Geometric validity masks
    person_selection: PersonSelectionData
    
@dataclass
class PoseData:
    landmarks: List[Landmark]  # MediaPipe format (x, y, z, visibility)
    angles: Dict[str, float]  # shoulder_rotation, torso_lean, etc.
    confidence: float  # Overall pose confidence (0-1)
    quality: str  # "frontal", "side_profile", "partial", "occluded"
    
@dataclass
class PersonSelectionData:
    num_people: int
    selected_person_id: int
    selection_confidence: float  # 0-1
```

**Validity Regions**:
```python
validity_regions = {
    'torso': np.ndarray,  # Convex polygon (not axis-aligned rectangle)
    'arms': np.ndarray,   # Future
    'legs': np.ndarray,   # Future
}
```

**Critical Upgrade**: Convex polygon torso (not axis-aligned rectangle)
- Fixes side-profile clipping
- Aligned to shoulder slope

### Failure Ownership

| Failure Mode | Ownership | Responsibility |
|--------------|-----------|----------------|
| Side-Profile Torso Clipping | **Primary** | Provide convex polygon torso |
| Arms Raised Shrinkage | **Primary** | Use shoulder-hip distance (not absolute) |
| Loose Garment Suppression | **Primary** | Conservative validity regions |
| Multi-Person Misbinding | **Primary** | Robust person selection |
| Foreground Object Failure | **Secondary** | Detect hands/objects (until Depth module) |

### Performance Contract

- Pose detection: < 30ms @ 640×480 (target)
- Memory: < 200MB
- Thread-safe: No (single-threaded inference)

---

## Module 3: Occlusion Policy Module (Trust Authority)

### Authority

**Answers**: *"Who wins when signals conflict?"*

### Responsibilities

- Enforce dominance rules (LAYER_DOMINANCE.md)
- Resolve semantic vs geometric conflicts
- Enforce zero-tolerance invariants (P0 failures)
- Produce ordered layers for compositor

### Does NOT Handle

- ❌ Pixel classification
- ❌ Geometric calculations
- ❌ Rendering/blending
- ❌ Performance optimization

### Interface Contract

```python
class OcclusionPolicyModule:
    """Trust Authority: Conflict resolution and layer ordering"""
    
    def apply_policy(
        self,
        semantic: ParsingResult,
        anatomical: AnatomicalResult
    ) -> PolicyResult:
        """
        Resolve conflicts and produce ordered layers
        
        Args:
            semantic: From ParsingModule
            anatomical: From PoseGeometryModule
        
        Returns:
            PolicyResult with:
            - layers: List[Layer] (ordered back-to-front)
            - violations: List[PolicyViolation] (P0 failures)
            - decisions: List[PolicyDecision] (conflict resolutions)
        
        Guarantees:
            - Layer ordering matches OcclusionLayer enum
            - P0 violations detected and reported
            - Geometry-over-semantics rule enforced
        """
```

### Input Guarantees

- Semantic masks are valid (from ParsingModule)
- Anatomical data is valid (from PoseGeometryModule)

### Output Guarantees

```python
@dataclass
class PolicyResult:
    layers: List[Layer]  # Ordered back-to-front
    violations: List[PolicyViolation]  # P0 failures detected
    decisions: List[PolicyDecision]  # Conflict resolutions logged
    
@dataclass
class Layer:
    name: str  # "background", "torso_skin", "garment", "face", "hair"
    mask: np.ndarray  # Alpha mask (HxW, uint8, 0-255)
    priority: int  # OcclusionLayer enum value
    source: str  # "semantic", "geometric", "hybrid"
    
@dataclass
class PolicyViolation:
    violation_type: str  # "face_contamination", etc.
    severity: float  # 0-1
    frame_number: int
    
@dataclass
class PolicyDecision:
    conflict_type: str  # "semantic_vs_geometric", "hair_vs_garment"
    winner: str  # "semantic", "geometric", "policy_override"
    rationale: str  # Human-readable explanation
```

**Critical Feature**: Confidence-aware decisions
- Use semantic confidence to inform dominance
- Low-confidence semantic → defer to geometric
- High-confidence semantic → challenge geometric (with logging)

### Dominance Rules (from LAYER_DOMINANCE.md)

1. Face always dominates garment
2. Hair always dominates garment (except short-hair future exception)
3. Geometry can suppress semantics, never invert
4. Garment dominates torso skin only

### Failure Ownership

| Failure Mode | Ownership | Responsibility |
|--------------|-----------|----------------|
| Face/Neck Contamination | **Primary** | Enforce face dominance (P0) |

### Performance Contract

- Policy application: < 10ms @ 640×480 (target)
- Memory: < 50MB
- Thread-safe: Yes (stateless)

---

## Module 4: Compositor (Rendering Authority)

### Authority

**Answers**: *"How do pixels get blended?"*

### Responsibilities

- Alpha blend layers in order
- Optimize rendering performance
- Handle color space conversions

### Does NOT Handle

- ❌ Layer ordering decisions (Policy owns this)
- ❌ Semantic classification
- ❌ Geometric constraints
- ❌ Business logic

**Critical Rule**: Compositor is **policy-dumb**. No "if face then..." logic.

### Interface Contract

```python
class Compositor:
    """Rendering Authority: Pure alpha blending"""
    
    def composite(
        self,
        base_frame: np.ndarray,
        layers: List[Layer]
    ) -> np.ndarray:
        """
        Blend layers onto base frame
        
        Args:
            base_frame: Background image (HxWx3, BGR, uint8)
            layers: Ordered layers from OcclusionPolicy (back-to-front)
        
        Returns:
            Composited frame (HxWx3, BGR, uint8)
        
        Guarantees:
            - Layers blended in order (no reordering)
            - Alpha blending mathematically correct
            - Output same resolution as input
        """
```

### Input Guarantees

- Base frame is valid BGR image
- Layers are ordered (back-to-front)
- All layer masks same resolution as base frame

### Output Guarantees

- Output frame same resolution as input
- Color space preserved (BGR)
- No clipping artifacts (proper alpha blending)

### Failure Ownership

None - Compositor does not own failures (rendering is deterministic)

### Performance Contract

- Composite time: < 5ms @ 640×480 (target)
- Memory: < 100MB
- Thread-safe: Yes (stateless)

---

## Module 5: Depth/Occlusion Module (Spatial Authority) - FUTURE

### Authority

**Answers**: *"What is in front of what, physically?"*

### Responsibilities

- Estimate depth from monocular image
- Detect foreground objects (hands, bags, phones)
- Provide spatial occlusion masks

### Does NOT Handle

- ❌ Semantic classification (Parsing owns)
- ❌ Anatomical landmarks (PoseGeometry owns)
- ❌ Dominance rules (Policy owns)

**Critical**: This is **NOT** part of Geometry module. Separate authority.

### Interface Contract (Future)

```python
class DepthOcclusionModule:
    """Spatial Authority: Physical occlusion"""
    
    def analyze(self, frame: np.ndarray) -> SpatialResult:
        """
        Estimate depth and detect foreground objects
        
        Args:
            frame: Input BGR image (HxWx3, uint8)
        
        Returns:
            SpatialResult with:
            - depth_map: np.ndarray (HxW, float32, meters)
            - foreground_objects: List[ObjectMask]
            - occlusion_order: List[int] (layer IDs by depth)
        """
```

### Failure Ownership

| Failure Mode | Ownership | Responsibility |
|--------------|-----------|----------------|
| Foreground Object Occlusion Failure | **Primary** | Detect objects in front of torso |

---

## Cross-Module Data Contracts

### Coordinate Systems

All modules use **image space** (pixels):
- Origin: Top-left (0, 0)
- X-axis: Left to right
- Y-axis: Top to bottom
- Normalized coordinates: (0, 1) range

### Mask Format

All masks follow same format:
- Type: `np.ndarray`
- Shape: `(H, W)`
- Dtype: `uint8`
- Values: `0` (background) or `255` (foreground)

### Confidence Format

All confidence maps follow same format:
- Type: `np.ndarray`
- Shape: `(H, W)`
- Dtype: `float32`
- Values: `[0.0, 1.0]` (0 = no confidence, 1 = certain)

---

## Failure-to-Module Traceability

| Failure Mode | Primary Module | Secondary Module | Signal Used | Guarantee |
|--------------|----------------|------------------|-------------|-----------|
| Face/Neck Contamination | Occlusion Policy | Pose & Geometry | Face mask + torso validity | Zero tolerance (P0) |
| Hairline/Collar Conflict | Parsing | Occlusion Policy | Hair mask + confidence | < 15% violation rate |
| Side-Profile Clipping | Pose & Geometry | - | Convex torso polygon | < 20% violation rate |
| Loose Garment Suppression | Pose & Geometry | - | Conservative validity | < 25% violation (acceptable) |
| Arms Raised Shrinkage | Pose & Geometry | - | Shoulder-hip distance | < 30% violation rate |
| Foreground Object Failure | Depth/Occlusion (future) | Pose & Geometry | Depth map + object masks | < 20% violation rate |
| Multi-Person Misbinding | Pose & Geometry | - | Person selection | < 5% violation rate |

**Key Insight**: Every failure has **exactly one primary owner**.

---

## Module Interaction Rules

### Rule 1: No Direct Module-to-Module Calls

Modules communicate through **data contracts only**:
```
❌ BAD:  parsing.get_mask() → policy.check_face() → geometry.validate()
✅ GOOD: app → parsing → policy(parsing_result, geometry_result) → compositor
```

### Rule 2: No Shared Mutable State

Each module is **stateless** or manages its own state:
```
❌ BAD:  global_cache['masks'] = parsing_result
✅ GOOD: parsing_module.cache = parsing_result  # Internal only
```

### Rule 3: Explicit Failure Propagation

Failures bubble up through return values, not exceptions:
```python
@dataclass
class ModuleResult:
    success: bool
    data: Optional[Any]
    error: Optional[str]
```

### Rule 4: Authority Boundaries Are Inviolable

```
❌ BAD:  Parsing module calculates torso geometry
❌ BAD:  Policy module runs ONNX inference
❌ BAD:  Compositor makes dominance decisions
✅ GOOD: Each module stays within its authority
```

---

## Degradation Policy (SaaS Reliability)

### Purpose

Define **graceful degradation rules** when modules fail or performance drops. Critical for SaaS uptime and user experience.

### Per-Module Latency Budgets

| Module | Target Latency | Budget | Action if Exceeded |
|--------|----------------|--------|-------------------|
| Parsing | 100ms | 150ms | Switch to lower resolution |
| Pose & Geometry | 30ms | 50ms | Skip geometric constraint |
| Occlusion Policy | 10ms | 20ms | Use cached policy |
| Compositor | 5ms | 10ms | Reduce layer count |
| **Total Pipeline** | **145ms** | **230ms** | **Disable overlay** |

**FPS Target**: ≥ 14 FPS (70ms per frame budget)

### Degradation Triggers

#### 1. Parsing Confidence Drop

**Trigger**: `avg(semantic.confidence) < 0.6`

**Action**:
1. Disable garment overlay
2. Show "Adjusting position..." message
3. Log frame for analysis
4. Resume when confidence > 0.7

**Rationale**: Low-confidence parsing → high error rate → breaks trust

---

#### 2. FPS Drop

**Trigger**: `fps < 14` for 3 consecutive seconds

**Action**:
1. Reduce parsing resolution: 473×473 → 384×384
2. Disable temporal smoothing (save compute)
3. If still slow: Disable geometric constraint
4. If still slow: Disable overlay

**Rationale**: Choppy experience worse than simplified rendering

---

#### 3. Pose Confidence Drop

**Trigger**: `pose.confidence < 0.5`

**Action**:
1. Skip geometric constraint (use semantic only)
2. Increase face/hair dominance margins
3. Log "partial pose" event

**Rationale**: Bad pose → bad torso geometry → prefer semantic

---

#### 4. P0 Violation Detected

**Trigger**: `policy.violations` contains P0 failure (face contamination)

**Action**:
1. **Immediately freeze garment overlay**
2. Show last-known-good frame
3. Alert user: "Repositioning..."
4. Resume only when violation cleared for 5 frames

**Rationale**: Zero tolerance for face contamination

---

#### 5. Memory Pressure

**Trigger**: System memory warning (iOS/Android)

**Action**:
1. Clear parsing cache
2. Reduce model precision (FP32 → FP16)
3. Disable temporal smoothing
4. If critical: Unload ONNX model, use MediaPipe fallback

**Rationale**: Prevent app crash, maintain core functionality

---

### Degradation Hierarchy (What Shuts Off First)

Priority order (highest to lowest):

1. **Keep**: Face/hair parsing (identity-critical)
2. **Keep**: P0 violation detection
3. **Degrade**: Temporal smoothing → OFF
4. **Degrade**: Geometric constraint → OFF
5. **Degrade**: Parsing resolution → 384×384 → 256×256
6. **Degrade**: Garment overlay → FREEZE last-good-frame
7. **Disable**: Garment overlay → OFF

**Never Degrade**: P0 enforcement (face contamination detection)

---

### Recovery Strategy

**Trigger**: Performance/confidence restored

**Action**:
1. Wait for 3 consecutive "good" frames
2. Re-enable features in reverse order
3. Log recovery event
4. Resume normal operation

**Hysteresis**: Prevent flapping (disable at threshold, re-enable at threshold + margin)

---

### SaaS-Specific Degradation

#### Retailer-Specific Tolerances

```python
@dataclass
class RetailerPolicy:
    brand_id: str
    
    # Performance tolerances
    min_fps: int  # Brand-specific FPS requirement
    max_latency_ms: int  # Maximum acceptable latency
    
    # Quality tolerances
    min_parsing_confidence: float  # When to disable overlay
    max_p1_violation_rate: float  # Acceptable error rate
    
    # Degradation preferences
    prefer_quality_over_fps: bool  # True = maintain quality, False = maintain FPS
    allow_geometric_skip: bool  # Can we skip torso constraint?
```

**Usage**: Policy module adjusts degradation based on retailer preferences.

---

### Monitoring & Alerts

**Metrics to Track**:
- Degradation events per session
- Time spent in degraded mode
- P0 violations (should be ~0)
- Recovery success rate

**Alerts** (SaaS dashboard):
- P0 violation rate > 0.1% → Critical alert
- Degradation rate > 20% → Warning
- FPS < 10 for > 10s → Performance alert

---

## Testing Contracts

### Unit Testing

Each module must provide:
- Mock inputs (valid data contracts)
- Golden outputs (frozen test cases)
- Failure mode tests (from FAILURE_TAXONOMY.md)

### Integration Testing

Module pairs must test:
- Contract compatibility (output → input)
- Failure propagation
- Performance under load

---

## Migration Strategy (From Current Monolith)

### Phase 1: Extract Parsing Module
1. Move ONNX/MediaPipe logic to `ParsingModule`
2. Add confidence/metadata outputs
3. Test: No behavior change

### Phase 2: Extract Pose & Geometry Module
1. Move pose detection to `PoseGeometryModule`
2. Upgrade to convex polygon torso
3. Add person selection logic
4. Test: No behavior change (or documented improvements)

### Phase 3: Extract Occlusion Policy Module
1. Move dominance rules to `OcclusionPolicyModule`
2. Add confidence-aware decisions
3. Add P0 violation detection
4. Test: No behavior change

### Phase 4: Extract Compositor
1. Move alpha blending to `Compositor`
2. Remove all policy logic
3. Test: No behavior change

### Phase 5: Add Depth/Occlusion Module (Future)
1. Implement depth estimation
2. Add foreground object detection
3. Integrate with Policy module

---

## Design Review Checklist

Before implementation:

- [ ] Each module owns exactly one authority
- [ ] No module knows about other modules' internals
- [ ] All data contracts explicitly defined
- [ ] Every failure mode has exactly one primary owner
- [ ] No "if face then..." logic in Compositor
- [ ] Parsing provides confidence/entropy (not just masks)
- [ ] Geometry uses convex polygon (not axis-aligned rectangle)
- [ ] Policy enforces P0 invariants
- [ ] Depth/Occlusion is separate from Geometry

---

## Summary

| Module | Authority | Answers | Owns (Failures) |
|--------|-----------|---------|-----------------|
| Parsing | Semantic | "What category?" | Hairline/collar conflict |
| Pose & Geometry | Anatomical | "Where is body?" | Side-profile, arms raised, loose garment, multi-person |
| Occlusion Policy | Trust | "Who wins?" | Face contamination (P0) |
| Compositor | Rendering | "How to blend?" | None (deterministic) |
| Depth/Occlusion | Spatial | "What's in front?" | Foreground objects (future) |

**Critical Principle**: Modules separated by **authority** (responsibility), not **algorithm** (implementation).

**Status**: INTERFACE CONTRACTS ONLY - No implementations yet. Ready for design review.

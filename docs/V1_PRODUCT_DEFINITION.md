# V1 Product Definition: Kurta Upper-Body Try-On

## Purpose

This document defines the **Non-Negotiable v1** for AR Mirror SDK. This is the minimum production slice that can be sold to retailers.

**Critical Constraint**: If it's not listed here, **do not build it**.

---

## What v1 IS

> "Upper-body kurta fit + length + silhouette confirmation"

**Not**: Photorealistic rendering  
**Not**: Full-body try-on  
**Not**: Physics simulation  

**Yes**: Trust-preserving overlay that reduces size returns

---

## v1 MUST Do (Non-Negotiable)

### 1. Upper-Body Kurta Overlay Only

**Scope**:
- Chest, shoulders, sleeves
- Stop at hips (no lower-body logic)
- Single-layer garment (no dupatta, no jacket)

**Garment Types Supported**:
- Straight kurta (regular fit)
- A-line kurta (loose fit)
- Pathani kurta (collar variant)

**Garment Types NOT Supported** (v1):
- ❌ Saree
- ❌ Dupatta
- ❌ Jacket/Sherwani (multi-layer)
- ❌ Anarkali (complex draping)

---

### 2. Strict P0 Guarantees

**Zero Tolerance**:
- Face never occluded (0.0% violation rate)
- Hair always on top of garment
- No jitter when user is still (< 0.1 IoU delta)

**Implementation**:
- Runtime P0 violation detection
- Immediate freeze on violation
- Resume only after 5 clean frames

**Failure Mode**: Disable overlay, show "Repositioning..." message

---

### 3. Deterministic Behavior

**Requirement**: Same input → same output

**No Randomness**:
- No temporal noise
- No stochastic smoothing
- No random initialization

**Testing**: Golden frame regression tests

---

### 4. Single Person, Frontal Bias

**Supported**:
- Single person in frame
- Frontal pose (0-30° rotation)
- Side profile (30-60° rotation) with degradation

**NOT Supported** (v1):
- ❌ Multi-person (disable with explanation)
- ❌ Back view (disable)
- ❌ Extreme rotations (> 60°, disable)

**Degradation Strategy**:
- Side profile: Allow geometric clipping (documented trade-off)
- Extreme pose: Disable overlay, show "Face camera" message

---

### 5. Retailer-Controlled Policies

**Knobs Exposed**:

```python
@dataclass
class RetailerConfig:
    # Occlusion behavior
    occlusion_sensitivity: float  # 0.7 (relaxed) → 0.95 (strict)
    
    # Performance vs quality
    prefer_quality_over_fps: bool  # True = maintain quality, False = FPS
    
    # Degradation
    min_confidence_threshold: float  # When to disable overlay
    allow_geometric_skip: bool  # Can we skip torso constraint?
    
    # Garment-specific
    sleeve_fitting_enabled: bool  # Show sleeve fit or just torso?
    length_preview_enabled: bool  # Show kurta length below hips?
```

**Usage**: Retailer dashboard (cloud config, no app update needed)

---

## v1 MUST NOT Do (Forbidden)

### 1. Garment Physics

**NO**:
- Cloth simulation
- Wrinkle rendering
- Fabric flow

**Rationale**: Breaks determinism, adds latency, low ROI for kurtas

---

### 2. Depth Estimation

**NO** (v1):
- Monocular depth
- Foreground object detection
- Spatial occlusion

**Rationale**: Adds complexity, not critical for frontal kurta try-on

**Future**: Reserve for v2 (jackets, accessories)

---

### 3. Multiple Garments Stacked

**NO**:
- Kurta + dupatta
- Kurta + jacket
- Layered rendering

**Rationale**: Multi-layer policy is complex, defer to v2

---

### 4. Accessories

**NO**:
- Necklaces
- Glasses
- Watches

**Rationale**: Out of scope for kurta vertical

---

### 5. Realism Over Trust

**NO**:
- Photorealistic lighting
- Shadow rendering
- Texture detail enhancement

**Rationale**: Trust > realism. Prefer under-rendering to over-rendering.

---

## SDK Interface (v1)

### Input

```python
class ARMirrorSDK:
    def process_frame(
        self,
        frame: np.ndarray,  # BGR image (HxWx3)
        garment: GarmentAsset,  # Preprocessed garment
        config: RetailerConfig  # Retailer policy
    ) -> SDKResult
```

### Output

```python
@dataclass
class SDKResult:
    # Rendering
    composite_frame: np.ndarray  # RGBA output
    
    # Telemetry
    fps: float
    confidence: float  # Overall confidence (0-1)
    violations: List[PolicyViolation]  # P0 failures
    
    # Status
    status: str  # "active", "degraded", "disabled"
    status_reason: Optional[str]  # "low_confidence", "multi_person", etc.
    
    # Debug (QA only, not exposed to end-user)
    debug_masks: Optional[Dict[str, np.ndarray]]
    debug_metrics: Optional[Dict[str, float]]
```

---

## Garment Asset Format (v1)

### Preprocessed Package

Retailers upload garment image, cloud preprocesses to:

```
garment_package/
├── texture.png          # RGBA texture (alpha mask applied)
├── metadata.json        # GarmentProfile
└── thumbnail.png        # Preview image
```

### GarmentProfile (v1)

```json
{
  "category": "kurta",
  "subcategory": "straight_kurta",
  "neckline": "mandarin_collar",
  "fit": "regular",
  "rigidity": 0.3,
  "occlusion_sensitivity": 0.8,
  "hair_overlap_acceptable": false,
  "max_collar_error": 0.1,
  "max_fit_deviation": 0.4
}
```

---

## Performance Targets (v1)

| Metric | Target | Budget | Action if Exceeded |
|--------|--------|--------|-------------------|
| FPS | ≥ 14 | ≥ 10 | Reduce parsing resolution |
| Latency (total) | < 150ms | < 230ms | Disable overlay |
| P0 Violation Rate | 0.0% | 0.1% | Critical alert |
| Jitter (still user) | < 0.05 | < 0.1 | Increase smoothing |
| Memory | < 800MB | < 1GB | Reduce model precision |

---

## Retailer Onboarding (v1)

### Step 1: Garment Upload

Retailer uploads 10-20 kurta images via dashboard.

### Step 2: Cloud Preprocessing

Cloud pipeline:
1. Background removal
2. Garment segmentation
3. Generate GarmentProfile (auto + manual review)
4. Package to CDN

**SLA**: 24-hour turnaround

---

### Step 3: SDK Integration

Retailer integrates SDK (iOS or Android):

```swift
// iOS Example
import ARMirrorSDK

let sdk = ARMirrorSDK(apiKey: "retailer_key")

sdk.configure(config: RetailerConfig(
    occlusion_sensitivity: 0.8,
    prefer_quality_over_fps: true
))

sdk.loadGarment(garmentID: "kurta_123") { result in
    // Start AR session
}
```

**Integration Time**: 2-3 days (with support)

---

### Step 4: Pilot Launch

- 1-2 kurta SKUs live
- Internal QA testing
- Collect failure footage
- Iterate policy (not models)

**Pilot Duration**: 2-4 weeks

---

## Success Metrics (v1 Pilot)

### Technical Metrics

- P0 violation rate: **< 0.1%**
- FPS: **≥ 14** (90th percentile)
- Crash rate: **< 0.5%**
- Degradation rate: **< 20%** of sessions

### Business Metrics

- Conversion rate lift: **> 15%** (vs non-AR)
- Return rate drop: **> 10%** (vs non-AR)
- Session duration: **> 2 minutes** (engagement)
- Retailer NPS: **> 8/10**

---

## What Gets Deferred to v2

### Depth & Occlusion

- Foreground object detection
- Depth-based occlusion
- Hand/phone masking

**Rationale**: Not critical for frontal kurta try-on

---

### Multi-Garment

- Kurta + dupatta
- Kurta + jacket
- Layered rendering

**Rationale**: Adds policy complexity, low initial demand

---

### Advanced Garments

- Saree (complex draping)
- Anarkali (flared geometry)
- Sherwani (heavy embroidery)

**Rationale**: Requires garment-specific logic, defer until kurta proven

---

### Photorealism

- Realistic lighting
- Shadow rendering
- Fabric detail

**Rationale**: Trust > realism, not required for size confirmation

---

## v1 Constraints Summary

| Aspect | v1 Scope | NOT v1 |
|--------|----------|--------|
| Garment | Upper-body kurta | Saree, dupatta, jacket |
| Coverage | Chest to hips | Full-body, lower-body |
| Pose | Frontal + side (degraded) | Back view, extreme rotation |
| People | Single person | Multi-person |
| Layers | Single garment | Multi-layer stacking |
| Physics | None | Cloth simulation |
| Depth | None | Monocular depth, foreground objects |
| Realism | Trust-preserving overlay | Photorealistic rendering |

---

## Non-Negotiable Principle

> **If it's not required to reduce kurta returns for a frontal-facing single user, it's not in v1.**

Every feature request must pass this filter.

---

## Summary

**v1 Product**: Upper-body kurta fit + length + silhouette confirmation

**v1 Promise**: Zero face contamination, deterministic behavior, retailer-controlled policies

**v1 Exclusions**: Physics, depth, multi-garment, photorealism, accessories

**v1 Success**: 15% conversion lift, 10% return drop, < 0.1% P0 violations

**Status**: Constrained, executable, sellable.

# SaaS Product Strategy & Platform Architecture

## Purpose

This document defines the **product strategy** and **SaaS packaging** for AR Mirror as a retail platform, not just a perception system.

**Critical Insight**: The SaaS is **not the AR app**. The SaaS is the **perception SDK + cloud infrastructure + analytics**.

---

## Product Philosophy

### What We Are NOT Building

❌ General-purpose clothing try-on  
❌ Consumer-facing AR app  
❌ Fashion social network  
❌ Virtual wardrobe manager  

### What We ARE Building

✅ **Perception SDK** for retailers to embed in their apps  
✅ **Cloud garment preprocessing** pipeline  
✅ **Analytics dashboard** for trust metrics  
✅ **Policy tuning** per retailer/brand  

**Business Model**: B2B SaaS, not B2C app.

---

## Vertical-First Strategy

### Phase 1: Single Vertical Dominance

**Do NOT start with "general clothing"**. Pick ONE vertical and win it brutally well.

#### Option A: Formal Shirts (India Market)

**Why**:
- High purchase intent (office wear)
- Strict fit requirements → high return rate → strong ROI
- Clear success metric: collar fit accuracy
- Existing pain point: online shirt sizing is terrible

**Target Retailers**: Van Heusen, Peter England, Raymond

**Success Metric**: < 5% return rate due to fit (vs industry 20-30%)

---

#### Option B: Kurtas (India-Specific Advantage)

**Why**:
- Massive market (festivals, weddings, daily wear)
- Loose fit → geometric constraint works perfectly
- Less competition (Western AR focuses on Western wear)
- Cultural advantage (understanding draping, embroidery visibility)

**Target Retailers**: Manyavar, Fabindia, Biba

**Success Metric**: 2x time-on-product-page vs non-AR

---

#### Option C: T-Shirts (Global Market)

**Why**:
- Simplest garment (high tolerance for errors)
- Huge market volume
- Fast fashion → frequent purchases
- Easiest to prove ROI

**Target Retailers**: H&M, Zara, Uniqlo (India)

**Success Metric**: 30% increase in conversion rate

---

### Recommended: **Kurtas First**

**Rationale**:
1. **India-specific advantage** (less global competition)
2. **Loose fit** → current geometric constraint works well
3. **High cultural relevance** → strong product-market fit
4. **Clear differentiation** → not competing with Western AR platforms

**Proof Point**: "AR Mirror is the only platform that understands Indian ethnic wear"

---

## SaaS Architecture

### Layer 1: Perception SDK (Native)

**Platform**: C++ / Rust core, platform bindings (iOS/Android)

**NOT Flutter** - Flutter is UI shell only.

```
┌─────────────────────────────────────┐
│  Retailer App (Flutter/React Native)│
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  AR Mirror SDK (Native Bridge)      │
│  - iOS: Swift/Objective-C wrapper   │
│  - Android: Kotlin/JNI wrapper      │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Perception Core (C++/Rust)         │
│  - ONNX Runtime / TensorRT          │
│  - OpenCV / Eigen                   │
│  - Module architecture (contracts)  │
└─────────────────────────────────────┘
```

**Why Native**:
- GPU acceleration (Metal/Vulkan)
- Model updates without app recompile
- Performance ceiling (60 FPS possible)
- Cross-platform code reuse

**SDK API** (retailer-facing):

```swift
// iOS Example
let arMirror = ARMirrorSDK(apiKey: "retailer_key")

arMirror.configure(
    garmentProfile: GarmentProfile(
        category: "kurta",
        neckline: "mandarin_collar",
        fit: "regular"
    ),
    policy: PolicyConfig(
        occlusion_sensitivity: 0.8,
        degradation_preference: .quality_over_fps
    )
)

arMirror.startSession { result in
    switch result {
    case .success(let composite):
        // Display composite frame
    case .degraded(let reason):
        // Show "Adjusting..." message
    case .failure(let error):
        // Fallback to static image
    }
}
```

---

### Layer 2: Cloud Garment Preprocessing

**Purpose**: Retailers upload garment images, cloud preprocesses for AR.

**Pipeline**:

```
Retailer uploads garment image
         ↓
Cloud preprocessing:
  1. Background removal
  2. Garment segmentation
  3. Texture extraction
  4. Perspective normalization
  5. Generate GarmentProfile metadata
         ↓
Preprocessed garment package
  - Texture atlas
  - Alpha mask
  - GarmentProfile JSON
  - Thumbnail
         ↓
CDN distribution (fast SDK download)
```

**Business Value**:
- Retailers don't need ML expertise
- Consistent quality across catalog
- A/B test garment rendering
- Update garments without app update

**Pricing**: Per-garment preprocessing fee + storage

---

### Layer 3: Analytics Dashboard

**Purpose**: Retailers see trust metrics, not just engagement.

**Metrics Dashboard**:

```
┌─────────────────────────────────────────────┐
│  AR Mirror Analytics - Manyavar Dashboard   │
├─────────────────────────────────────────────┤
│                                             │
│  Trust Metrics (Last 7 Days)                │
│  ────────────────────────────────────────   │
│  P0 Violations (Face):        0.02%  ✅     │
│  P1 Collar Errors:           12.3%   ⚠️     │
│  Avg Session Duration:        2m 34s        │
│  Conversion Rate (AR):       18.2%   ↑ 4%   │
│  Return Rate (AR users):      8.1%   ↓ 12%  │
│                                             │
│  Top Failure Modes                          │
│  ────────────────────────────────────────   │
│  1. Side-profile clipping    (18% sessions) │
│  2. Hairline conflict        (12% sessions) │
│  3. Arms raised shrinkage    ( 8% sessions) │
│                                             │
│  Garment Performance                        │
│  ────────────────────────────────────────   │
│  Kurta SKU-1234:  95% success rate  ✅      │
│  Kurta SKU-5678:  78% success rate  ⚠️      │
│    → Recommendation: Adjust GarmentProfile  │
│                                             │
└─────────────────────────────────────────────┘
```

**Key Insight**: Retailers care about **conversion** and **returns**, not FPS.

---

### Layer 4: Policy Tuning (Per-Retailer)

**Purpose**: Different brands have different tolerance thresholds.

**Example**:

| Retailer | Occlusion Sensitivity | FPS Priority | Degradation Preference |
|----------|----------------------|--------------|------------------------|
| Manyavar (Premium) | 0.9 (strict) | Quality > FPS | Disable if < 0.8 confidence |
| FabIndia (Casual) | 0.7 (relaxed) | FPS > Quality | Allow geometric skip |
| Raymond (Formal) | 0.95 (very strict) | Quality only | Never degrade |

**Business Value**:
- Brand-specific tuning without code changes
- A/B test policies per retailer
- White-label customization

**Implementation**: Policy module reads `RetailerPolicy` from cloud config.

---

## Go-to-Market Strategy

### Phase 1: Proof of Concept (3 months)

**Goal**: Prove kurta try-on works better than competitors.

**Deliverables**:
1. SDK demo (iOS + Android)
2. 10 kurta garments preprocessed
3. Analytics dashboard (basic)
4. Pilot with 1 retailer (Manyavar or FabIndia)

**Success Metric**: 
- P0 violation rate < 0.1%
- Conversion lift > 15%
- Return rate drop > 10%

---

### Phase 2: Vertical Expansion (6 months)

**Goal**: Dominate kurta market, expand to related categories.

**Expansion**:
- Kurtas → Sherwanis (wedding wear)
- Kurtas → Nehru jackets (formal ethnic)
- Kurtas → Kurtis (women's ethnic)

**Target**: 5 retailers, 500 garments

---

### Phase 3: Horizontal Expansion (12 months)

**Goal**: Add second vertical (formal shirts or t-shirts).

**Leverage**: Proven platform, retailer relationships, analytics.

---

## Pricing Model

### Tier 1: SDK License

- **$500/month** per retailer
- Includes: SDK access, cloud preprocessing (100 garments/month), basic analytics

### Tier 2: Enterprise

- **$2,000/month** per retailer
- Includes: Unlimited garments, custom policy tuning, priority support, white-label

### Tier 3: Revenue Share

- **3% of AR-attributed sales**
- For large retailers (Myntra, Ajio, Flipkart Fashion)

---

## Technical Stack (SaaS)

### Perception Core (Native SDK)

- **Language**: C++ (core) + Rust (safety-critical modules)
- **ML Runtime**: ONNX Runtime (CPU), TensorRT (GPU)
- **CV**: OpenCV, Eigen
- **Platform**: iOS (Metal), Android (Vulkan)

### Cloud Infrastructure

- **Garment Preprocessing**: Python (FastAPI), ONNX, OpenCV
- **Storage**: S3 (garment assets), PostgreSQL (metadata)
- **CDN**: CloudFront (fast garment download)
- **Analytics**: ClickHouse (time-series metrics)

### Retailer Dashboard

- **Frontend**: React + TypeScript
- **Backend**: Node.js (Express)
- **Auth**: Auth0 (retailer login)

---

## Competitive Moat

### What Makes AR Mirror Different

1. **India-first** (ethnic wear focus, not Western wear)
2. **Trust metrics** (not just engagement metrics)
3. **Retailer SaaS** (not consumer app)
4. **Perception platform** (not just AR overlay)

### Defensibility

- **Data moat**: Retailer-specific failure patterns
- **Integration moat**: Embedded in retailer apps (high switching cost)
- **IP moat**: Perception architecture (authority-based modules)

---

## Next Steps (Immediate)

### Week 1-2: Vertical Decision

**Action**: Choose kurtas, formal shirts, or t-shirts.

**Deliverable**: Single-page vertical strategy doc.

---

### Week 3-4: SDK Prototype

**Action**: Build minimal SDK (iOS or Android).

**Deliverable**: Demo app with 3 kurtas.

---

### Week 5-8: Retailer Pilot

**Action**: Approach Manyavar/FabIndia with pilot proposal.

**Deliverable**: Signed pilot agreement.

---

## Summary

**Product**: Perception SDK for retailers (not consumer AR app)  
**Vertical**: Kurtas first (India-specific advantage)  
**SaaS**: SDK + Cloud + Analytics + Policy Tuning  
**Moat**: India-first, trust metrics, retailer integration  

**Critical Decision**: Which vertical to dominate first?

**Recommendation**: **Kurtas** (cultural advantage, loose fit works well, less competition)

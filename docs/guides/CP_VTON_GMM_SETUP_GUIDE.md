# CP-VTON GMM (TPS Warper) - Known-Good Setup Guide

**Status**: Pre-Integration Checklist  
**Date**: January 17, 2026  
**Critical**: Do NOT integrate into app.py until standalone test passes

---

## PART 1: Canonical Implementation (DO NOT IMPROVISE)

### Reference Repository
```
https://github.com/sergeywong/cp-vton
```

**Why this repo:**
- Author-aligned implementation
- Correct GMM (TPS warper) preprocessing
- Expected tensor formats documented
- Known-good checkpoint compatibility

**⚠️ DO NOT mix repos. Many forks subtly break GMM.**

---

## PART 2: Required Pretrained Checkpoints

### GMM (Geometric Matching Module) - REQUIRED
```
File: gmm_final.pth
Size: ~120 MB
Purpose: TPS warper (cloth deformation)
```

### TOM (Try-On Module) - OPTIONAL (Phase 2)
```
File: tom_final.pth
Size: ~380 MB
Purpose: Appearance synthesis (later)
```

### Download Links (Manual - Cannot Automate)

**From CP-VTON repo README:**

GMM checkpoint (verify link in original repo):
```
https://drive.google.com/file/d/1h6v4IYhQ4Z5z9nWqYyR3k3r6y8XG0u3v
```

TOM checkpoint (verify link in original repo):
```
https://drive.google.com/file/d/1V2E7Z9b8vP3E6ZJ5wqQ0zZy6rU5KxvX_
```

**⚠️ UNCERTAINTY NOTICE:**
Google Drive IDs may have changed. If links are dead:
1. Check original CP-VTON README
2. Check repo Issues section for mirrors
3. Look for "pretrained models" or "checkpoints" in releases

---

## PART 3: Correct Directory Layout (NON-NEGOTIABLE)

Create this structure in your project:

```
C:\Users\HP\Projects\AR Mirror\
├── cp-vton/                    ← Clone from GitHub
│   ├── checkpoints/
│   │   ├── gmm_final.pth       ← Place downloaded checkpoint here
│   │   └── tom_final.pth       ← (Optional for now)
│   ├── models/
│   │   ├── networks.py         ← From repo
│   │   ├── gmm.py              ← GMM architecture
│   │   └── tom.py              ← TOM architecture
│   ├── data/
│   │   └── dataset.py          ← VITON dataset loader
│   ├── utils/
│   │   └── utils.py            ← Preprocessing helpers
│   ├── test.py                 ← Standalone test script
│   └── README.md
├── dataset/                    ← Your existing VITON data
│   └── train/
│       ├── cloth/
│       ├── cloth-mask/
│       ├── image/
│       └── openpose_json/      ← Keypoints (needs conversion)
└── app.py                      ← Your current app (DO NOT MODIFY YET)
```

---

## PART 4: GMM Input Requirements (EXACT TENSOR FORMATS)

### Critical Correction: OpenPose JSON ≠ Pose Map

**You said:** "I already have OpenPose JSON"

**Reality:** GMM does NOT consume JSON keypoints.

**GMM expects:** 18-channel pose heatmaps (Gaussian blobs per joint)

### Required Inputs (Per Sample)

| Tensor Name | Shape         | Type    | Meaning                    |
|-------------|---------------|---------|----------------------------|
| cloth       | (3, 256, 192) | float32 | RGB garment [0,1]          |
| cloth_mask  | (1, 256, 192) | float32 | Binary {0,1}               |
| pose_map    | (18, 256, 192)| float32 | OpenPose heatmaps (18 joints) |
| agnostic    | (3, 256, 192) | float32 | Person without clothes     |

**If ANY of these are wrong → warping fails silently or produces garbage.**

### Pose Map Generation (MANDATORY STEP)

Convert OpenPose JSON to 18-channel heatmaps:

```python
# CP-VTON includes this code in data/dataset.py
def get_pose_map(pose_keypoints, height=256, width=192):
    """
    Args:
        pose_keypoints: OpenPose JSON keypoints (18 joints × 3 values)
        height: Output height (256)
        width: Output width (192)
    
    Returns:
        pose_map: (18, 256, 192) tensor with Gaussian blobs at joint locations
    """
    # Each joint gets a Gaussian heatmap
    # Sigma typically 3-5 pixels
    # Background joints = zero channel
```

**Use CP-VTON's existing implementation. Do not rewrite.**

---

## PART 5: Standalone Test (BEFORE INTEGRATION)

### Step 1: Clone CP-VTON Repository
```powershell
cd "C:\Users\HP\Projects\AR Mirror"
git clone https://github.com/sergeywong/cp-vton.git
cd cp-vton
```

### Step 2: Install Dependencies
```powershell
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install scipy
pip install tqdm
```

### Step 3: Place Checkpoint
```powershell
# Create checkpoints directory
mkdir checkpoints

# Move downloaded gmm_final.pth to:
# cp-vton/checkpoints/gmm_final.pth
```

### Step 4: Run GMM Standalone Test
```powershell
# From cp-vton directory
python test.py --name gmm --stage GMM --dataset viton --dataroot ../dataset/train
```

**Expected output location:**
```
cp-vton/results/gmm/
├── warped_cloth/     ← Geometrically matched garments
└── warped_mask/      ← Warped masks
```

---

## PART 6: Visual Verification (CRITICAL CHECKPOINTS)

### ✅ GMM is Working If You See:

1. **Cloth narrows at waist**
   - Not a rigid rectangle
   - Visible hourglass shape

2. **Shoulders align with body**
   - Even when person tilts
   - Shoulder seams match anatomically

3. **Sleeves rotate with arms**
   - Arms up → sleeves follow
   - Arms down → sleeves hang naturally

4. **No rigid transformations**
   - Not just scaled/rotated
   - Visible non-rigid deformation

### ❌ GMM is NOT Working If You See:

1. **Still a rectangle**
   - No shape change
   - Just scaled/positioned
   - → Bypassing TPS entirely

2. **Warped but garbage**
   - Extreme distortion
   - Cloth torn/shredded
   - → Wrong normalization or tensor format

3. **No output files**
   - Empty results folder
   - → Checkpoint not loading or path wrong

---

## PART 7: Integration Plan (DO NOT EXECUTE YET)

### Phase 1: Extract GMM Warper (After Standalone Test Passes)

Create a service module:

```python
# File: src/warp/gmm_warper.py

import torch
from cp-vton.models.gmm import GMM

class CPVTONWarper:
    def __init__(self, checkpoint_path):
        self.gmm = GMM()
        self.gmm.load_state_dict(torch.load(checkpoint_path))
        self.gmm.eval()
        self.gmm.cuda()  # or .cpu() if no GPU
    
    def warp(self, cloth, cloth_mask, pose_map):
        """
        Args:
            cloth: (3, 256, 192) float32 [0,1]
            cloth_mask: (1, 256, 192) float32 {0,1}
            pose_map: (18, 256, 192) float32 heatmaps
        
        Returns:
            warped_cloth: (3, 256, 192) deformed garment
            warped_mask: (1, 256, 192) deformed mask
        """
        with torch.no_grad():
            warped_cloth, warped_mask = self.gmm(cloth, cloth_mask, pose_map)
        
        return warped_cloth, warped_mask
```

### Phase 2: Replace Resize in app.py (ONLY After Phase 1 Works)

```python
# OLD (current app.py):
cloth_resized = cv2.resize(cloth_rgb, (roi_w, roi_h))

# NEW (after GMM verified):
warped_cloth = gmm_warper.warp(cloth, cloth_mask, pose_map)
```

### Phase 3: Visualization Test (No Blending Yet)

```python
# Just display warped_cloth alone
cv2.imshow("Warped Cloth", warped_cloth)
cv2.imshow("Warped Mask", warped_mask)

# NO alpha compositing yet
# NO background blending yet
# JUST INSPECTION
```

---

## PART 8: Known Failure Modes (Brittleness Points)

### 1. Tensor Shape Mismatch
```
Error: Expected (3, 256, 192), got (192, 256, 3)
Fix: Transpose and normalize correctly
```

### 2. Wrong Normalization
```
Symptom: Warped output is all white/black
Cause: Input not in [0,1] or [-1,1] depending on model
Fix: Check CP-VTON preprocessing exactly
```

### 3. Pose Map Format Wrong
```
Symptom: Cloth doesn't deform, stays rectangular
Cause: Pose keypoints not converted to heatmaps
Fix: Use CP-VTON's get_pose_map() function
```

### 4. Checkpoint Version Mismatch
```
Error: KeyError or shape mismatch on load
Fix: Download EXACT checkpoint from CP-VTON repo, not forks
```

---

## PART 9: Hard Truth (Manage Expectations)

### Even Perfect GMM Output Will Look Fake When Overlaid

**Why:**
- GMM only handles **geometry** (shape deformation)
- GMM does **NOT** handle **appearance** (lighting, shadows, texture)
- Alpha blending on top of existing clothes = always fake

### Locked Roadmap (Cannot Skip Steps):

```
Step 1: GMM (TPS)           → Geometry correct ✓
Step 2: Human Parsing       → Remove old clothes (REQUIRED)
Step 3: TOM / Generator     → Realism (REQUIRED)
```

You are currently at Step 1 preparation.

Steps 2 and 3 are **mandatory** for "perfection."

---

## PART 10: Next Actions (Precise Order)

### DO NOW:
1. ☐ Clone CP-VTON repository
2. ☐ Download `gmm_final.pth` checkpoint (manual)
3. ☐ Place in `cp-vton/checkpoints/`
4. ☐ Run `python test.py --stage GMM`
5. ☐ Inspect warped_cloth output visually
6. ☐ Report back: "warped output looks correct / incorrect / rectangular"

### DO NOT DO YET:
- ❌ Modify app.py
- ❌ Integrate GMM into live demo
- ❌ Worry about FPS
- ❌ Add UI controls
- ❌ Optimize performance

### DECISION POINT:

Your next message should be ONE of:
- "GMM standalone output looks correct" → Proceed to integration
- "GMM output still rectangular" → Debug pose map generation
- "Checkpoint doesn't load" → Verify file path and PyTorch version
- "Output is garbage/distorted" → Check tensor normalization
- "Need help with pose map conversion" → I'll provide exact code

---

## PART 11: Critical File Checklist

Before running test.py, verify these files exist:

```powershell
# Check CP-VTON structure
Test-Path "cp-vton/models/gmm.py"          # Should be True
Test-Path "cp-vton/checkpoints/gmm_final.pth"  # Should be True
Test-Path "cp-vton/test.py"                # Should be True
Test-Path "cp-vton/data/dataset.py"        # Should be True

# Check VITON dataset
Test-Path "dataset/train/cloth/"           # Should be True
Test-Path "dataset/train/cloth-mask/"      # Should be True
Test-Path "dataset/train/openpose_json/"   # Should be True
```

If ANY return False → fix before proceeding.

---

## PART 12: Troubleshooting Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Import error: No module 'models' | Running from wrong directory | cd into cp-vton/ first |
| FileNotFoundError: gmm_final.pth | Checkpoint not placed | Check cp-vton/checkpoints/ |
| Tensor size mismatch | Wrong input shape | Use (3,256,192) not (192,256,3) |
| Output is all zeros | Pose map not generated | Convert OpenPose JSON to heatmaps |
| CUDA out of memory | Batch size too large | Reduce or use .cpu() |
| Warped cloth looks same as input | TPS not applied | Check model loaded correctly |

---

**Status**: Guide complete. Awaiting standalone test results before integration.

**No shortcuts from here onward.**

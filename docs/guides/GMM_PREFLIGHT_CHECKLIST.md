# CP-VTON GMM Integration - Pre-Flight Checklist

**Date**: January 17, 2026  
**Status**: ✅ Ready for standalone test

---

## ✅ COMPLETED PREPARATION

### 1. VITON Dataset Structure - VERIFIED
```
✓ dataset/train/cloth/ exists (11,648 garment images)
✓ dataset/train/cloth-mask/ exists (binary segmentation masks)
✓ dataset/train/openpose_json/ exists (keypoint annotations)
```

### 2. Pose Map Conversion - WORKING
```
✓ convert_pose_map.py created
✓ Tested with 00000_00_keypoints.json
✓ Output: (18, 256, 192) float32 [0,1] ✓
✓ 13/18 keypoints visible
✓ Visualization: pose_map_test.jpg generated
```

### 3. VITON Loader - PRODUCTION READY
```
✓ load_viton_cloth() integrated
✓ Proper cloth + cloth-mask alpha compositing
✓ Float32 [0,1] normalization
✓ All assertions passing
✓ Performance: 16.6 FPS (baseline only, no GMM yet)
```

---

## ⏳ PENDING MANUAL STEPS (DO NOW)

### Step 1: Clone CP-VTON Repository
```powershell
cd "C:\Users\HP\Projects\AR Mirror"
git clone https://github.com/sergeywong/cp-vton.git
```

**Expected result:**
```
cp-vton/
├── models/
│   ├── gmm.py
│   ├── tom.py
│   └── networks.py
├── test.py
├── data/
│   └── dataset.py
└── README.md
```

### Step 2: Download GMM Checkpoint (Manual - Google Drive)

**From CP-VTON repo README, look for:**
- gmm_final.pth (~120 MB)
- Or similar named checkpoint for GMM/TPS warper

**⚠️ VERIFY LINK in original repo (may have changed):**
```
https://drive.google.com/file/d/1h6v4IYhQ4Z5z9nWqYyR3k3r6y8XG0u3v
```

**If link dead:**
1. Check CP-VTON GitHub Issues for mirrors
2. Check README for updated links
3. Search for "pretrained models" in releases

### Step 3: Place Checkpoint
```powershell
# Create checkpoints directory in cp-vton
mkdir "C:\Users\HP\Projects\AR Mirror\cp-vton\checkpoints"

# Move downloaded file
Move-Item gmm_final.pth "C:\Users\HP\Projects\AR Mirror\cp-vton\checkpoints\"

# Verify
Test-Path "C:\Users\HP\Projects\AR Mirror\cp-vton\checkpoints\gmm_final.pth"
# Should return: True
```

### Step 4: Install CP-VTON Dependencies
```powershell
# If not already installed
pip install torch torchvision
pip install scipy
pip install tqdm
```

---

## 🧪 STANDALONE TEST (CRITICAL CHECKPOINT)

### Run CP-VTON GMM Test
```powershell
cd "C:\Users\HP\Projects\AR Mirror\cp-vton"

python test.py --name gmm --stage GMM --dataset viton --dataroot ../dataset/train
```

### Expected Output
```
cp-vton/results/gmm/
├── warped_cloth/
│   ├── 00000_00.jpg     ← Deformed garment
│   ├── 00001_00.jpg
│   └── ...
└── warped_mask/
    ├── 00000_00.jpg     ← Deformed mask
    ├── 00001_00.jpg
    └── ...
```

---

## ✅ VISUAL VERIFICATION (PASS/FAIL CRITERIA)

Open `cp-vton/results/gmm/warped_cloth/00000_00.jpg`

### ✅ PASS - GMM is Working If You See:

1. **Cloth shape changed (not rectangular)**
   - Visible hourglass shape
   - Waist narrowing visible
   - Non-rigid deformation

2. **Shoulders aligned anatomically**
   - Match body pose
   - Seams positioned correctly

3. **Sleeves follow arm direction**
   - Bent if arms bent
   - Straight if arms straight

4. **Smooth deformation**
   - No torn/shredded regions
   - Gradual warping

### ❌ FAIL - Debugging Required If You See:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Still rectangular | TPS not applied | Check checkpoint loaded |
| Extreme distortion | Wrong normalization | Verify tensor format |
| All black/white | Input preprocessing wrong | Check cloth loading |
| No output files | Script failed silently | Check error logs |

---

## 🚫 DO NOT PROCEED TO INTEGRATION UNTIL:

- [ ] CP-VTON standalone test produces warped_cloth output
- [ ] Visual inspection shows NON-RIGID deformation
- [ ] Warped cloth is NOT just scaled/rotated
- [ ] Multiple samples (00000, 00001, etc.) all show warping

**If standalone test fails → stop. Do NOT modify app.py yet.**

---

## 📊 CURRENT STATUS SUMMARY

### What's Working Now (Baseline)
```
✓ VITON dataset: 11,648 pairs loaded
✓ Proper alpha compositing: cloth * mask + bg * (1 - mask)
✓ OpenPose JSON to 18-channel heatmap conversion
✓ Performance: 16.6 FPS (no neural networks yet)
```

### What's Still Missing (Why It Looks Fake)
```
❌ Cloth deformation (GMM/TPS) - rectangular overlay only
❌ Human parsing (existing clothes visible)
❌ Appearance synthesis (alpha blending artifacts)
```

### Roadmap After GMM Integration
```
Phase 1 (Current): GMM standalone test
Phase 2 (Next):    Integrate GMM into app.py (geometry only)
Phase 3 (Later):   Add human parsing (remove old clothes)
Phase 4 (Final):   Add TOM/generator (photorealism)
```

---

## 📝 YOUR NEXT MESSAGE SHOULD BE:

One of these (nothing else):

1. **"GMM standalone output looks correct"**
   → I'll proceed to app.py integration

2. **"GMM output still rectangular"**
   → I'll debug TPS warping

3. **"Checkpoint doesn't load / path errors"**
   → I'll help troubleshoot file structure

4. **"Output is distorted garbage"**
   → I'll check tensor normalization

5. **"Need pose map visualization"**
   → Open `pose_map_test.jpg` to verify skeleton

6. **"CP-VTON repo missing files"**
   → I'll provide alternative repo links

---

## 🛠️ TROUBLESHOOTING COMMANDS

### Verify File Structure
```powershell
# Check CP-VTON files
Test-Path "cp-vton/models/gmm.py"
Test-Path "cp-vton/checkpoints/gmm_final.pth"
Test-Path "cp-vton/test.py"

# Check dataset
Test-Path "dataset/train/cloth/00000_00.jpg"
Test-Path "dataset/train/cloth-mask/00000_00.jpg"
Test-Path "dataset/train/openpose_json/00000_00_keypoints.json"
```

### Check PyTorch Installation
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Visualize Pose Map
```powershell
python convert_pose_map.py
# Then open: pose_map_test.jpg
```

---

**STATUS**: Awaiting standalone test results.

**NO app.py modifications until GMM verified working.**

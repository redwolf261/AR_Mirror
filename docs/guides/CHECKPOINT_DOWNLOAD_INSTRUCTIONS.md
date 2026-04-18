# CP-VTON Checkpoint Download Instructions

**Status**: Repository cloned ✅  
**Next Step**: Download pretrained checkpoints (manual)

---

## ✅ COMPLETED

```
✓ CP-VTON repository cloned
✓ Directory structure verified
✓ GMM architecture loadable
✓ Checkpoint directories created:
  - cp-vton/checkpoints/
  - cp-vton/checkpoints/gmm_train_new/
```

---

## ⚠️ REQUIRED: Manual Checkpoint Download

### Option 1: Official CP-VTON Checkpoints (Recommended)

**Problem**: The original CP-VTON repo (sergeywong/cp-vton) does NOT include pretrained checkpoints in the repository.

**Solution**: Search for pretrained CP-VTON checkpoints from these sources:

#### A. GitHub Issues/Discussions
```
https://github.com/sergeywong/cp-vton/issues
```
Look for:
- "pretrained model"
- "checkpoint"
- "gmm_final.pth"
- "trained weights"

#### B. Alternative Repos with Checkpoints

Many forks include pretrained weights:

1. **CP-VTON+ (improved version)**
   ```
   https://github.com/minar09/cp-vton-plus
   ```
   May include checkpoints in releases

2. **Search GitHub for "cp-vton checkpoint"**
   ```
   https://github.com/search?q=cp-vton+checkpoint
   ```

3. **Papers With Code**
   ```
   https://paperswithcode.com/paper/toward-characteristic-preserving-image-based
   ```
   Check "Code" section for implementations with weights

---

### Option 2: Use HR-VITON Checkpoints (Different Model)

**Note**: HR-VITON is a different architecture but similar pipeline.

From your earlier terminal output, you have these links:

```
GMM: https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ
TOM: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy
```

**⚠️ WARNING**: These are HR-VITON weights, NOT CP-VTON.
- Different architecture
- May not be directly compatible
- Tensor shapes might differ

---

### Option 3: Train From Scratch (Last Resort)

If no pretrained checkpoints available:

```powershell
cd "C:\Users\HP\Projects\AR Mirror\cp-vton"
python train.py --name gmm_train_new --stage GMM --workers 4 --save_count 5000 --shuffle
```

**Time estimate**: 2-3 days on RTX 2050  
**Dataset**: Your 11,648 VITON pairs are sufficient

---

## 🎯 RECOMMENDED ACTION PLAN

### Step 1: Search for CP-VTON Pretrained Weights

Try each source in order:

1. **GitHub Issues** (sergeywong/cp-vton)
   - Look for users sharing trained models
   - Check closed issues too

2. **CP-VTON+ repo**
   - More maintained fork
   - May have releases with weights

3. **Google/Academic Sources**
   - Authors may have shared weights separately
   - Check paper supplementary materials

### Step 2: Verify Checkpoint Compatibility

Once you find a checkpoint:

```powershell
# Check file size
Get-Item path/to/gmm_final.pth | Select-Object Name, Length

# Expected: ~120-150 MB for GMM
```

### Step 3: Place Checkpoint

```powershell
# Copy to correct location
Copy-Item gmm_final.pth "C:\Users\HP\Projects\AR Mirror\cp-vton\checkpoints\gmm_train_new\gmm_final.pth"

# Verify
Test-Path "C:\Users\HP\Projects\AR Mirror\cp-vton\checkpoints\gmm_train_new\gmm_final.pth"
# Should return: True
```

### Step 4: Test Checkpoint Loading

```powershell
cd "C:\Users\HP\Projects\AR Mirror\cp-vton"

python -c "import torch; from networks import GMM; gmm = GMM(4); state = torch.load('checkpoints/gmm_train_new/gmm_final.pth'); gmm.load_state_dict(state['model'] if 'model' in state else state); print('✓ Checkpoint loads successfully')"
```

---

## 🔍 ALTERNATIVE: Inspect Your Existing Models

You mentioned having `models/` directory. Let's check:

```powershell
Get-ChildItem "C:\Users\HP\Projects\AR Mirror\models" -Recurse -Include *.pth,*.pt,*.ckpt | Select-Object Name, Length, Directory
```

You may already have compatible weights from previous work.

---

## 📊 CHECKPOINT SPECIFICATIONS

### GMM (Geometric Matching Module)

**Expected format:**
```python
{
    'model': OrderedDict({...}),  # State dict
    'epoch': int,                  # Training epoch
    'optimizer': {...}             # Optional
}
```

**Key layers** (for verification):
- extraction: Feature extraction (VGG-based)
- regression: TPS regression network
- Up sampling layers (up1, up2, up3, up4)

**File size**: 120-150 MB

### TOM (Try-On Module) - Optional for Now

**File size**: 350-400 MB  
**Note**: Not needed for initial GMM test

---

## 🚦 DECISION POINT

Once checkpoint is located, your message should be:

1. **"Found checkpoint at [location]"**
   → I'll help verify compatibility

2. **"Cannot find pretrained checkpoint anywhere"**
   → I'll set up training pipeline

3. **"Using HR-VITON checkpoint instead"**
   → I'll adapt integration for HR-VITON architecture

4. **"Found checkpoint in models/ directory"**
   → I'll inspect and validate

---

## 🛑 BLOCKING ISSUE

**Cannot proceed to standalone test without:**
- gmm_final.pth checkpoint (or equivalent)
- Placed in: `cp-vton/checkpoints/gmm_train_new/`

**Next message from you should include:**
- Checkpoint source (where you found it)
- File size verification
- Or: "Cannot find checkpoint, need training setup"

---

**Status**: Waiting for checkpoint download/location confirmation.

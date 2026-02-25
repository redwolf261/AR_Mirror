# AR Mirror - Conda Setup Guide

Complete setup guide for AR Mirror with GPU acceleration using Anaconda/Miniconda.

---

## Prerequisites

### Install Anaconda or Miniconda

**Option 1: Anaconda (Full Package - 3GB)**
- Download: https://www.anaconda.com/download
- Includes Python, Jupyter, and 250+ data science packages
- Recommended for beginners

**Option 2: Miniconda (Minimal - 400MB)**
- Download: https://docs.conda.io/en/latest/miniconda.html
- Minimal installation, install only what you need
- Recommended for advanced users or limited disk space

---

## Quick Start (Recommended)

### Step 1: Install Anaconda/Miniconda

Run the installer and follow the prompts. Use default settings.

### Step 2: Open Anaconda PowerShell Prompt

Search for "Anaconda PowerShell Prompt" in Windows Start Menu and open it.

### Step 3: Navigate to Project

```powershell
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"
```

### Step 4: Create Environment from File

```powershell
# Create environment with all dependencies
conda env create -f environment.yml

# This will:
# - Create environment named 'ar_mirror'
# - Install Python 3.11
# - Install PyTorch with CUDA 12.1
# - Install OpenCV, NumPy, MediaPipe
# - Install all other dependencies
# 
# Takes 5-10 minutes depending on internet speed
```

### Step 5: Activate Environment

```powershell
conda activate ar_mirror
```

### Step 6: Verify GPU Support

```powershell
# Quick check
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Full verification
python scripts\verify_gpu.py

# Run validation
python phase2_validation.py
```

### Step 7: Run AR Mirror

```powershell
python app.py --phase 2
```

**Expected**: 200-300 FPS with GPU acceleration! 🚀

---

## Manual Setup (Alternative)

If you prefer manual control:

### Step 1: Create Empty Environment

```powershell
conda create -n ar_mirror python=3.11 -y
conda activate ar_mirror
```

### Step 2: Install PyTorch with CUDA

```powershell
# Install from PyTorch channel
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 3: Install Other Dependencies

```powershell
# Install via conda
conda install opencv numpy scipy pillow jupyter -c conda-forge -y

# Install via pip
pip install mediapipe>=0.10.30 kaggle>=1.5.16 gdown>=4.7.1
```

### Step 4: Verify Installation

```powershell
python scripts\verify_gpu.py
```

---

## Environment Management

### Activate Environment

**Always activate before using AR Mirror:**

```powershell
conda activate ar_mirror
```

### Deactivate Environment

```powershell
conda deactivate
```

### List All Environments

```powershell
conda env list
```

### Update Environment

```powershell
# Update from environment.yml
conda env update -f environment.yml --prune

# Update specific package
conda update pytorch -c pytorch
```

### Delete Environment

```powershell
# If you need to start fresh
conda env remove -n ar_mirror
```

### Export Environment

```powershell
# Save current environment state
conda env export > environment_backup.yml
```

---

## Common Issues & Solutions

### Issue: "conda: command not found"

**Solution**: Anaconda/Miniconda not installed or not in PATH

1. Reinstall Anaconda/Miniconda
2. Check "Add Anaconda to PATH" during installation
3. Or use "Anaconda PowerShell Prompt" instead of regular PowerShell

### Issue: CUDA not available after installation

**Solution**: Verify PyTorch CUDA installation

```powershell
conda activate ar_mirror
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If shows `False`, reinstall PyTorch:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --force-reinstall -y
```

### Issue: Environment creation fails

**Solution**: Update conda first

```powershell
conda update -n base conda -y
conda env create -f environment.yml
```

### Issue: Slow conda solver

**Solution**: Use mamba (faster solver)

```powershell
conda install mamba -n base -c conda-forge -y
mamba env create -f environment.yml
```

---

## GPU Verification Checklist

After setup, verify everything works:

**1. Check PyTorch Installation**
```powershell
conda activate ar_mirror
python -c "import torch; print('PyTorch:', torch.__version__)"
```
Expected: `PyTorch: 2.x.x+cu121`

**2. Check CUDA Availability**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Expected: `CUDA: True`

**3. Check GPU Name**
```powershell
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```
Expected: `GPU: NVIDIA GeForce RTX 2050`

**4. Run Full Verification**
```powershell
python scripts\verify_gpu.py
```
Expected: All tests pass, 10-15x speedup shown

**5. Run AR Mirror Validation**
```powershell
python phase2_validation.py
```
Expected: 7/7 tests pass, 200+ FPS benchmark

---

## VS Code Integration (Optional)

### Step 1: Install Python Extension

In VS Code, install "Python" extension by Microsoft.

### Step 2: Select Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `Python 3.11.x ('ar_mirror')`

### Step 3: Terminal Integration

VS Code will automatically activate the conda environment in the integrated terminal.

---

## Jupyter Notebook Setup (Optional)

The environment includes Jupyter for interactive development.

### Start Jupyter

```powershell
conda activate ar_mirror
jupyter notebook
```

### Register Kernel

```powershell
conda activate ar_mirror
python -m ipykernel install --user --name ar_mirror --display-name "AR Mirror (GPU)"
```

Now you can use "AR Mirror (GPU)" kernel in Jupyter notebooks.

---

## Performance Comparison

| Setup Method | Time | Difficulty | Recommended For |
|--------------|------|------------|-----------------|
| `environment.yml` | 10 min | Easy | Everyone (recommended) |
| Manual conda | 15 min | Medium | Advanced users |
| pip only | 5 min | Hard | Experts (dependency issues) |

---

## Next Steps After Setup

Once environment is created and activated:

### 1. Verify GPU
```powershell
python scripts\verify_gpu.py
```

### 2. Run Validation
```powershell
python phase2_validation.py
```

### 3. Test AR Mirror
```powershell
python app.py --phase 2
```

### 4. Check Performance
- Monitor FPS in window title
- Check GPU usage in Task Manager
- Expected: 200-300 FPS (vs 21 FPS CPU)

---

## Environment Specifications

**What's Included:**

- **Python 3.11**: Latest stable version with performance improvements
- **PyTorch 2.x + CUDA 12.1**: GPU acceleration for neural networks
- **OpenCV 4.8+**: Computer vision and video processing
- **MediaPipe 0.10+**: Real-time pose detection
- **NumPy, SciPy, Pillow**: Scientific computing and imaging
- **Jupyter**: Interactive development (optional)
- **Kaggle, gdown**: Dataset downloading utilities

**Total Size**: ~5-7 GB (including PyTorch + CUDA)

**GPU Requirements**: 
- NVIDIA GPU (RTX 2050 or better)
- 4GB+ VRAM recommended
- CUDA 12.1 compatible drivers

---

## Troubleshooting

### Get Help

```powershell
# Check conda info
conda info

# List packages in environment
conda activate ar_mirror
conda list

# Check PyTorch installation
conda list pytorch

# View environment details
conda env export
```

### Reset Everything

If something goes wrong, start fresh:

```powershell
# Remove environment
conda env remove -n ar_mirror

# Clear conda cache
conda clean --all -y

# Recreate environment
conda env create -f environment.yml
```

---

## Summary

**Recommended workflow:**

1. Install Anaconda/Miniconda
2. `conda env create -f environment.yml`
3. `conda activate ar_mirror`
4. `python scripts\verify_gpu.py`
5. `python app.py --phase 2`

**Result**: AR Mirror running at 200-300 FPS with full GPU acceleration! 🎉

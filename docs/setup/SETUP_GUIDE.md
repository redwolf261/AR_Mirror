# AR Mirror - Complete Setup Guide

Choose the best installation method for your needs.

---

## 🚀 **RECOMMENDED: Conda Setup** (Easiest)

### Why Conda?
- ✅ One command installs everything
- ✅ Automatic dependency resolution
- ✅ Isolated environment (won't break other projects)
- ✅ Easy to update/remove
- ✅ Industry standard for ML projects

### Quick Start

**Step 1**: Install Anaconda
- Download: https://www.anaconda.com/download
- Run installer, use default settings

**Step 2**: Run automated setup
```powershell
# Open: Anaconda PowerShell Prompt
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"
.\setup_conda.ps1
```

**That's it!** The script will:
- Create conda environment
- Install Python 3.11
- Install PyTorch with CUDA 12.1
- Install all dependencies
- Verify GPU support

**Time**: 10-15 minutes (one-time setup)

---

## 📦 Alternative: Direct Python Install

### When to use?
- You already have Python 3.9-3.12 installed
- You don't want Anaconda
- You want minimal installation

### Quick Start

**Step 1**: Install Python (if needed)
- Download: https://www.python.org/downloads/
- **MUST CHECK**: "Add Python to PATH" during install

**Step 2**: Install dependencies
```powershell
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"

# Install PyTorch with CUDA
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
python -m pip install -r requirements.txt
```

**Time**: 5-10 minutes

---

## 📊 Comparison

| Method | Time | Ease | Isolation | Recommended For |
|--------|------|------|-----------|-----------------|
| **Conda** | 15 min | ⭐⭐⭐⭐⭐ | Yes | Everyone |
| **Python + pip** | 10 min | ⭐⭐⭐ | No | Advanced users |

---

## After Installation

**Verify GPU**:
```powershell
# Conda users:
conda activate ar_mirror

# All users:
python scripts\verify_gpu.py
```

**Run AR Mirror**:
```powershell
python app.py --phase 2
```

**Expected Result**: 200-300 FPS (10-15x boost from 21 FPS)

---

## Troubleshooting

### Conda not found
- Install Anaconda: https://www.anaconda.com/download
- Use "Anaconda PowerShell Prompt" not regular PowerShell

### Python not found
- Install Python: https://www.python.org/downloads/
- Check "Add Python to PATH" during installation
- Restart PowerShell after installation

### CUDA not available
- Check GPU visible in Task Manager
- Install/update NVIDIA drivers
- Restart computer
- System will fallback to CPU (still works, just slower)

---

## Files Reference

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment specification |
| `setup_conda.ps1` | Automated conda setup script |
| `setup_gpu_simple.py` | Pip-based GPU setup (alternative) |
| `requirements.txt` | Pip requirements (base dependencies) |
| `CONDA_SETUP.md` | Detailed conda guide |
| `PYTHON_INSTALL_REQUIRED.md` | Python installation guide |
| `GPU_QUICKSTART.md` | GPU optimization guide |

---

## Quick Commands

```powershell
# CONDA METHOD (recommended)
# ---------------------------
# 1. Install Anaconda
# 2. Open Anaconda PowerShell Prompt
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"
.\setup_conda.ps1
conda activate ar_mirror
python scripts\verify_gpu.py
python app.py --phase 2

# PIP METHOD (alternative)
# ---------------------------
# 1. Install Python 3.11
# 2. Open PowerShell
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts\verify_gpu.py
python app.py --phase 2
```

---

## Need Help?

1. **Read the guides**:
   - Conda: `CONDA_SETUP.md`
   - Python: `PYTHON_INSTALL_REQUIRED.md`
   - GPU: `GPU_QUICKSTART.md`

2. **Check prerequisites**:
   - Windows 10/11
   - NVIDIA GPU (RTX 2050 or better)
   - 8GB+ RAM
   - 10GB+ free disk space

3. **Common issues**:
   - GPU not detected → Update NVIDIA drivers
   - Conda not found → Install Anaconda first
   - Python not found → Check PATH or use Anaconda
   - Slow download → Be patient, PyTorch is ~3GB

---

**Recommendation**: Use Conda setup for best experience! 🚀

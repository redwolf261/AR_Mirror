# Python Installation Required

## Issue Detected

Your system shows Python in Windows App Execution Aliases (stub), but no actual Python installation was found.

The AR Mirror project requires Python 3.9+ to be properly installed.

---

## Solution: Install Python

### Step 1: Download Python

Go to: **https://www.python.org/downloads/**

Download: **Python 3.11.x** or **Python 3.12.x** (latest stable)

### Step 2: Install Python

**IMPORTANT SETTINGS during installation**:

1. ✅ **Check "Add Python to PATH"** (CRITICAL!)
2. ✅ **Check "Install for all users"** (recommended)
3. Click **"Customize installation"**
4. Enable all optional features
5. Choose installation directory (default is fine)
6. Complete installation

### Step 3: Verify Installation

Open a NEW PowerShell window and run:

```powershell
python --version
# Should show: Python 3.11.x or 3.12.x

python -m pip --version
# Should show pip version
```

### Step 4: Install Project Dependencies

```powershell
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"

# Install base dependencies
python -m pip install -r requirements.txt

# Install CUDA PyTorch (for GPU acceleration)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Verify GPU Support

```powershell
# Run GPU verification
python scripts\verify_gpu.py

# Run full validation
python phase2_validation.py
```

---

##Alternative: Use Anaconda (Recommended for ML projects)

Anaconda includes Python + many scientific packages pre-installed.

### Step 1: Download Anaconda

Go to: **https://www.anaconda.com/download**

Download: **Anaconda for Windows (64-bit)**

### Step 2: Install Anaconda

Run the installer with default settings.

### Step 3: Create Environment

```powershell
# Open Anaconda PowerShell Prompt

# Create environment for AR Mirror
conda create -n ar_mirror python=3.11 -y

# Activate environment
conda activate ar_mirror

# Navigate to project
cd "c:\Users\shind\OneDrive\Dokumen\GitHub\AR_Mirror"

# Install dependencies
pip install -r requirements.txt

# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Always Activate Environment

```powershell
# Before running AR Mirror, always activate:
conda activate ar_mirror
```

---

## After Python is Installed

Once Python is properly installed, run:

```powershell
# Quick setup (installs CUDA PyTorch)
python setup_gpu_simple.py

# Verify GPU
python scripts\verify_gpu.py

# Run AR Mirror
python app.py --phase 2
```

---

## Disable Windows App Aliases (Optional)

To prevent confusion with the Python stubs:

1. Press `Win + I` (Settings)
2. Go to: **Apps** > **Advanced app settings** > **App execution aliases**
3. Turn OFF:
   - `python.exe`
   - `python3.exe`

This ensures the real Python installation is used.

---

## Quick Check: Is Python Already Installed?

Try these locations:

```powershell
# Check if Python exists elsewhere
Get-ChildItem "C:\Program Files" -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object FullName

Get-ChildItem "C:\Users\shind\AppData\Local\Programs" -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object FullName

# Check Anaconda
Get-Command conda -ErrorAction SilentlyContinue
```

If any of these show results, you may already have Python installed somewhere.

---

## Need Help?

After installing Python, run:

```powershell
python --version
python -m pip --version
```

If both work, you're ready to continue with:

```powershell
python setup_gpu_simple.py
```

# Fix NVIDIA Code 45 - "Device Not Connected"

## Problem
Your HP Victus has **NVIDIA GeForce RTX 2050** but Windows shows "Code 45: Device not connected"
This is due to **NVIDIA Optimus** power management.

## Solutions (Try in order)

### Solution 1: Restart and Rescan Hardware
```powershell
# Run PowerShell as Administrator
Restart-Computer -Force
# After restart:
pnputil /scan-devices
```

### Solution 2: Update/Reinstall NVIDIA Drivers

**Uninstall current driver:**
1. Press `Win + X` → Apps and Features
2. Search "NVIDIA" → Uninstall all NVIDIA software
3. Restart computer

**Install latest driver:**
1. Go to: https://www.nvidia.com/download/index.aspx
2. Select:
   - Product Type: GeForce
   - Product Series: GeForce RTX 20 Series (Notebooks)
   - Product: GeForce RTX 2050
   - Operating System: Windows 11
   - Download Type: Game Ready Driver
3. Download and install (choose Custom → Clean Install)
4. Restart computer

### Solution 3: Enable in BIOS
1. Restart → Press `F10` (or `Esc` then `F10`) during boot to enter BIOS
2. Navigate to: **Advanced** → **Device Configuration**
3. Find: **Switchable Graphics** or **Discrete Graphics**
4. Set to: **Enabled** or **Discrete**
5. Save and Exit (`F10`)

### Solution 4: Windows Power Settings
1. Open NVIDIA Control Panel (right-click desktop → NVIDIA Control Panel)
2. If it doesn't open, the driver isn't working
3. Go to: **Manage 3D Settings** → **Global Settings**
4. Preferred graphics processor: **High-performance NVIDIA processor**
5. Apply

### Solution 5: Force GPU Detection
Run PowerShell as Administrator:
```powershell
# Uninstall the device
$gpu = Get-PnpDevice -FriendlyName "*NVIDIA GeForce*"
pnputil /remove-device $gpu.InstanceId

# Scan for hardware changes
pnputil /scan-devices

# Restart
Restart-Computer -Force
```

### Solution 6: Check Windows Graphics Settings
1. Settings → System → Display → Graphics settings
2. Click "Browse" → Add Python: `C:\Users\HP\Projects\AR Mirror\.venv\Scripts\python.exe`
3. Click Options → Select **High performance**
4. Save

### Solution 7: Disable Optimus (Advanced)
If you want the NVIDIA GPU always active:
1. BIOS → Advanced → Switchable Graphics → **Disable** (forces discrete GPU only)
2. Note: This reduces battery life significantly

## Verify Fix
After trying solutions, run:
```powershell
python phase2_validation.py
```

Expected output:
```
✓ CUDA Available: True
✓ Using CUDA: NVIDIA GeForce RTX 2050
```

## Alternative: Use CPU (Already Working)
Your system is already optimized for CPU and runs at **21.6 FPS**. The GPU will give 10-20x speedup but isn't required for development. When you deploy to a cloud server with GPU, it will automatically use it.

## Quick Test
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

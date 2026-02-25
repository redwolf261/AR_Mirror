# GPU BIOS Setup Guide - HP Victus Laptop

## 🎯 Goal
Enable your **NVIDIA GeForce RTX 2050** discrete GPU in BIOS to unlock 10-15x performance boost (21 FPS → 200-300 FPS)

---

## 📊 Visual Guides

### BIOS Navigation Flow
![BIOS Navigation Steps](file:///C:/Users/HP/.gemini/antigravity/brain/b42e385c-c864-499c-9f17-dc1923bc1687/bios_navigation_guide_1769791379523.png)

### Keyboard Shortcuts
![BIOS Keyboard Guide](file:///C:/Users/HP/.gemini/antigravity/brain/b42e385c-c864-499c-9f17-dc1923bc1687/bios_keyboard_guide_1769791397963.png)

---

## ⚠️ Before You Start

> [!IMPORTANT]
> **Save all your work** - You will need to restart your computer multiple times

> [!TIP]
> **Take photos with your phone** - If you get stuck in BIOS, having photos helps troubleshoot

**What you'll need:**
- 10-15 minutes of uninterrupted time
- Your laptop plugged into power (not on battery)
- This guide open on your phone or printed

---

## 📋 Step-by-Step Instructions

### Step 1: Save Your Work
1. **Close all applications** (especially any running Python scripts)
2. **Save all open files**
3. **Close Visual Studio Code** or any IDEs
4. Make sure nothing important is running

---

### Step 2: Restart and Enter BIOS

#### Option A: Restart from Windows
1. Click **Start Menu** → **Power** → **Restart**
2. **Immediately** when the HP logo appears, start **tapping F10 repeatedly**
   - Don't hold it down, tap it every half second
   - Keep tapping until you see the BIOS screen

#### Option B: If F10 doesn't work
1. Restart computer
2. When HP logo appears, tap **Esc** repeatedly
3. You'll see a startup menu
4. Press **F10** for "BIOS Setup"

> [!TIP]
> **Timing is key!** Start tapping F10 as soon as you see the HP logo. If you miss it, just restart and try again.

---

### Step 3: Navigate BIOS (Use Arrow Keys)

Once in BIOS, you'll see a blue/gray screen with menus at the top.

**BIOS Navigation:**
- **Arrow Keys** (←↑→↓) - Move between options
- **Enter** - Select/Open menu
- **F10** - Save and Exit
- **Esc** - Go back/Cancel

---

### Step 4: Find Graphics Settings

Your HP Victus BIOS may have one of these layouts:

#### Layout 1: Advanced Tab
```
Main | Advanced | Security | Boot | Exit
      ^^^^^^^^
```

1. Use **Right Arrow** to move to **Advanced** tab
2. Press **Down Arrow** to highlight **Device Configuration**
3. Press **Enter**

#### Layout 2: System Configuration
```
Main | System Configuration | Security | Boot | Exit
      ^^^^^^^^^^^^^^^^^^^^^
```

1. Use **Right Arrow** to move to **System Configuration**
2. Look for **Graphics Options** or **Switchable Graphics**
3. Press **Enter**

---

### Step 5: Enable Discrete Graphics

You're looking for one of these options:

#### Option 1: "Switchable Graphics"
```
Switchable Graphics: [Enabled]  ← Should be here
                     [Disabled]
```
- Highlight **Switchable Graphics**
- Press **Enter**
- Select **Enabled**
- Press **Enter** to confirm

#### Option 2: "Discrete Graphics"
```
Discrete Graphics: [Enabled]  ← Should be here
                   [Disabled]
```
- Same process as above

#### Option 3: "Graphics Mode"
```
Graphics Mode: [Hybrid]      ← Select this
               [Discrete]    ← Or this (GPU always on, less battery)
               [Integrated]  ← Don't select this
```
- Select **Hybrid** (recommended) or **Discrete** (max performance)

> [!WARNING]
> **Do NOT select "Integrated Only"** - This disables the NVIDIA GPU entirely

---

### Step 6: Save and Exit

1. Press **F10** (Save Changes and Exit)
2. You'll see a confirmation dialog:
   ```
   Save configuration changes and exit now?
   [Yes]  [No]
   ```
3. Make sure **Yes** is highlighted
4. Press **Enter**
5. Computer will restart automatically

---

### Step 7: Verify GPU is Enabled

After Windows boots up:

#### Quick Check (30 seconds)
1. Open **PowerShell** (Win + X → Windows PowerShell)
2. Run:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```
3. **Expected result**: `CUDA Available: True`

#### Detailed Check (1 minute)
1. Run the verification script:
   ```powershell
   cd "C:\Users\HP\Projects\AR Mirror"
   python verify_gpu_setup.py
   ```
2. Look for:
   ```
   ✓ CUDA Available: True
   ✓ GPU: NVIDIA GeForce RTX 2050
   ✓ Memory: 4.0 GB
   ```

---

## 🔧 Troubleshooting

### Problem 1: Can't Enter BIOS (F10 not working)

**Solution A: Try different key combinations**
- HP Victus models vary - try: **Esc**, then **F10**
- Some models: **F2** or **F9**

**Solution B: Use Windows Advanced Startup**
1. Hold **Shift** while clicking **Restart**
2. Choose **Troubleshoot** → **Advanced Options** → **UEFI Firmware Settings**
3. Click **Restart**

---

### Problem 2: Can't Find Graphics Settings

**Your BIOS might use different names:**
- Look for: "Device Configuration", "System Configuration", "Advanced Options"
- Search for keywords: "Graphics", "Video", "Display", "GPU", "Switchable"

**If you still can't find it:**
1. Take a photo of each BIOS tab
2. Press **Esc** to exit without saving
3. Share the photos - I can help identify the exact location

---

### Problem 3: CUDA Still Shows "False" After Enabling

**Step 1: Check Device Manager**
1. Press **Win + X** → **Device Manager**
2. Expand **Display adapters**
3. You should see:
   - Intel(R) Iris(R) Xe Graphics
   - NVIDIA GeForce RTX 2050

**If NVIDIA has a yellow warning:**
- Right-click → **Update driver**
- Choose "Search automatically for drivers"

**Step 2: Run the fix script**
```powershell
cd "C:\Users\HP\Projects\AR Mirror"
powershell -ExecutionPolicy Bypass -File fix_nvidia_gpu.ps1
```

**Step 3: Install/Update NVIDIA Drivers**
1. Go to: https://www.nvidia.com/download/index.aspx
2. Select:
   - Product Type: **GeForce**
   - Product Series: **GeForce RTX 20 Series (Notebooks)**
   - Product: **GeForce RTX 2050**
   - Operating System: **Windows 11**
3. Download and install (choose **Custom** → **Clean Install**)
4. Restart computer

---

### Problem 4: GPU Enabled but PyTorch Can't See It

**Install CUDA Toolkit:**
1. Download CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **11** (or your Windows version)
   - Installer Type: **exe (local)**
3. Run installer (takes 10-15 minutes)
4. Restart computer
5. Verify:
   ```powershell
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

---

## ✅ Success Checklist

After completing all steps, verify:

- [ ] Computer restarted successfully
- [ ] No BIOS errors on boot
- [ ] Windows boots normally
- [ ] Device Manager shows NVIDIA GPU (no yellow warnings)
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `verify_gpu_setup.py` shows all green checkmarks

---

## 🚀 Next Steps

Once GPU is enabled:

1. **Run baseline test:**
   ```bash
   python phase2_validation.py
   ```
   - Should show: "Using CUDA: NVIDIA GeForce RTX 2050"
   - FPS should jump from ~21 to 150-200+

2. **Install CUDA Toolkit** (if not done):
   - See "Problem 4" above for instructions

3. **Run performance benchmark:**
   ```bash
   python benchmarks/week1_performance_test.py
   ```
   - This will create detailed before/after comparison

---

## 📞 Need Help?

If you get stuck:

1. **Take photos** of any error messages or BIOS screens
2. **Note exactly** where you got stuck
3. **Run diagnostic:**
   ```bash
   python check_gpu.bat > gpu_diagnostic.txt
   ```
4. Share the `gpu_diagnostic.txt` file

---

## 🔄 Reverting Changes

If you need to disable the GPU (for battery life):

1. Enter BIOS again (F10 on startup)
2. Navigate to same Graphics setting
3. Change to **Integrated** or **Disabled**
4. Save and exit (F10)

> [!NOTE]
> You can switch between modes anytime - changes take effect after restart

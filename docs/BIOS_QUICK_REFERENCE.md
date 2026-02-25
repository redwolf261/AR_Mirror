# 🚀 BIOS GPU Setup - Quick Reference Card

## ⚡ QUICK STEPS

1. **Restart** your computer
2. **Press F10** repeatedly when HP logo appears
3. Navigate to **Advanced** tab (use arrow keys →)
4. Select **Device Configuration** (press Enter)
5. Find **Switchable Graphics**
6. Set to **Enabled** (press Enter)
7. **Press F10** to Save and Exit
8. Confirm **Yes** (press Enter)

---

## ⌨️ KEYBOARD CONTROLS

| Key | Function |
|-----|----------|
| **F10** | Enter BIOS / Save & Exit |
| **Arrow Keys** | Navigate menus |
| **Enter** | Select option |
| **Esc** | Go back / Cancel |

---

## 🎯 WHAT TO LOOK FOR

**Option names may vary:**
- "Switchable Graphics" → Set to **Enabled**
- "Discrete Graphics" → Set to **Enabled**  
- "Graphics Mode" → Set to **Hybrid** or **Discrete**

**DO NOT select:**
- ❌ "Integrated Only"
- ❌ "Disabled"

---

## ✅ AFTER RESTART

Open PowerShell and run:
```powershell
cd "C:\Users\HP\Projects\AR Mirror"
python verify_gpu_setup.py
```

**Expected result:**
```
✓ CUDA Available: True
✓ GPU: NVIDIA GeForce RTX 2050
```

---

## 🆘 TROUBLESHOOTING

**F10 not working?**
- Try: **Esc** then **F10**
- Or: Hold **Shift** → Restart → UEFI Settings

**Can't find Graphics option?**
- Check all tabs: Main, Advanced, System Configuration
- Look for: "Device", "Graphics", "Video", "Display"

**Still need help?**
- Full guide: `docs/GPU_BIOS_SETUP_GUIDE.md`
- Run diagnostic: `python check_gpu.bat`

---

## 📱 TIP: Save This on Your Phone

Take a photo of this card or open it on your phone before restarting, so you have it available while in BIOS!

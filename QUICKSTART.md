# 🚀 AR MIRROR - QUICK START GUIDE

## Full-Fledged System Setup Complete!

### ✅ What's Been Configured

1. **Python ML Backend** (Phase 0 - Alpha Blending)
   - ✅ All dependencies installed
   - ✅ 10 sample garments generated
   - ✅ MediaPipe pose detection ready
   - ✅ ONNX human parsing model downloaded (255MB)
   - ✅ Integrated web UI server (port 5050)
   - ✅ Data flywheel session logging active

2. **NestJS Backend** (Optional)
   - ✅ Dependencies installed (679 packages)
   - ✅ Prisma client generated
   - ✅ Environment configured
   - ⚠️ Database not initialized (PostgreSQL required for production)

3. **Models Available**
   - ✅ `pose_landmarker_lite.task` - MediaPipe pose model
   - ✅ `models/schp_lip.onnx` - Human parsing (255MB)
   - ⚠️ GMM/TOM models not downloaded (optional for Phase 2)

---

## 🎮 How to Run

### Option 1: Quick Start (Python Only - Recommended)
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the AR Mirror
python app.py --phase 0 --duration 0

# Or use the launcher
python launch.py
```

**Access:**
- Camera Window: Opens automatically
- Web UI: http://localhost:5050
- React UI: http://localhost:3001 (if available)

**Controls:**
- Press `→` or `Tab` - Next garment
- Press `←` - Previous garment
- Press `o` - Toggle overlay
- Press `d` - Toggle debug mode
- Press `q` - Quit

---

### Option 2: Full Stack (Python + NestJS + Frontend)

**Terminal 1 - Python ML Backend:**
```bash
.venv\Scripts\activate
python app.py --phase 0 --duration 0
```

**Terminal 2 - NestJS Backend (Optional):**
```bash
cd backend

# Initialize database (first time only)
npx prisma migrate dev --name init

# Start server
npm run dev
```

**Terminal 3 - React Frontend (Optional):**
```bash
# Create frontend if it doesn't exist
npx create-react-app frontend
cd frontend
npm start
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AR MIRROR SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │ Python ML Backend│◄────►│  NestJS Backend  │       │
│  │  • Camera (720p) │      │  • REST API       │       │
│  │  • MediaPipe Pose│      │  • PostgreSQL     │       │
│  │  • Garment Render│      │  • Fit Prediction │       │
│  │  • Web UI Server │      │  • Analytics      │       │
│  │  Port: 5050      │      │  Port: 3000       │       │
│  └──────┬───────────┘      └──────────────────┘       │
│         │                                               │
│         │                                               │
│  ┌──────▼───────────────────────────────────────┐     │
│  │          React Frontend (Optional)           │     │
│  │          • Garment Gallery                   │     │
│  │          • Live AR Preview                   │     │
│  │          • Measurements HUD                  │     │
│  │          Port: 3001                          │     │
│  └──────────────────────────────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Performance Metrics

**Current Status (Phase 0):**
- Average FPS: 7.8
- Camera Resolution: 1280x720
- Pose Detection: 33.3ms per frame
- Garments: 10 samples loaded

**Target Performance:**
- Phase 0: 30+ FPS (Alpha Blending)
- Phase 2: 21+ FPS (Neural Warping)
- Phase 3: 228 FPS (3D Mesh Pipeline)

---

## 🔧 Troubleshooting

### Low FPS (<10 FPS)
- Reduce camera resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
- Increase frame skip intervals
- Use Phase 0 instead of Phase 2

### Camera Not Opening
- Check if another app is using the camera
- Try different camera index: `cv2.VideoCapture(1)`

### Missing Models
- Run `python scripts/download_onnx_model.py` for human parsing
- Download GMM/TOM models for Phase 2 (optional)

### Backend Won't Start
- Install PostgreSQL or use SQLite
- Run `npx prisma migrate dev`
- Check `.env` file configuration

---

## 📁 Project Structure

```
AR_Mirror/
├── app.py                    # Main application (START HERE)
├── launch.py                 # Quick launcher script
├── generate_dataset_samples.py  # Generate garments
├── .venv/                    # Python virtual environment
├── dataset/
│   └── train/
│       ├── cloth/            # 10 sample garments
│       └── cloth-mask/       # Garment masks
├── models/
│   └── schp_lip.onnx         # Human parsing (255MB)
├── backend/                  # NestJS API
│   ├── src/                  # TypeScript source
│   ├── prisma/               # Database schema
│   └── .env                  # Configuration
├── logs/                     # Session logs
└── learned_corrections/      # SKU bias data
```

---

## 🎨 Garment Management

### Current Garments (10 samples):
1. Red shirt
2. Blue shirt
3. Green shirt
4. Yellow shirt
5. Purple shirt
6. Cyan shirt
7. Orange shirt
8. Pink shirt
9. White shirt
10. Gray shirt

### Add Real Garments:
```bash
# Place garment images in:
dataset/train/cloth/GARMENT_NAME.jpg

# Place corresponding masks in:
dataset/train/cloth-mask/GARMENT_NAME.jpg
```

---

## 🚀 Next Steps

### For Better Performance:
1. Install CUDA-enabled PyTorch for GPU acceleration
2. Reduce camera resolution
3. Optimize frame processing pipeline

### For Phase 2 (Neural Warping):
1. Download GMM model checkpoint
2. Download TOM model checkpoint
3. Run: `python app.py --phase 2`

### For Production Deployment:
1. Set up PostgreSQL database
2. Configure production environment variables
3. Deploy NestJS backend to cloud
4. Build React frontend
5. Set up reverse proxy (nginx)

---

## 📞 Support

- Documentation: `docs/` directory
- Issues: https://github.com/redwolf261/AR_Mirror/issues
- Email: rivanshetty771@gmail.com

---

**Made with ❤️ for the future of virtual try-on technology**

Last Updated: 2026-03-20

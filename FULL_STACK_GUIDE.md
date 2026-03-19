# 🚀 AR Mirror - Complete System Startup

## Full-Stack Launch Guide

---

## 🎯 Option 1: Python ML Backend Only (Quickest)

**Best for:** Immediate AR try-on experience

```bash
# Terminal 1: Start AR Mirror
cd "c:\Users\HP\Projects\AR Mirror"
.venv\Scripts\activate
python app.py --phase 0 --duration 0
```

**Access:**
- **Camera Window:** Opens automatically
- **Basic Web UI:** http://localhost:5050

---

## 🎨 Option 2: Python + Apple-Style Frontend (Recommended)

**Best for:** Premium user experience with refined design

### Terminal 1: Python ML Backend
```bash
cd "c:\Users\HP\Projects\AR Mirror"
.venv\Scripts\activate
python app.py --phase 0 --duration 0
```

### Terminal 2: React Frontend
```bash
cd "c:\Users\HP\Projects\AR Mirror\frontend\ar-mirror-ui"
npm start
```

**Access:**
- **Python Backend:** http://localhost:5050 (API + stream)
- **React Frontend:** http://localhost:3001 (Premium UI) ✨

---

## 🏢 Option 3: Complete Production Stack

**Best for:** Full production deployment with database

### Terminal 1: Python ML Backend
```bash
cd "c:\Users\HP\Projects\AR Mirror"
.venv\Scripts\activate
python app.py --phase 0 --duration 0
```

### Terminal 2: NestJS Backend API
```bash
cd "c:\Users\HP\Projects\AR Mirror\backend"

# First time only: Initialize database
npx prisma migrate dev --name init

# Start server
npm run dev
```

### Terminal 3: React Frontend
```bash
cd "c:\Users\HP\Projects\AR Mirror\frontend\ar-mirror-ui"
npm start
```

**Access:**
- **Frontend:** http://localhost:3001
- **Python ML:** http://localhost:5050
- **NestJS API:** http://localhost:3000

---

## 🎨 Frontend Features (Option 2 & 3)

### Apple-Inspired Design
✨ **Clean, minimal, elegant** - Apple.com aesthetic
✨ **Smooth animations** - 60fps transitions
✨ **iOS-style toggles** - Native feel
✨ **Refined typography** - SF Pro-inspired (Inter font)
✨ **Subtle shadows** - No harsh borders
✨ **Generous whitespace** - Professional layout

### Components
- **Video Viewport** - Centered, rounded, with subtle shadow
- **Measurements Panel** - Clean metric cards with hover effects
- **Garment Selector** - Grid with smooth interactions
- **Toggle Controls** - Show/hide skeleton & measurements
- **Connection Status** - Live indicator with FPS display

### Interactions
- **Garment Cards** - Click to select, smooth scale on hover
- **Toggle Switches** - Apple-style sliding animation
- **Metric Cards** - Subtle hover glow effect
- **Video Window** - Slight scale on hover (1.01x)

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 AR MIRROR - FULL STACK                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  React Frontend (Port 3001)                    │    │
│  │  • Apple-inspired design                       │    │
│  │  • Real-time video stream                      │    │
│  │  • Measurements display                        │    │
│  │  • Garment selector                            │    │
│  │  • Smooth animations                           │    │
│  └─────────────────┬──────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  Python ML Backend (Port 5050)                 │    │
│  │  • Camera feed (1280x720)                      │    │
│  │  • MediaPipe pose detection                    │    │
│  │  • Garment overlay                             │    │
│  │  • Body measurements                           │    │
│  │  • MJPEG stream server                         │    │
│  │  • REST API endpoints                          │    │
│  └─────────────────┬──────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  NestJS Backend (Port 3000) - Optional         │    │
│  │  • PostgreSQL database                         │    │
│  │  • Product catalog                             │    │
│  │  • Analytics tracking                          │    │
│  │  • SKU correction learning                     │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🎮 User Controls

### Camera Window (OpenCV)
- `→` or `Tab` - Next garment
- `←` - Previous garment
- `o` - Toggle info overlay
- `d` - Toggle debug mode
- `q` - Quit

### React Frontend (Browser)
- **Garment Cards** - Click to select
- **Toggle Switches** - Click to show/hide skeleton/measurements
- **Auto-refresh** - Updates every 100ms

---

## 📁 Port Configuration

| Service | Port | URL |
|---------|------|-----|
| **React Frontend** | 3001 | http://localhost:3001 |
| **Python ML Backend** | 5050 | http://localhost:5050 |
| **NestJS API** | 3000 | http://localhost:3000 |

---

## 🔧 Configuration Files

### Frontend `.env` (Optional)
```bash
# frontend/ar-mirror-ui/.env
REACT_APP_API_URL=http://localhost:5050
```

### Backend `.env`
```bash
# backend/.env
PORT=3000
DATABASE_URL="postgresql://user:pass@localhost:5432/ar_mirror"
ML_SERVICE_URL="http://localhost:5050"
CORS_ORIGIN="http://localhost:3001"
```

---

## 🎨 Frontend Design System

### Colors
```css
Background: #ffffff (Pure white)
Surface: #f5f5f7 (Very light gray)
Accent: #0071e3 (Apple blue)
Text Primary: #1d1d1f (Almost black)
Text Secondary: #6e6e73 (Gray)
```

### Typography
```
Font: Inter (SF Pro fallback)
Heading: 28px-40px, Semibold
Body: 15px-17px, Regular
Caption: 11px-13px, Medium
```

### Spacing (8px grid)
```
xs:  4px
sm:  8px
md:  16px
lg:  24px
xl:  32px
2xl: 48px
```

### Animations
```
Duration: 150-350ms
Easing: cubic-bezier(0.4, 0, 0.2, 1)
```

---

## 📊 Performance

### Current Status
- **Python Backend:** 7.8 FPS (Phase 0)
- **Frontend:** 60 FPS UI animations
- **API Response:** < 100ms

### Optimization Tips
1. Reduce camera resolution to 640x480 for more FPS
2. Increase skip-frame intervals in Python
3. Use Phase 0 (alpha blending) for fastest rendering
4. Enable GPU acceleration (CUDA PyTorch)

---

## 🐛 Troubleshooting

### Frontend won't start
```bash
cd frontend/ar-mirror-ui
rm -rf node_modules package-lock.json
npm install
npm start
```

### Video stream not showing
- Check Python backend is running on port 5050
- Verify camera is working (`python app.py`)
- Check browser console for CORS errors

### Measurements not updating
- Ensure MediaPipe is installed
- Check Python logs for pose detection errors
- Verify API endpoint: http://localhost:5050/api/state

### Garments not appearing
- Check `dataset/train/cloth/` has .jpg files
- Run `python generate_dataset_samples.py` for samples
- Verify API: http://localhost:5050/api/garments

---

## 🚀 Production Deployment

### Build Frontend
```bash
cd frontend/ar-mirror-ui
npm run build
```

Outputs to `build/` directory - deploy to:
- **Vercel:** `vercel --prod`
- **Netlify:** `netlify deploy --prod`
- **Static hosting:** Serve `build/` folder

### Deploy Backend
1. Set up PostgreSQL database
2. Configure environment variables
3. Run migrations: `npx prisma migrate deploy`
4. Start with PM2: `pm2 start npm --name "ar-mirror-backend" -- run dev`

---

## 📁 Directory Structure

```
AR Mirror/
├── frontend/
│   └── ar-mirror-ui/          ← React app (Port 3001)
│       ├── src/
│       │   ├── components/    ← UI components
│       │   ├── App.js         ← Main app
│       │   └── App.css        ← Design system
│       └── public/
│
├── backend/                    ← NestJS API (Port 3000)
│   ├── src/
│   ├── prisma/
│   └── .env
│
├── src/                        ← Python ML modules
├── app.py                      ← Python ML backend (Port 5050)
├── dataset/
│   └── train/
│       └── cloth/              ← Garment images
└── models/                     ← ML models
```

---

## ✅ Pre-Flight Checklist

### Before starting:
- [ ] Python virtual environment activated
- [ ] All Python dependencies installed
- [ ] Node.js and npm installed
- [ ] Camera permissions granted
- [ ] Ports 3000, 3001, 5050 available

### For full stack:
- [ ] PostgreSQL installed (or using SQLite)
- [ ] Database initialized (`prisma migrate dev`)
- [ ] All `.env` files configured

---

## 🎯 Quick Commands

### Start Everything (3 terminals)
```bash
# Terminal 1
cd "c:\Users\HP\Projects\AR Mirror" && .venv\Scripts\activate && python app.py --phase 0 --duration 0

# Terminal 2
cd "c:\Users\HP\Projects\AR Mirror\frontend\ar-mirror-ui" && npm start

# Terminal 3 (optional)
cd "c:\Users\HP\Projects\AR Mirror\backend" && npm run dev
```

### Stop Everything
Press `Ctrl+C` in each terminal

---

## 📞 Support

**Frontend Guide:** `frontend/FRONTEND_GUIDE.md`
**Backend API:** `web_server.py` - API documentation
**Python App:** `QUICKSTART.md` - Setup guide

---

**🎨 Premium Apple-level design + Powerful AR technology**

**Made with ❤️ by Claude Code**

*Last Updated: 2026-03-20*

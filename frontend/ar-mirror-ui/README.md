# AR Mirror - Apple-Style Frontend

> Premium body tracking interface with Apple.com-level design refinement

![Status](https://img.shields.io/badge/status-production_ready-success)
![React](https://img.shields.io/badge/React-18.2-blue)
![Design](https://img.shields.io/badge/design-Apple_inspired-lightgrey)

---

## 🎨 Design Preview

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  AR Mirror                                        ● Live  7.8 FPS  ┃
┃  ─────────────────────────────────────────────────────────────── ┃
┃                                                                    ┃
┃  ┌────────────────────────────────┐  ┌──────────────────────┐   ┃
┃  │                                │  │ BODY MEASUREMENTS    │   ┃
┃  │                                │  │                      │   ┃
┃  │         Video Stream           │  │ ┌────────┬────────┐ │   ┃
┃  │      (Centered, 16:9)          │  │ │Shoulder│  Chest │ │   ┃
┃  │                                │  │ │ 42.5cm │ 96.3cm │ │   ┃
┃  │    ┌──────────────────┐       │  │ └────────┴────────┘ │   ┃
┃  │    │  [Person with    │       │  │ ┌────────┬────────┐ │   ┃
┃  │    │   pose overlay]  │       │  │ │ Waist  │  Torso │ │   ┃
┃  │    │                  │       │  │ │ 82.1cm │ 58.7cm │ │   ┃
┃  │    └──────────────────┘       │  │ └────────┴────────┘ │   ┃
┃  │                                │  │                      │   ┃
┃  │   "Pose tracking active"      │  │ PERFORMANCE          │   ┃
┃  │                                │  │  Frame Rate: 7.8 FPS │   ┃
┃  └────────────────────────────────┘  │                      │   ┃
┃                                      │ GARMENTS             │   ┃
┃                                      │ ┌────┬────┐         │   ┃
┃                                      │ │ 🔵 │    │ ...    │   ┃
┃                                      │ └────┴────┘         │   ┃
┃                                      │                      │   ┃
┃                                      │ DISPLAY OPTIONS      │   ┃
┃                                      │ Show Skeleton [●]   │   ┃
┃                                      │ Show Measure  [●]   │   ┃
┃                                      └──────────────────────┘   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## ✨ Key Features

### **Visual Design**
- 🎨 **Apple-inspired** aesthetic (apple.com, Apple Health)
- 🤍 **Pure white** background with subtle grays
- 📐 **8px grid** spacing system (no arbitrary values)
- ✍️ **Inter font** (SF Pro-like rendering)
- 🌊 **Smooth transitions** with Apple's easing curves

### **Components**
- 📹 **Video Viewport** - Centered, 16:9, rounded corners
- 📊 **Metric Cards** - 2x2 grid with hover effects
- 🎛️ **iOS Toggles** - 51px exact dimensions
- 🖼️ **Garment Grid** - Visual selection with thumbnails
- 📈 **FPS Monitor** - Real-time performance display

### **Interactions**
- 🖱️ **Smooth hovers** - Scale & translateY
- 🎬 **Fade animations** - 250ms transitions
- 🔄 **Auto-reconnect** - Graceful error handling
- ⚡ **100ms polling** - Real-time state updates

---

## 🚀 Quick Start

```bash
# Terminal 1: Start Python backend
cd "c:\Users\HP\Projects\AR Mirror"
.venv\Scripts\activate
python app.py --phase 0 --duration 0

# Terminal 2: Start React frontend
cd "c:\Users\HP\Projects\AR Mirror\frontend\ar-mirror-ui"
npm start
```

Opens at **http://localhost:3001** ✨

---

## 🏗️ Component Architecture

```javascript
App
├── Header                    // Top nav, status, FPS
├── VideoSection
│   └── VideoFeed            // MJPEG stream display
└── MeasurementsPanel        // Right sidebar
    ├── MetricCard × 4       // Shoulder, Chest, Waist, Torso
    ├── FPSDisplay           // Performance monitor
    ├── GarmentSelector      // Visual garment grid
    └── Controls             // Toggle switches
```

### **State Management**
```javascript
const [systemState, setSystemState] = useState({
  fps: 0,
  garment: '',
  measurements: { shoulder_cm, chest_cm, waist_cm, torso_cm },
  connected: false
});

// Poll backend every 100ms
useEffect(() => {
  const interval = setInterval(async () => {
    const data = await fetch('/api/state');
    setSystemState(data);
  }, 100);
}, []);
```

---

## 🎨 Design System

### **Colors**
```css
--color-background: #ffffff      /* Pure white */
--color-surface: #f5f5f7         /* Apple gray */
--color-accent: #0071e3          /* Apple blue */
--color-text-primary: #1d1d1f    /* Near black */
--color-text-secondary: #6e6e73  /* Gray */
```

### **Typography**
```css
Font: 'Inter', -apple-system, SF Pro
Sizes: 11px, 13px, 15px, 17px, 21px, 28px, 40px
Weights: 300, 400, 500, 600, 700
Letter-spacing: -0.5px (headings), 0.3px-0.5px (labels)
```

### **Spacing (8px base)**
```css
xs: 4px, sm: 8px, md: 16px, lg: 24px, xl: 32px, 2xl: 48px
```

### **Shadows** (Very subtle)
```css
sm: 0 1px 3px rgba(0,0,0,0.04)
md: 0 4px 16px rgba(0,0,0,0.06)
lg: 0 8px 32px rgba(0,0,0,0.08)
```

### **Animations**
```css
Easing: cubic-bezier(0.4, 0, 0.2, 1)  /* Apple's curve */
Durations: 150ms (fast), 250ms (base), 350ms (slow)
```

---

## 🔌 Backend API Integration

### **Endpoints**
```
GET  /stream              → MJPEG video stream
GET  /api/state           → { fps, garment, measurements }
GET  /api/garments        → { garments: [...] }
POST /api/garment         → { name: "00001_00.jpg" }
POST /api/params          → { show_skeleton: true }
GET  /api/garment_image/:name  → Image thumbnail
```

### **Polling Strategy**
- **State:** Every 100ms (10 updates/sec)
- **Video:** MJPEG stream (continuous)
- **Garments:** Once on mount

---

## 📱 Responsive Breakpoints

```css
Desktop:  > 1200px  (Sidebar: 360px, Metrics: 2-col)
Tablet:   968-1200px (Sidebar: 320px, Metrics: 1-col)
Mobile:   < 968px   (Stacked layout, full-width video)
```

---

## 🎯 Apple Design Principles Applied

### **1. Minimal & Elegant**
- No unnecessary elements
- Generous whitespace (24-48px gaps)
- Clean hierarchy (size, weight, color)

### **2. Subtle Motion**
- 60fps CSS animations (GPU-accelerated)
- Apple's cubic-bezier easing
- Micro-interactions on hover

### **3. Refined Typography**
- System fonts (Inter → SF Pro fallback)
- Clear hierarchy (40px → 11px)
- Precise letter-spacing

### **4. Focus on Clarity**
- Main content centered
- Secondary info in sidebar
- Only essential data visible
- Toggles for optional details

### **5. Perfect Alignment**
- 8px grid system
- Consistent spacing
- Optical balance

### **6. Soft Neutrals**
- No harsh colors
- Subtle shadows (< 0.08 opacity)
- Light borders (rgba)

### **7. Professional Polish**
- Loading states
- Error handling
- Hover feedback
- Connection monitoring

---

## 📁 File Structure

```
frontend/ar-mirror-ui/
├── public/
│   └── index.html                 # Inter font, meta tags
├── src/
│   ├── components/
│   │   ├── Header.js              # Navigation + status
│   │   ├── VideoFeed.js           # MJPEG stream
│   │   ├── MeasurementsPanel.js   # Sidebar container
│   │   ├── MetricCard.js          # Individual metric
│   │   ├── Controls.js            # iOS toggles
│   │   ├── GarmentSelector.js     # Garment grid
│   │   └── Enhanced.css           # Animations
│   ├── App.js                     # Main app
│   ├── App.css                    # Design system (600 lines)
│   ├── index.js                   # React entry
│   └── index.css                  # Reset
├── package.json                   # Dependencies
└── FRONTEND_GUIDE.md              # Documentation
```

---

## 🔧 Development

```bash
# Install dependencies
npm install

# Start dev server (port 3001)
npm start

# Build for production
npm run build

# Serve production build
npx serve -s build -p 3001
```

---

## 🎯 Future Enhancements

- [ ] Dark mode (macOS-style)
- [ ] Keyboard shortcuts (Space, M, arrows)
- [ ] Measurement history timeline
- [ ] Export measurements (PDF/CSV)
- [ ] Garment comparison view
- [ ] Swipe gestures for mobile
- [ ] Offline mode
- [ ] PWA support

---

## 📞 Documentation

- **Frontend Guide:** `FRONTEND_GUIDE.md` (Complete API & design docs)
- **Full Stack:** `../FULL_STACK_GUIDE.md` (All components)
- **Implementation:** `../FRONTEND_COMPLETE.md` (What's been built)

---

## ✅ Production Ready

**Status:** ✅ **COMPLETE**
**Quality:** **Apple.com Standard**
**Lines of Code:** ~1500
**Components:** 7
**Design System:** Complete

---

**🎨 Built with Apple-level attention to detail**

*Ready to launch: `npm start`*

# 🎨 AR Mirror Frontend - Apple-Inspired UI

**Premium body tracking interface with Apple-level design refinement**

---

## ✨ Design Philosophy

This frontend embodies Apple's design principles:

### Visual Identity
- **Minimal & Elegant** - Generous whitespace, perfect alignment
- **Subtle Motion** - Smooth transitions with Apple's easing curves
- **Refined Typography** - System fonts (Inter/SF Pro), clear hierarchy
- **Soft Neutrals** - Light backgrounds, subtle shadows, no sci-fi aesthetics

### User Experience
- **Centered Focus** - Main video viewport commands attention
- **Clean Hierarchy** - Side panel for metrics, never cluttered
- **Smooth Interactions** - Hover states, micro-animations, polished feel
- **Clarity First** - Essential data only, toggle for more details

---

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- AR Mirror Python backend running (port 5050)

### Installation

```bash
# Navigate to frontend directory
cd "c:\Users\HP\Projects\AR Mirror\frontend"

# Install dependencies (if not auto-installed)
npm install

# Start development server
npm start
```

The frontend will open at **http://localhost:3001**

---

## 🏗️ Architecture

### Component Structure

```
frontend/
├── public/
│   └── index.html          # HTML shell with fonts
├── src/
│   ├── components/
│   │   ├── Header.js       # Top navigation bar
│   │   ├── VideoFeed.js    # Main video viewport
│   │   ├── MeasurementsPanel.js  # Right sidebar
│   │   ├── MetricCard.js   # Individual measurement cards
│   │   ├── Controls.js     # Toggle switches
│   │   ├── GarmentSelector.js    # Garment grid
│   │   └── Enhanced.css    # Additional animations
│   ├── App.js              # Main application
│   ├── App.css             # Design system & styles
│   ├── index.js            # React entry point
│   └── index.css           # Global reset
└── package.json            # Dependencies + proxy config
```

### Component Breakdown

#### **App.js** - Main Orchestrator
- Manages global state (measurements, FPS, connection)
- Polls backend API every 100ms
- Coordinates all child components
- Handles API communication

#### **Header.js** - Top Navigation
- Displays app title
- Shows connection status (live indicator)
- Displays real-time FPS
- Translucent background with blur effect

#### **VideoFeed.js** - Video Viewport
- Displays MJPEG stream from `/stream`
- Handles loading states
- Auto-reconnect on error
- Centered with rounded corners & shadow

#### **MeasurementsPanel.js** - Sidebar Panel
- Shows body measurements in grid
- Displays performance metrics
- Houses garment selector
- Contains display toggle controls

#### **MetricCard.js** - Measurement Display
- Individual metric (shoulder, chest, waist, torso)
- Hover effects
- Smooth transitions
- Null-safe rendering

#### **Controls.js** - Toggle Switches
- iOS-style toggle switches
- Controls skeleton visibility
- Controls measurement overlay
- Syncs with backend parameters

#### **GarmentSelector.js** - Garment Grid
- Displays garment thumbnails
- Active selection highlighting
- Smooth hover states
- Lazy loading images

---

## 🎨 Design System

### Color Palette

```css
/* Neutrals */
--color-background: #ffffff
--color-surface: #f5f5f7
--color-text-primary: #1d1d1f
--color-text-secondary: #6e6e73

/* Accent */
--color-accent: #0071e3  /* Apple blue */

/* Skeleton/Pose */
--color-skeleton: #0071e3  /* Soft blue, not neon */
```

### Typography

```css
Font Family: Inter (fallback: SF Pro, System UI)

Sizes:
  Heading:  28px - 40px (semibold)
  Body:     15px - 17px (regular)
  Caption:  11px - 13px (medium)

Weights:
  300 - Light
  400 - Regular
  500 - Medium
  600 - Semibold
  700 - Bold
```

### Spacing (8px base)

```css
4px   - xs  (tight gaps)
8px   - sm  (component padding)
16px  - md  (section spacing)
24px  - lg  (panel padding)
32px  - xl  (major sections)
48px+ - 2xl+  (page-level spacing)
```

### Animations

All transitions use Apple's easing:
```css
cubic-bezier(0.4, 0, 0.2, 1)

Durations:
  Fast:  150ms  (hover states)
  Base:  250ms  (standard transitions)
  Slow:  350ms  (large motions)
```

---

## 🔌 Backend Integration

### API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/stream` | GET | MJPEG video stream |
| `/api/state` | GET | Current FPS, garment, measurements |
| `/api/garments` | GET | List of available garments |
| `/api/garment` | POST | Switch active garment |
| `/api/params` | GET | Get display parameters |
| `/api/params` | POST | Update display parameters |
| `/api/garment_image/:name` | GET | Garment thumbnail |

### State Polling

The frontend polls `/api/state` every 100ms for:
- Real-time FPS updates
- Body measurements
- Connection status

### Example Response

```json
{
  "fps": 7.8,
  "garment": "00001_00.jpg",
  "measurements": {
    "shoulder_cm": 42.5,
    "chest_cm": 96.3,
    "waist_cm": 82.1,
    "torso_cm": 58.7,
    "size": "M"
  }
}
```

---

## 🎮 User Interactions

### Keyboard Shortcuts
*(Can be added)*
- `Space` - Toggle skeleton
- `M` - Toggle measurements
- `←/→` - Previous/Next garment

### Mouse Interactions
- **Garment Cards** - Click to select
- **Toggle Switches** - Click to toggle
- **Video** - Subtle scale on hover

---

## 🎯 Key Features

### 1. **Centered Video Viewport**
- Maximum width: 960px
- 16:9 aspect ratio
- Rounded corners (24px radius)
- Subtle shadow
- Smooth hover scale (1.01x)

### 2. **Elegant Measurements Panel**
- Fixed width: 360px
- Clean metric cards in 2-column grid
- Hover effects on cards
- Smooth value transitions

### 3. **iOS-Style Toggle Switches**
- 51px wide, perfect proportions
- Smooth sliding animation
- Apple blue when active
- Gray when inactive

### 4. **Garment Selector**
- 2-column grid
- 3:4 aspect ratio cards
- Active border: blue
- Hover scale: 1.05x

### 5. **Connection Indicator**
- Live dot animation (pulsing)
- Real-time FPS display
- Disconnected state handling

---

## 📱 Responsive Design

```css
/* Large screens (default) */
Sidebar: 360px

/* Medium screens (<1200px) */
Sidebar: 320px
Metrics: 1-column

/* Small screens (<968px) */
Layout: stacked (column)
Video on top
Panel below
```

---

## 🎨 Apple-Inspired Details

### Subtle Elements That Matter

1. **Backdrop Blur** - Header uses `backdrop-filter: blur(20px)`
2. **Soft Shadows** - No harsh drop-shadows, only 0.04-0.08 opacity
3. **Precise Spacing** - 8px grid system, nothing arbitrary
4. **System Fonts** - Inter mimics SF Pro, perfect rendering
5. **Smooth Easing** - All transitions use Apple's cubic-bezier
6. **Minimal Borders** - rgba(0,0,0,0.08) maximum, often lighter
7. **Generous Whitespace** - Breathing room between all elements
8. **Hover Micro-interactions** - -1px translateY, scale(1.01)
9. **Loading States** - Elegant spinners, never jarring
10. **Error Handling** - Graceful, informative, retry logic

---

## 🚀 Production Build

```bash
# Create optimized build
npm run build

# Serve static files
npx serve -s build -p 3001

# Or deploy to hosting
# Vercel: vercel --prod
# Netlify: netlify deploy --prod
```

### Environment Variables

```bash
# .env file
REACT_APP_API_URL=http://localhost:5050
```

For production:
```bash
REACT_APP_API_URL=https://your-backend-domain.com
```

---

## 🔧 Customization

### Change Accent Color

```css
/* App.css */
--color-accent: #0071e3;  /* Change this */
--color-accent-hover: #0077ED;  /* And this */
```

### Adjust Panel Width

```css
/* App.css */
.measurements-panel {
  width: 360px;  /* Change this */
}
```

### Modify Spacing

```css
/* App.css - change base unit */
--space-sm: 8px;   /* 4px, 8px, 12px, etc. */
```

---

## 📊 Performance

### Optimization Techniques

1. **Lazy Loading** - Garment images load on-demand
2. **Memoization** - React.memo for expensive components
3. **Debounced Updates** - State polls throttled to 100ms
4. **CSS Transitions** - Hardware-accelerated GPU animations
5. **MJPEG Streaming** - Efficient video transport

### Performance Targets

- **Time to Interactive:** < 2s
- **Smooth 60fps** UI transitions
- **< 100ms** API response time
- **Minimal re-renders** via React optimization

---

## 🐛 Troubleshooting

### Issue: Video not loading
**Solution:** Check backend is running on port 5050
```bash
curl http://localhost:5050/stream
```

### Issue: CORS errors
**Solution:** Backend has `flask-cors` installed and enabled

### Issue: Garment images 404
**Solution:** Ensure `dataset/train/cloth/` has .jpg files

### Issue: No measurements showing
**Solution:** Backend needs MediaPipe working, check Python logs

---

## 📁 File Checklist

Ensure these files exist:

```
✅ frontend/package.json
✅ frontend/public/index.html
✅ frontend/src/App.js
✅ frontend/src/App.css
✅ frontend/src/index.js
✅ frontend/src/index.css
✅ frontend/src/components/Header.js
✅ frontend/src/components/VideoFeed.js
✅ frontend/src/components/MeasurementsPanel.js
✅ frontend/src/components/MetricCard.js
✅ frontend/src/components/Controls.js
✅ frontend/src/components/GarmentSelector.js
✅ frontend/src/components/Enhanced.css
```

---

## 🎯 Future Enhancements

### Planned Features

1. **Dark Mode** - macOS-style dark theme
2. **Animations** - More subtle micro-interactions
3. **Gestures** - Swipe for garment change
4. **Keyboard Nav** - Full keyboard accessibility
5. **History** - View past measurements
6. **Export** - Save measurements as PDF
7. **Comparison** - Side-by-side garment view
8. **Mobile App** - React Native version

---

## 📞 Support

**Documentation:** This file + inline code comments
**Backend API:** See `web_server.py` for endpoint details
**Design System:** See `App.css` for all variables

---

**🎨 Built with Apple-level attention to detail**

*Last Updated: 2026-03-20*
*Version: 1.0.0*

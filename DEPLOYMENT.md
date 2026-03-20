# AR Mirror - Vercel Deployment Guide

## 🚀 **Deployment Architecture**

### **Frontend (React) → Vercel ✅**
### **Backend (Python ML) → Railway/Render ⚡**

## 📋 **Step 1: Frontend to Vercel**

### **1.1 Prepare Frontend for Production**

Update API configuration for different environments:

```bash
cd "frontend/ar-mirror-ui/src"
# Create environment config
```

Create `src/config.js`:
```javascript
const config = {
  API_BASE_URL: process.env.NODE_ENV === 'production'
    ? process.env.REACT_APP_API_URL || 'https://your-backend-url.railway.app'
    : 'http://localhost:5050'
};

export default config;
```

### **1.2 Update API Calls**
Modify your API calls to use the config:
```javascript
import config from './config';

// Instead of: fetch('/api/measurements')
// Use: fetch(`${config.API_BASE_URL}/api/measurements`)
```

### **1.3 Deploy to Vercel**

**Option A: GitHub Integration (Recommended)**
```bash
# Push to GitHub first
git add .
git commit -m "Prepare frontend for Vercel deployment"
git push origin main

# Then connect to Vercel:
# 1. Go to vercel.com
# 2. Import your GitHub repo
# 3. Set root directory to: frontend/ar-mirror-ui
# 4. Deploy!
```

**Option B: Vercel CLI**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd "frontend/ar-mirror-ui"
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your username
# - Link to existing project? No
# - What's your project's name? ar-mirror-ui
# - In which directory is your code located? ./
```

## 🐍 **Step 2: Backend Deployment Options**

### **Option A: Railway (Recommended for ML apps)**

**2.1 Create Railway account** → [railway.app](https://railway.app)

**2.2 Prepare backend for Railway:**
Create `Procfile` in project root:
```
web: python app.py --host 0.0.0.0 --port $PORT
```

Create `runtime.txt`:
```
python-3.11
```

**2.3 Deploy to Railway:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### **Option B: Render**
1. Go to [render.com](https://render.com)
2. Connect GitHub repo
3. Create new Web Service
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

### **Option C: Heroku**
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-ar-mirror-backend

# Add buildpacks
heroku buildpacks:add heroku/python

# Deploy
git push heroku main
```

## 🔗 **Step 3: Connect Frontend to Backend**

### **3.1 Set Environment Variable**
In your Vercel project settings:
- Add environment variable: `REACT_APP_API_URL`
- Value: `https://your-backend-url.railway.app`

### **3.2 Update CORS in Backend**
Add your Vercel URL to CORS settings in `web_server.py`:
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://your-app-name.vercel.app"
])
```

## 🌐 **Step 4: Final URLs**

After deployment:
- **Frontend**: `https://your-app-name.vercel.app`
- **Backend**: `https://your-backend.railway.app`

## ⚡ **Quick Deploy Commands**

```bash
# 1. Frontend to Vercel
cd "frontend/ar-mirror-ui"
vercel --prod

# 2. Backend to Railway
railway up

# 3. Update environment
# Set REACT_APP_API_URL in Vercel dashboard
```

## 🔧 **Important Notes**

### **Backend Considerations:**
- **Remove camera requirements** for cloud deployment (unless using headless mode)
- **Use environment variables** for sensitive configs
- **Optimize model loading** for faster cold starts
- **Consider using smaller models** for cloud deployment

### **Frontend Considerations:**
- **Build optimization** is automatic on Vercel
- **Environment variables** are injected at build time
- **API proxy** handled through configuration

## 🚨 **Limitations & Alternatives**

### **Camera Access:**
- Cloud backends can't access user cameras directly
- Consider **client-side processing** with TensorFlow.js
- Or **image upload** workflow instead of real-time

### **Alternative: Serverless Approach**
Deploy a simplified version using:
- **Frontend**: Vercel
- **AI Processing**: Vercel Edge Functions + TensorFlow.js
- **Simpler models**: MediaPipe in browser

## 📊 **Cost Estimates**

- **Vercel**: Free tier covers most frontend needs
- **Railway**: ~$5/month for basic backend
- **Render**: Free tier available, ~$7/month for production
- **Heroku**: ~$7/month (no free tier anymore)

---

**Ready to deploy?** Start with the frontend on Vercel, then choose your preferred backend option!
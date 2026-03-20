# AR Mirror - Cloud Deployment Guide

## 🚀 **Deployment Architecture Options**

### **Frontend (React) → Vercel ✅ or Netlify ✅**
### **Backend (Python ML) → Railway/Render/Heroku ⚡**

## 📋 **Step 1: Frontend Deployment**

Choose your preferred frontend platform:

### **Option A: Deploy to Vercel**
### **Option B: Deploy to Netlify**

---

## 🌐 **Option A: Frontend to Vercel**

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

---

## 🟦 **Option B: Frontend to Netlify**

### **B.1 Prepare Frontend for Netlify**

Frontend is already configured with `netlify.toml` file.

### **B.2 Deploy to Netlify**

**Method 1: GitHub Integration (Recommended)**
```bash
# Push to GitHub first (if not done already)
git add .
git commit -m "Prepare frontend for Netlify deployment"
git push origin main

# Then connect to Netlify:
# 1. Go to netlify.com
# 2. New site from Git → Connect GitHub
# 3. Select your AR_Mirror repository
# 4. Set base directory: frontend/ar-mirror-ui
# 5. Build command: npm run build
# 6. Publish directory: build
# 7. Deploy!
```

**Method 2: Netlify CLI**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login and deploy
cd "frontend/ar-mirror-ui"
netlify login
netlify init
netlify deploy --prod --dir=build

# Follow prompts:
# - Create & configure a new site
# - Choose your team
# - Site name: ar-mirror-ui (or custom name)
```

**Method 3: Drag & Drop**
```bash
# Build the project locally
cd "frontend/ar-mirror-ui"
npm run build

# Then drag the 'build' folder to netlify.com/drop
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

**For Vercel:**
- Go to your Vercel project settings
- Add environment variable: `REACT_APP_API_URL`
- Value: `https://your-backend-url.railway.app`

**For Netlify:**
- Go to your Netlify site settings
- Build & Deploy → Environment variables
- Add: `REACT_APP_API_URL` = `https://your-backend-url.railway.app`

### **3.2 Update CORS in Backend**
Add your deployment URL to CORS settings (already configured in `web_server.py`):
```python
allowed_origins = [
    "http://localhost:3000",           # Local development
    "https://*.vercel.app",           # All Vercel deployments
    "https://*.netlify.app",          # All Netlify deployments
    "https://your-app.vercel.app",    # Your specific Vercel URL
    "https://your-app.netlify.app",   # Your specific Netlify URL
]
```

## 🌐 **Step 4: Final URLs**

After deployment:
**Vercel Option:**
- Frontend: `https://your-app-name.vercel.app`
- Backend: `https://your-backend.railway.app`

**Netlify Option:**
- Frontend: `https://your-app-name.netlify.app`
- Backend: `https://your-backend.railway.app`

## ⚡ **Quick Deploy Commands**

**Vercel Path:**
```bash
# 1. Frontend to Vercel
cd "frontend/ar-mirror-ui"
vercel --prod

# 2. Backend to Railway
railway up
```

**Netlify Path:**
```bash
# 1. Frontend to Netlify
cd "frontend/ar-mirror-ui"
netlify deploy --prod --dir=build

# 2. Backend to Railway
railway up
```

## 🔧 **Important Notes**

### **Backend Considerations:**
- **Remove camera requirements** for cloud deployment (unless using headless mode)
- **Use environment variables** for sensitive configs
- **Optimize model loading** for faster cold starts
- **Consider using smaller models** for cloud deployment

### **Frontend Considerations:**
- **Build optimization** is automatic on both platforms
- **Environment variables** are injected at build time
- **API proxy** handled through configuration
- **Both platforms** offer excellent performance and CDN

## 🆚 **Vercel vs Netlify Comparison**

| Feature | Vercel | Netlify |
|---------|--------|---------|
| **Deployment** | GitHub integration ✅ | GitHub integration ✅ |
| **Build Time** | Fast ⚡ | Fast ⚡ |
| **Free Tier** | 100GB bandwidth | 100GB bandwidth |
| **Environment Variables** | ✅ Easy setup | ✅ Easy setup |
| **Custom Domains** | ✅ Free HTTPS | ✅ Free HTTPS |
| **Form Handling** | ❌ Not included | ✅ Built-in forms |
| **Functions** | ✅ Edge Functions | ✅ Netlify Functions |
| **Analytics** | ✅ Built-in | ✅ Built-in |
| **Deploy Previews** | ✅ PR previews | ✅ Branch previews |
| **CLI** | ✅ Excellent | ✅ Excellent |

**Recommendation:** Both are excellent! Choose based on:
- **Vercel**: If you prefer Vercel's developer experience
- **Netlify**: If you need form handling or prefer Netlify's features

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

### **Frontend (Choose One):**
- **Vercel**: Free tier (100GB bandwidth/month)
- **Netlify**: Free tier (100GB bandwidth/month)

### **Backend:**
- **Railway**: ~$5/month for basic backend
- **Render**: Free tier available, ~$7/month for production
- **Heroku**: ~$7/month (no free tier anymore)

---

**Ready to deploy?** Start with the frontend on Vercel, then choose your preferred backend option!
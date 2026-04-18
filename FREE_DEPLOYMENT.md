# AR Mirror - 100% FREE Deployment Guide 🆓

## 🎯 **Completely Free Architecture**

### **Frontend: Netlify/Vercel (FREE FOREVER)**
### **Backend: Render.com (FREE TIER)**

**Total Cost: $0/month forever! 🎉**

---

## 🌐 **Step 1: FREE Frontend Deployment**

Choose either platform (both completely free):

### **Option A: Netlify (Recommended for beginners)**
1. Go to [netlify.com](https://netlify.com)
2. **New site from Git** → Connect GitHub
3. Select your `AR_Mirror` repository
4. Settings:
   - **Base directory**: `frontend/ar-mirror-ui`
   - **Build command**: `npm run build`
   - **Publish directory**: `build`
5. **Deploy site** 🚀

### **Option B: Vercel**
1. Go to [vercel.com](https://vercel.com)
2. **Import Project** → GitHub
3. Select your `AR_Mirror` repository
4. **Root Directory**: `frontend/ar-mirror-ui`
5. **Deploy** 🚀

---

## 🐍 **Step 2: FREE Backend Deployment**

### **Render.com (Free Tier)**

**2.1** Go to [render.com](https://render.com) and create free account

**2.2** Create **Web Service**:
- **Connect GitHub** → Select `AR_Mirror` repository
- **Name**: `ar-mirror-backend`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py --host 0.0.0.0 --port $PORT`

**2.3** Set environment variables:
- `FLASK_ENV` = `production`
- `PORT` = `10000` (or whatever Render assigns)

**2.4** Deploy! (Takes ~5-10 minutes first time)

---

## 🔗 **Step 3: Connect Frontend to Backend**

### **3.1 Get your backend URL**
After Render deployment completes, you'll get a URL like:
`https://ar-mirror-backend.onrender.com`

### **3.2 Set environment variable in frontend**

**For Netlify:**
1. Go to your site dashboard
2. **Site settings** → **Environment variables**
3. Add: `REACT_APP_API_URL` = `https://your-backend.onrender.com`
4. **Redeploy** (triggers automatic rebuild)

**For Vercel:**
1. Go to project dashboard
2. **Settings** → **Environment Variables**
3. Add: `REACT_APP_API_URL` = `https://your-backend.onrender.com`
4. **Redeploy**

---

## ⚡ **Quick Deploy Commands (CLI Method)**

### **Frontend:**
```bash
cd "frontend/ar-mirror-ui"

# Netlify
npm install -g netlify-cli
netlify deploy --prod --dir=build

# OR Vercel
npm install -g vercel
vercel --prod
```

### **Backend:**
Using GitHub integration (recommended) - just connect your repo!

---

## 🎯 **Free Tier Limits (Generous!)**

### **Frontend (Netlify/Vercel):**
- ✅ **100GB bandwidth/month**
- ✅ **Unlimited sites**
- ✅ **Custom domains**
- ✅ **HTTPS included**
- ✅ **Deploy previews**

### **Backend (Render Free Tier):**
- ✅ **750 hours/month** (basically 24/7)
- ⚠️ **Sleeps after 15 minutes of inactivity** (wakes up automatically)
- ✅ **Custom domain support**
- ✅ **Automatic deploys from GitHub**

---

## 🚨 **Important: Free Tier Considerations**

### **Render Free Tier "Sleep" Mode:**
- App **sleeps after 15 minutes** of no requests
- **Wakes up** when someone visits (takes ~30 seconds)
- **Solution**: Use a service like [UptimeRobot](https://uptimerobot.com) (also free) to ping your app every 14 minutes

### **Camera Limitation:**
- Free backends can't access user cameras
- Solution: Frontend handles camera, sends images to backend for processing

---

## 🛠️ **Modified Backend for Free Deployment**

Create `app_cloud.py` for free deployment (without camera):

```python
# Simplified cloud version - no camera access needed
# Frontend handles camera, sends images via API

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Process uploaded image for measurements
    # Return size recommendations
    return jsonify({
        'measurements': {...},
        'size_recommendation': 'M',
        'confidence': 0.85
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
```

---

## 🎉 **Final FREE URLs**

After deployment:
- **Frontend**: `https://your-app.netlify.app` or `https://your-app.vercel.app`
- **Backend**: `https://your-backend.onrender.com`

## 💸 **Cost Breakdown:**
- **Frontend**: $0/month (Netlify/Vercel free tier)
- **Backend**: $0/month (Render free tier)
- **Total**: **$0/month FOREVER** 🎉

---

## ✅ **What You Get for FREE:**

- ✅ **Live AR Mirror** accessible worldwide
- ✅ **Body measurements** and size recommendations
- ✅ **Custom domain** support (buy domain separately ~$10/year)
- ✅ **HTTPS** included
- ✅ **Global CDN** for fast loading
- ✅ **Automatic deploys** from GitHub

---

## 🚀 **Deploy Now!**

**Total Time**: ~30 minutes setup, then free forever!

1. **Frontend** → Netlify/Vercel (5 mins)
2. **Backend** → Render.com (15 mins)
3. **Connect** → Environment variables (5 mins)
4. **Done** → Share with the world! 🌍

**Your GitHub repo has everything ready**: `https://github.com/redwolf261/AR_Mirror`
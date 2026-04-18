# 🆓 FREE DEPLOYMENT QUICK START

Deploy your AR Mirror **completely free** in under 30 minutes!

## 🎯 What You'll Get (100% Free)
- ✅ Live AR Mirror on the web
- ✅ Body measurements & size recommendations
- ✅ Global CDN for fast loading
- ✅ Custom domain support
- ✅ HTTPS included
- ✅ **$0/month forever!**

## ⚡ Quick Deploy (3 Steps)

### 1️⃣ Frontend → Netlify (5 minutes)
1. Go to [netlify.com](https://netlify.com)
2. **New site from Git** → GitHub → AR_Mirror
3. **Base directory**: `frontend/ar-mirror-ui`
4. **Deploy!** 🚀

### 2️⃣ Backend → Render (15 minutes)
1. Go to [render.com](https://render.com)
2. **New Web Service** → GitHub → AR_Mirror
3. **Build**: `pip install -r requirements_cloud.txt`
4. **Start**: `python app_cloud.py`
5. **Deploy!** 🚀

### 3️⃣ Connect (5 minutes)
1. Copy your Render URL (like `https://xyz.onrender.com`)
2. In Netlify: **Site settings** → **Environment variables**
3. Add: `REACT_APP_API_URL` = your Render URL
4. **Redeploy!** ✨

## 🎉 Done!
Your AR Mirror is now live and free forever!

---

**📋 Detailed Guide**: See `FREE_DEPLOYMENT.md`
**⚡ Quick Script**: Run `./deploy-free.sh`
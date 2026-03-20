#!/bin/bash

# AR Mirror - 100% FREE Deployment Script

echo "🆓 Deploying AR Mirror - 100% FREE! 🎉"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Free Deployment Checklist:${NC}"
echo "✅ Frontend → Netlify/Vercel (FREE)"
echo "✅ Backend → Render.com (FREE tier)"
echo "✅ Total cost: $0/month forever!"
echo ""

# Step 1: Frontend
echo -e "${GREEN}Step 1: Frontend Deployment${NC}"
echo "🌐 Choose your platform:"
echo "  A) Netlify (recommended for beginners)"
echo "  B) Vercel (great DX)"
echo ""
echo "📝 Go to your chosen platform and:"
echo "   1. Connect GitHub repository"
echo "   2. Set base directory: frontend/ar-mirror-ui"
echo "   3. Build command: npm run build"
echo "   4. Publish directory: build"
echo "   5. Deploy!"
echo ""

# Step 2: Backend
echo -e "${GREEN}Step 2: Backend Deployment (Render.com)${NC}"
echo "🐍 Go to render.com and:"
echo "   1. New → Web Service"
echo "   2. Connect GitHub → AR_Mirror repo"
echo "   3. Name: ar-mirror-backend"
echo "   4. Environment: Python 3"
echo "   5. Build Command: pip install -r requirements_cloud.txt"
echo "   6. Start Command: python app_cloud.py"
echo "   7. Deploy!"
echo ""

# Step 3: Connect
echo -e "${GREEN}Step 3: Connect Frontend to Backend${NC}"
echo "🔗 After backend deploys, you'll get a URL like:"
echo "   https://ar-mirror-backend.onrender.com"
echo ""
echo "📝 Add this as environment variable in your frontend:"
echo "   Variable: REACT_APP_API_URL"
echo "   Value: https://your-backend.onrender.com"
echo ""

# Final
echo -e "${YELLOW}🎯 That's it! Your AR Mirror is now FREE and live!${NC}"
echo ""
echo -e "${BLUE}📊 What you get for FREE:${NC}"
echo "✅ Global AR Mirror application"
echo "✅ Body measurements & size recommendations"
echo "✅ Custom domain support"
echo "✅ HTTPS included"
echo "✅ Automatic deploys from GitHub"
echo ""
echo -e "${GREEN}🌍 Share your AR Mirror with the world! 🚀${NC}"
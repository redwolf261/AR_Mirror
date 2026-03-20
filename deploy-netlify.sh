#!/bin/bash

# AR Mirror Frontend - Netlify Deployment Script

echo "🚀 Deploying AR Mirror Frontend to Netlify..."

# Navigate to frontend directory
cd frontend/ar-mirror-ui

# Ensure dependencies are installed
echo "📦 Installing dependencies..."
npm install

# Build the project to check for errors
echo "🔨 Building project..."
npm run build

# Deploy to Netlify
echo "🌐 Deploying to Netlify..."
npx netlify-cli deploy --prod --dir=build

echo "✅ Deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Update environment variable REACT_APP_API_URL in Netlify dashboard"
echo "2. Deploy backend to Railway/Render"
echo "3. Update CORS settings with your Netlify URL"
echo ""
echo "🔗 Access your Netlify dashboard: https://app.netlify.com/"
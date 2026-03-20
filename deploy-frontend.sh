#!/bin/bash

# AR Mirror Frontend - Vercel Deployment Script

echo "🚀 Deploying AR Mirror Frontend to Vercel..."

# Navigate to frontend directory
cd frontend/ar-mirror-ui

# Ensure dependencies are installed
echo "📦 Installing dependencies..."
npm install

# Build the project to check for errors
echo "🔨 Building project..."
npm run build

# Deploy to Vercel
echo "🌐 Deploying to Vercel..."
npx vercel --prod

echo "✅ Deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Update environment variable REACT_APP_API_URL in Vercel dashboard"
echo "2. Deploy backend to Railway/Render"
echo "3. Update CORS settings with your Vercel URL"
echo ""
echo "🔗 Access your Vercel dashboard: https://vercel.com/dashboard"
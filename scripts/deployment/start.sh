#!/bin/bash
# Chic India AR Platform - Quick Start Script (Linux/Mac)
# Launches complete system with one command

echo ""
echo "======================================================================"
echo "CHIC INDIA AR PLATFORM - QUICK START"
echo "======================================================================"
echo ""

echo "[1/4] Checking system..."
python test_integration.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Integration tests failed!"
    echo "Please fix errors before launching."
    exit 1
fi

echo ""
echo "[2/4] Installing dependencies..."
pip install --quiet fastapi uvicorn requests 2>/dev/null
cd backend
npm install --silent 2>/dev/null
cd ..

echo ""
echo "[3/4] Generating garment samples..."
python generate_garment_samples.py

echo ""
echo "[4/4] Launching system..."
echo ""
echo "======================================================================"
echo "SYSTEM STARTING"
echo "======================================================================"
echo "Services will be available at:"
echo "  - Python ML Service: http://localhost:8000"
echo "  - Backend API:       http://localhost:3000"
echo "  - PostgreSQL:        localhost:5432"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================================================"
echo ""

python orchestrator.py

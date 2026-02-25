@echo off
REM Chic India AR Platform - Quick Start Script
REM Launches complete system with one command

echo.
echo ======================================================================
echo CHIC INDIA AR PLATFORM - QUICK START
echo ======================================================================
echo.

echo [1/4] Checking system...
python test_integration.py
if errorlevel 1 (
    echo.
    echo ERROR: Integration tests failed!
    echo Please fix errors before launching.
    pause
    exit /b 1
)

echo.
echo [2/4] Installing dependencies...
pip install --quiet fastapi uvicorn requests 2>nul
cd backend
call npm install --silent 2>nul
cd ..

echo.
echo [3/4] Generating garment samples...
python generate_garment_samples.py

echo.
echo [4/4] Launching system...
echo.
echo ======================================================================
echo SYSTEM STARTING
echo ======================================================================
echo Services will be available at:
echo   - Python ML Service: http://localhost:8000
echo   - Backend API:       http://localhost:3000
echo   - PostgreSQL:        localhost:5432
echo.
echo Press Ctrl+C to stop all services
echo ======================================================================
echo.

python orchestrator.py

pause

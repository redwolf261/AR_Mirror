@echo off
echo ============================================================================
echo AR MIRROR - FULL STACK STARTUP
echo ============================================================================
echo.
echo Starting all components:
echo  - Python ML Backend (Camera + AR Processing) on port 5050
echo  - NestJS API Server on port 3000
echo  - React Frontend on port 3001
echo.
echo ============================================================================
echo.

REM Start Python ML Backend
echo [1/3] Starting Python ML Backend...
start "AR Mirror - Python ML" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate && python app.py --phase 0 --duration 0"

REM Wait a moment
timeout /t 5 /nobreak > nul

REM Start NestJS Backend
echo [2/3] Starting NestJS Backend...
cd /d "%~dp0backend"
start "AR Mirror - NestJS" cmd /k "npm run dev"

REM Wait a moment
timeout /t 3 /nobreak > nul

REM Instructions for React Frontend
echo [3/3] React Frontend Setup...
echo.
echo ============================================================================
echo NEXT STEPS:
echo ============================================================================
echo.
echo 1. Python ML Backend: http://localhost:5050 (video stream)
echo 2. NestJS API: http://localhost:3000
echo 3. Create React frontend:
echo    cd frontend
echo    npx create-react-app ar-mirror-ui
echo    cd ar-mirror-ui
echo    npm start
echo.
echo Or access the Python web interface directly at http://localhost:5050
echo.
echo Press any key to open Python ML interface in browser...
pause > nul
start http://localhost:5050

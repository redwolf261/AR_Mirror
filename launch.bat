@echo off
title Zyro AR Mirror Launcher
color 0B

echo.
echo  ████████████████████████████████████████████████████████
echo  █                                                      █
echo  █        ZYRO AR MIRROR  ^|  SYSTEM LAUNCH             █
echo  █                                                      █
echo  ████████████████████████████████████████████████████████
echo.

REM ── Resolve project root ────────────────────────────────────────────────────
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ── Select Python env (prefer ar, fallback .venv) ────────────────────────────
set "PYTHON_EXE="
set "ACTIVATE_BAT="
if exist "ar\Scripts\python.exe" (
    set "PYTHON_EXE=ar\Scripts\python.exe"
    set "ACTIVATE_BAT=ar\Scripts\activate.bat"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    set "ACTIVATE_BAT=.venv\Scripts\activate.bat"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] No Python environment found. Expected "ar" or ".venv".
    echo        Create one with: py -3.12 -m venv ar
    pause
    exit /b 1
)

REM ── Check Node / npm ─────────────────────────────────────────────────────────
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM ── Install Python share dependency if missing ────────────────────────────────
echo  [SETUP] Checking qrcode dependency...
%PYTHON_EXE% -c "import qrcode" >nul 2>&1
if errorlevel 1 (
    echo  [SETUP] Installing qrcode[pil] for QR share feature...
    %PYTHON_EXE% -m pip install "qrcode[pil]>=7.4" --quiet
)

REM ── Install frontend JS deps if needed ───────────────────────────────────────
if not exist "frontend\ar-mirror-ui\node_modules" (
    echo  [SETUP] Installing frontend dependencies...
    cd frontend\ar-mirror-ui
    call npm install --silent
    cd ..\..
)

echo.
echo  Starting services in separate windows...
echo.

REM ── Window 1: AR Mirror Python backend (app.py) ──────────────────────────────
echo  [1/2]  AR Backend   ^|  webcam + ML pipeline + API on :5051
start "AR-Backend" cmd /k "^
    title AR-Backend ^| port 5051 ^& ^
    color 0A ^& ^
    cd /d "%ROOT%" ^& ^
    call %ACTIVATE_BAT% ^& ^
    echo. ^& ^
    echo  AR MIRROR BACKEND — Python ML Pipeline ^& ^
    echo  ========================================= ^& ^
    echo  Web API  : http://localhost:5051 ^& ^
    echo  Stream   : http://localhost:5051/stream ^& ^
    echo  Share    : http://localhost:5051/api/share/generate ^& ^
    echo  Press Ctrl+C to stop. ^& ^
    echo. ^& ^
    %PYTHON_EXE% app.py --phase 2 --duration 0 & pause"

REM ── Window 2: React frontend (ar-mirror-ui) ───────────────────────────────────
echo  [2/2]  React UI     ^|  dev server on http://localhost:3001
start "AR-Frontend" cmd /k "^
    title AR-Frontend ^| port 3001 ^& ^
    color 0B ^& ^
    cd /d "%ROOT%\frontend\ar-mirror-ui" ^& ^
    echo. ^& ^
    echo  AR MIRROR FRONTEND — Vite Dev Server ^& ^
    echo  ======================================== ^& ^
    echo  UI URL   : http://localhost:3001 ^& ^
    echo  Backend  : http://localhost:5051 ^& ^
    echo  Press Ctrl+C to stop. ^& ^
    echo. ^& ^
    npm run dev"

REM ── Wait for backend to initialise ───────────────────────────────────────────
echo.
echo  Waiting for AR backend to initialise (8 seconds)...
timeout /t 8 /nobreak >nul

REM ── Open browser ─────────────────────────────────────────────────────────────
echo  Opening browser at http://localhost:3001
start "" "http://localhost:3001"

echo.
echo  ████████████████████████████████████████████████████████
echo  █  All services launched. Close sub-windows to stop.   █
echo  ████████████████████████████████████████████████████████
echo.
echo  AR Backend  →  http://localhost:5051
echo  AR UI       →  http://localhost:3001
echo  Share API   →  POST http://localhost:5051/api/share/generate
echo.
echo  This launcher window can be closed safely.
echo.
pause

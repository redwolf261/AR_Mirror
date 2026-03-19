@echo off
title AR Mirror Launcher
color 0B

echo.
echo  ████████████████████████████████████████████████████████
echo  █                                                      █
echo  █           AR MIRROR  ^|  SYSTEM LAUNCH               █
echo  █                                                      █
echo  ████████████████████████████████████████████████████████
echo.

REM ── Resolve project root (wherever this script lives) ──────────────────────
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ── Select Python env (prefer ar, fallback .venv) ─────────────────────────
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

REM ── Check Node / npm ────────────────────────────────────────────────────────
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM ── Install JS deps if needed ───────────────────────────────────────────────
if not exist "web-ui\node_modules" (
    echo [SETUP] Installing web-ui dependencies...
    cd web-ui
    call npm install --silent
    cd ..
)

echo.
echo  Starting services in separate windows...
echo.

REM ── Window 1: AR Mirror backend (app.py --phase 2) ─────────────────────────
echo  [1/2]  AR Backend   ^|  webcam + TPS pipeline + API on :5050
start "AR-Backend" cmd /k "^
    title AR-Backend ^| port 5050 & ^
    color 0A & ^
    cd /d "%ROOT%" & ^
    call %ACTIVATE_BAT% & ^
    echo. & ^
    echo  AR MIRROR BACKEND — Phase 2 Pipeline & ^
    echo  ======================================= & ^
    echo  Web API  : http://localhost:5050 & ^
    echo  Press Ctrl+C to stop. & ^
    echo. & ^
    %PYTHON_EXE% app.py --phase 2 --duration 0 & pause"

REM ── Window 2: React frontend (npm run dev) ──────────────────────────────────
echo  [2/2]  React UI     ^|  dev server on :3001 (or :3002 fallback)
start "AR-Frontend" cmd /k "^
    title AR-Frontend ^| port 3001 & ^
    color 0B & ^
    cd /d "%ROOT%\web-ui" & ^
    echo. & ^
    echo  AR MIRROR FRONTEND — Vite Dev Server & ^
    echo  ======================================= & ^
    echo  UI URL   : http://localhost:3001 & ^
    echo  Press Ctrl+C to stop. & ^
    echo. & ^
    npm run dev"

REM ── Wait for backend to be ready ────────────────────────────────────────────
echo.
echo  Waiting for backend to initialise...
timeout /t 5 /nobreak >nul

REM ── Open browser ────────────────────────────────────────────────────────────
echo  Opening browser at http://localhost:3001
start "" "http://localhost:3001"

echo.
echo  ████████████████████████████████████████████████████████
echo  █  All services launched.  Close sub-windows to stop.  █
echo  ████████████████████████████████████████████████████████
echo.
echo  AR Backend  →  http://localhost:5050
echo  AR UI       →  http://localhost:3001
echo.
echo  This launcher window can be closed safely.
echo.
pause

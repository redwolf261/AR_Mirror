# launch.ps1 — AR Mirror unified launcher
# Usage:  .\launch.ps1
# Starts backend (app.py --phase 2) + React dev server, then opens browser.
# Ctrl+C stops both processes cleanly.

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# ── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host "  █                                                      █" -ForegroundColor Cyan
Write-Host "  █           AR MIRROR  |  SYSTEM LAUNCH               █" -ForegroundColor Cyan
Write-Host "  █                                                      █" -ForegroundColor Cyan
Write-Host "  ████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
$PythonExe = $null
if (Test-Path "$Root\ar\Scripts\python.exe") {
    $PythonExe = "$Root\ar\Scripts\python.exe"
} elseif (Test-Path "$Root\.venv\Scripts\python.exe") {
    $PythonExe = "$Root\.venv\Scripts\python.exe"
}

if (-not $PythonExe) {
    Write-Host "  [ERROR] No Python environment found (expected ar or .venv)." -ForegroundColor Red
    Write-Host "  Run:  py -3.12 -m venv ar  and install dependencies" -ForegroundColor Yellow
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "  [ERROR] npm not found. Install Node.js from https://nodejs.org" -ForegroundColor Red
    exit 1
}

# Install JS deps if node_modules is missing
if (-not (Test-Path "$Root\web-ui\node_modules")) {
    Write-Host "  [SETUP] Installing web-ui npm dependencies..." -ForegroundColor Yellow
    Push-Location "$Root\web-ui"
    npm install --silent
    Pop-Location
}

Write-Host "  Starting services..." -ForegroundColor Cyan
Write-Host ""

# ── Python runtime ───────────────────────────────────────────────────────────
Write-Host "  Python runtime: $PythonExe" -ForegroundColor DarkGray

# ── Start AR Backend ─────────────────────────────────────────────────────────
Write-Host "  [1/2] AR Backend  — webcam + TPS + Flask API on :5050" -ForegroundColor Green
$backend = Start-Process -FilePath $PythonExe `
    -ArgumentList "app.py", "--phase", "2", "--duration", "0" `  # 0 = run until stopped
    -WorkingDirectory $Root `
    -PassThru `
    -WindowStyle Normal

# ── Start React Frontend ─────────────────────────────────────────────────────
Write-Host "  [2/2] React UI    — Vite dev server on :3001" -ForegroundColor Green
$frontend = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory "$Root\web-ui" `
    -PassThru `
    -WindowStyle Normal

Write-Host ""
Write-Host "  Waiting 6 s for services to initialise..." -ForegroundColor DarkGray
Start-Sleep -Seconds 6

# ── Open browser ─────────────────────────────────────────────────────────────
Write-Host "  Opening  http://localhost:3001  in default browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3001"

Write-Host ""
Write-Host "  ████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host "  █  Services running.  Press Ctrl+C here to stop all.   █" -ForegroundColor Cyan
Write-Host "  ████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host ""
Write-Host "  AR Backend  →  http://localhost:5050/api/state" -ForegroundColor White
Write-Host "  AR UI       →  http://localhost:3001" -ForegroundColor White
Write-Host ""

# ── Wait and clean up on Ctrl+C ─────────────────────────────────────────────
try {
    # Keep script alive; monitor both processes
    while ($true) {
        if ($backend.HasExited) {
            Write-Host "  [WARN] Backend process exited (code $($backend.ExitCode))." -ForegroundColor Yellow
        }
        if ($frontend.HasExited) {
            Write-Host "  [WARN] Frontend process exited (code $($frontend.ExitCode))." -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 3
    }
} finally {
    Write-Host ""
    Write-Host "  Stopping all services..." -ForegroundColor Yellow

    # Kill backend tree (kill child processes too)
    taskkill /PID $backend.Id /T /F 2>$null
    # Kill frontend (cmd /c npm run dev and its children)
    taskkill /PID $frontend.Id /T /F 2>$null

    Write-Host "  Done." -ForegroundColor Green
}

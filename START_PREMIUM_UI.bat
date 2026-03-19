@echo off
cls
echo.
echo ================================================================================
echo    AR MIRROR - PREMIUM APPLE-STYLE INTERFACE
echo ================================================================================
echo.
echo Starting full-stack system with:
echo   [1] Python ML Backend (Camera + AR Processing)
echo   [2] React Frontend (Apple-inspired UI)
echo.
echo ================================================================================
echo.

REM Check if Python backend is already running
tasklist /FI "WINDOWTITLE eq AR Mirror - Python ML*" 2>NUL | find /I "cmd.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] Python backend already running
) else (
    echo [1/2] Starting Python ML Backend on port 5050...
    start "AR Mirror - Python ML" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate && python app.py --phase 0 --duration 0"
    timeout /t 5 /nobreak > nul
)

REM Start React Frontend
echo [2/2] Starting React Frontend on port 3001...
cd /d "%~dp0frontend\ar-mirror-ui"
start "AR Mirror - React UI" cmd /k "npm start"

echo.
echo ================================================================================
echo Starting up...
echo ================================================================================
echo.
echo Wait for both terminals to show:
echo   - Python: "Ready to start [PHASE 0: ALPHA BLENDING]"
echo   - React:  "Compiled successfully!"
echo.
echo Then navigate to: http://localhost:3001
echo.
echo ================================================================================
echo.
echo Press any key to open frontend in browser (wait 10-15 seconds first)...
pause > nul

start http://localhost:3001

echo.
echo ✅ System launched!
echo.
echo Access Points:
echo   Frontend (Premium UI):  http://localhost:3001
echo   Backend (API):          http://localhost:5050
echo.
echo Press any key to exit this window...
pause > nul

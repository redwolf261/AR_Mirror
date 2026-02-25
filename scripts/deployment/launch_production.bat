@echo off
REM Launch Production AR Try-On System
echo ================================================
echo   AR TRY-ON PRODUCTION SYSTEM v2.0
echo ================================================
echo.
echo Starting interactive demo...
echo.
echo Controls:
echo   Left/Right Arrow : Cycle garments
echo   G                : Toggle overlay
echo   Q or ESC         : Quit
echo.
echo ================================================
echo.

python interactive_demo.py %*

echo.
echo Session complete!
pause

#!/usr/bin/env python3
"""
AR Mirror - Premium UI Demo Launcher
Starts both Python backend and React frontend
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

print("="*80)
print("AR MIRROR - PREMIUM APPLE-STYLE UI LAUNCHER")
print("="*80)
print()

# Verify we're in the right directory
if not Path("app.py").exists():
    print("[ERROR] Must run from AR Mirror root directory")
    sys.exit(1)

print("Starting full-stack system...")
print()
print("Components:")
print("  [1] Python ML Backend (port 5050)")
print("  [2] React Frontend (port 3001)")
print()
print("="*80)
print()

# Start Python backend
print("[1/2] Starting Python ML Backend...")
python_process = subprocess.Popen(
    [sys.executable, "app.py", "--phase", "0", "--duration", "0"],
    shell=True
)

# Wait for backend to initialize
print("         Waiting for backend to start (5 seconds)...")
time.sleep(5)

# Start React frontend
print("[2/2] Starting React Frontend...")
print("         This will open a new terminal window")
print("         Wait for 'Compiled successfully!' message")
print()

frontend_dir = Path("frontend/ar-mirror-ui")
if frontend_dir.exists():
    subprocess.Popen(
        "npm start",
        cwd=str(frontend_dir),
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )

    # Wait and open browser
    print("="*80)
    print("Starting up...")
    print("="*80)
    print()
    print("Wait 10-15 seconds for React to compile, then:")
    print()
    print("  Frontend (Premium UI):  http://localhost:3001")
    print("  Backend (API):          http://localhost:5050")
    print()
    print("="*80)
    print()
    print("Opening frontend in 15 seconds...")

    time.sleep(15)
    webbrowser.open("http://localhost:3001")

    print()
    print("[OK] System running!")
    print()
    print("Controls:")
    print("  - Frontend UI: Click toggles and garments")
    print("  - Camera window: Use arrow keys")
    print("  - Press Ctrl+C here to stop Python backend")
    print()

    try:
        python_process.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down Python backend...")
        python_process.terminate()
        print("[OK] Stopped")
else:
    print("[ERROR] Frontend not found at: frontend/ar-mirror-ui")
    print("        Run 'npx create-react-app frontend/ar-mirror-ui' first")
    python_process.terminate()
    sys.exit(1)

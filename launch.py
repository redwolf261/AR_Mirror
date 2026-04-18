#!/usr/bin/env python3
"""
Unified AR Mirror Launcher
Starts Python ML backend with integrated web UI
"""
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

print("="*80)
print("🚀 AR MIRROR - PRODUCTION SYSTEM LAUNCHER")
print("="*80)
print()

# Check if we're in the right directory
if not Path("app.py").exists():
    print("❌ Error: Must run from AR Mirror root directory")
    sys.exit(1)

print("Starting AR Mirror...")
print()
print("Components:")
print("  ✅ Python ML Backend (Camera + AR Processing)")
print("  ✅ Integrated Web UI Server (port 5050)")
print("  ✅ Data Flywheel (Session Logging)")
print()
print("Access Points:")
print("  🌐 Web Interface: http://localhost:5050")
print("  📹 Camera View: Direct OpenCV window")
print()
print("="*80)
print()

# Start the main app
try:
    print("Launching...")
    proc = subprocess.Popen([
        sys.executable, "app.py",
        "--phase", "0",  # Phase 0 for speed
        "--duration", "0"  # Run indefinitely
    ])

    # Wait a moment for server to start
    time.sleep(2)

    # Open browser
    print("\n🌐 Opening web interface...")
    webbrowser.open("http://localhost:5050")

    print("\n✅ System running!")
    print("\nControls:")
    print("  • Browser UI: Control garments via web interface")
    print("  • Camera Window: Use arrow keys to change garments")
    print("  • Press 'q' in camera window to quit")
    print("\nPress Ctrl+C to stop...")

    # Wait for process
    proc.wait()

except KeyboardInterrupt:
    print("\n\n⏹️  Shutting down...")
    proc.terminate()
    print("✅ Stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)

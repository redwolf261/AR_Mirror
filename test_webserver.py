#!/usr/bin/env python3
"""Quick test to verify web server works"""

import time
import cv2
import numpy as np
from web_server import WebServer

print("Starting test web server...")
ws = WebServer(port=5050)

# Register dummy garment list
ws.register_garment_list(lambda: ["test-shirt-1.jpg", "test-shirt-2.jpg"])

# Start server
ws.start()
print("[OK] Web server started at http://localhost:5050")
print("     Frontend should connect at http://localhost:3001")

# Generate test frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] No camera, using test pattern")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "AR Mirror - Test Mode", (400, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
else:
    print("[OK] Camera opened")

print("\nRunning indefinitely (Ctrl+C to stop)...")
print("Check http://localhost:5050/stream for video")
print("Check http://localhost:5050/api/state for state")

i = 0
while True:  # Run indefinitely
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"Test Frame {i}", (500, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Push frame to web server
    ws.push_frame(frame)

    # Push state
    measurements_data = {
        "shoulder_cm": 42.5,
        "chest_cm": 38.0,
        "waist_cm": 32.5,
        "torso_cm": 65.0,
        "size": "M"
    }
    ws.push_state(5.0, "test-shirt-1.jpg", measurements_data)

    if i % 50 == 0:  # Debug print every 10 seconds
        print(f"[{i}] Pushed measurements: {measurements_data}")

    time.sleep(0.2)  # 5 FPS
    i += 1

if cap.isOpened():
    cap.release()

print("\nTest complete!")

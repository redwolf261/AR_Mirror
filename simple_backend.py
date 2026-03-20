#!/usr/bin/env python3
"""Minimal backend with test measurements"""

from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import time
import threading

app = Flask(__name__)
CORS(app)

# Test state
state = {
    "fps": 5.0,
    "garment": "test-shirt-1.jpg",
    "measurements": {
        "shoulder_cm": 42.5,
        "chest_cm": 38.0,
        "waist_cm": 32.5,
        "torso_cm": 65.0,
        "size": "M"
    },
    "ts": time.time()
}

latest_frame = None
frame_lock = threading.Lock()

def generate_frames():
    """Generate test video frames"""
    cap = cv2.VideoCapture(0)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = create_test_frame()
        else:
            frame = create_test_frame()

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with frame_lock:
            global latest_frame
            latest_frame = buffer.tobytes()

        time.sleep(0.1)

def create_test_frame():
    """Create test pattern frame"""
    frame = cv2.imread(r"dataset\train\cloth\00001_00.jpg")
    if frame is None:
        frame = cv2.cvtColor(cv2.imread(r"00001_00.jpg"), cv2.COLOR_BGR2RGB) if cv2.imread(r"00001_00.jpg") is not None else None
    if frame is None:
        frame = (np.random.rand(720, 1280, 3) * 255).astype('uint8')
    return cv2.resize(frame, (1280, 720))

@app.route('/stream')
def stream():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    state['ts'] = time.time()
    return jsonify(state)

@app.route('/api/garments')
def api_garments():
    return jsonify({"garments": ["00001_00.jpg", "00002_00.jpg", "00003_00.jpg"]})

if __name__ == '__main__':
    print("Starting simple backend with test measurements...")
    print("[OK] Web server at http://localhost:5050")
    print("     Frontend connects at http://localhost:3000")

    # Start frame generator
    import numpy as np
    thread = threading.Thread(target=generate_frames, daemon=True)
    thread.start()

    # Run Flask
    app.run(host='0.0.0.0', port=5050, threaded=True, use_reloader=False)

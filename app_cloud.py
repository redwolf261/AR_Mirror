#!/usr/bin/env python3
"""
app_cloud.py - Simplified AR Mirror for free cloud deployment

Differences from local version:
- No camera access required
- Frontend sends images via API
- Optimized for Render.com free tier
- Smaller memory footprint
"""

import os
import io
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://*.netlify.app",
    "https://*.vercel.app",
    "https://*.onrender.com"
])

# Initialize ML components - lazy load to save memory
body_fitter = None
size_engine = None

def init_ml_components():
    """Lazy initialization of ML components"""
    global body_fitter, size_engine

    if body_fitter is None:
        try:
            from src.core.body_aware_fitter import BodyAwareFitter
            body_fitter = BodyAwareFitter()
            log.info("✅ Body detection initialized")
        except Exception as e:
            log.error(f"Failed to initialize body detection: {e}")

    if size_engine is None:
        try:
            from src.core.size_recommendation import get_size_recommendation
            size_engine = get_size_recommendation
            log.info("✅ Size recommendation engine initialized")
        except Exception as e:
            log.error(f"Failed to initialize size engine: {e}")

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'service': 'AR Mirror Backend',
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for body measurements"""
    try:
        # Initialize ML components if needed
        init_ml_components()

        if not body_fitter:
            return jsonify({'error': 'Body detection not available'}), 500

        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)

        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Extract body measurements
        measurements = body_fitter.extract_body_measurements(cv_image)

        if measurements:
            # Get size recommendation
            size_data = {}
            if size_engine:
                try:
                    size_rec = size_engine(measurements)
                    size_data = {
                        'size_recommendation': size_rec.get('recommended_size', 'Unknown'),
                        'size_confidence': size_rec.get('confidence', 0.0),
                        'size_alternatives': size_rec.get('all_sizes', {})
                    }
                except Exception as e:
                    log.error(f"Size recommendation error: {e}")

            result = {
                'success': True,
                'measurements': {
                    'shoulder_width_cm': measurements.get('shoulder_width_cm', 0),
                    'chest_cm': measurements.get('chest_cm', 0),
                    'torso_length_cm': measurements.get('torso_length_cm', 0),
                    'confidence': measurements.get('confidence', 0)
                },
                **size_data
            }

            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'No body detected in image'
            })

    except Exception as e:
        log.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/garments', methods=['GET'])
def get_garments():
    """Get available garments list"""
    # Simple mock data for free deployment
    garments = [
        'shirt_01.jpg', 'shirt_02.jpg', 'dress_01.jpg',
        'jacket_01.jpg', 'top_01.jpg'
    ]
    return jsonify({'garments': garments})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV') == 'development'

    log.info(f"🚀 Starting AR Mirror Backend on port {port}")
    log.info(f"📝 Debug mode: {debug}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
"""
Flask Backend for Neural Network from Scratch
==============================================
Serves predictions using the NumPy from-scratch implementation
with PyTorch-trained weights.
"""

import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
from main import NeuralNetwork

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per minute"])


def _allowed_origins():
    raw = os.getenv("CORS_ORIGINS", "")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "https://sidvalecha.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ]


CORS(app, origins=_allowed_origins())

# Load pre-trained NumPy from-scratch model
print("Loading neural network model...")
model = NeuralNetwork([784, 128, 64, 10])
model.load_weights('weights/model_weights.npz')
print("Model loaded successfully!")


@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'message': 'Neural Network from Scratch API',
        'endpoints': {
            '/predict': 'POST - Send pixel array to get digit prediction',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        if 'pixels' not in data:
            return jsonify({'error': 'Missing "pixels" field'}), 400

        pixels = data['pixels']

        if not isinstance(pixels, list):
            return jsonify({'error': 'Pixels must be an array'}), 400

        if len(pixels) != 784:
            return jsonify({'error': 'Expected 784 pixels'}), 400

        # Convert to numpy array and reshape
        try:
            pixels = np.array(pixels, dtype='float32')
        except (TypeError, ValueError):
            return jsonify({'error': 'Pixel values must be numeric'}), 400

        if pixels.shape != (784,):
            return jsonify({'error': 'Pixels must be a flat array of length 784'}), 400

        if not np.isfinite(pixels).all():
            return jsonify({'error': 'Pixel values must be finite numbers'}), 400

        if (pixels < 0).any() or (pixels > 255).any():
            return jsonify({'error': 'Pixel values must be between 0 and 255'}), 400

        pixels = pixels.reshape(1, 784)

        # Normalize to [0, 1] if not already
        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        # Run inference using NumPy from-scratch implementation
        probs = model.forward(pixels)[0]

        # Get prediction
        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit])

        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probs.tolist()
        })

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)

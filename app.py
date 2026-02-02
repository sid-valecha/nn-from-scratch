"""
Flask Backend for Neural Network from Scratch
==============================================
Serves predictions using the NumPy from-scratch implementation
with PyTorch-trained weights.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from main import NeuralNetwork

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Netlify frontend

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
def predict():
    try:
        data = request.json

        if 'pixels' not in data:
            return jsonify({'error': 'Missing "pixels" field'}), 400

        pixels = data['pixels']

        if len(pixels) != 784:
            return jsonify({'error': f'Expected 784 pixels, got {len(pixels)}'}), 400

        # Convert to numpy array and reshape
        pixels = np.array(pixels, dtype='float32').reshape(1, 784)

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

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

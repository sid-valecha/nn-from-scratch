# Neural Network from Scratch

A fully connected feedforward neural network built **from scratch using only NumPy** - no TensorFlow, PyTorch, or other deep learning frameworks for inference.

**[Live Demo](https://sidvalecha.com/nn-from-scratch)** - Draw digits and watch the neural network classify them in real-time. Inspired by [Google's Quick, Draw!](https://quickdraw.withgoogle.com/)

## Overview

This project demonstrates a deep understanding of neural network fundamentals by implementing:

- **Forward propagation** with matrix operations
- **Backpropagation** using the chain rule
- **Activation functions** (ReLU, Softmax)
- **Loss computation** (Cross-entropy)
- **Weight initialization** (He initialization)
- **Mini-batch SGD with momentum**

## Architecture

```
Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax
```

- **Input**: 28×28 grayscale images flattened to 784 features
- **Hidden layers**: 128 and 64 neurons with ReLU activation
- **Output**: 10 classes (digits 0-9) with Softmax

## Training Approach

PyTorch is used **only for training** to leverage GPU acceleration and the Adam optimizer. The trained weights are then exported to NumPy format and loaded into the from-scratch implementation for inference.

```
PyTorch (training) → Export weights → NumPy (inference)
```

This demonstrates:
1. Understanding of how to transfer learned parameters between frameworks
2. The from-scratch implementation is fully functional for inference
3. Best of both worlds: fast training + educational inference code

## Accuracy

- **97.83%** on MNIST test set (10,000 images)
- Trained on 60,000 MNIST training images
- 20 epochs with Adam optimizer (lr=0.001)

## Project Structure

```
nn-from-scratch/
├── main.py              # NumPy from-scratch neural network
├── train_pytorch.py     # PyTorch training script
├── export_weights.py    # Weight conversion (PyTorch → NumPy)
├── app.py               # Flask API for web demo
├── weights/
│   ├── model_weights.npz    # NumPy weights for inference
│   └── pytorch_model.pth    # PyTorch checkpoint
└── requirements.txt
```

## Usage

### Run the from-scratch implementation

```bash
python main.py --epochs 10 --batch_size 128 --lr 0.01
```

### Train with PyTorch (faster, GPU support)

```bash
python train_pytorch.py
python export_weights.py
```

### Run the web API locally

```bash
pip install -r requirements.txt
python app.py
```

## Web Demo

The live demo at [sidvalecha.com/nn-from-scratch](https://sidvalecha.com/nn-from-scratch) features:

- Drawing canvas for digit input
- Real-time classification using the NumPy from-scratch model
- Probability distribution visualization for all 10 digits

**Stack**: Flask backend (Railway) + Static frontend (Netlify)

## Key Implementation Details

### From-Scratch Components (`main.py`)

```python
# Forward pass through a dense layer
def forward(self, x):
    self.input = x  # Cache for backprop
    return x @ self.W + self.b

# Backpropagation with momentum SGD
def backward(self, grad_out, lr, momentum):
    grad_W = self.input.T @ grad_out
    grad_b = grad_out.sum(axis=0, keepdims=True)
    grad_input = grad_out @ self.W.T

    # Momentum update
    self.v_W = momentum * self.v_W - lr * grad_W
    self.v_b = momentum * self.v_b - lr * grad_b
    self.W += self.v_W
    self.b += self.v_b

    return grad_input
```

### Weight Transfer

PyTorch stores weights as `(out_features, in_features)`, while the NumPy implementation uses `(in_features, out_features)`. The export script handles the transpose:

```python
numpy_W = pytorch_weight.T.numpy()
```

## Requirements

**For inference (NumPy only):**
- numpy

**For training:**
- torch
- torchvision

**For web API:**
- flask
- flask-cors
- gunicorn

## License

MIT

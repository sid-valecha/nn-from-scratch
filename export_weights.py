"""
Export PyTorch Weights to NumPy Format
======================================
Converts PyTorch model weights to NumPy .npz format for use
with the from-scratch NumPy implementation.

CRITICAL: PyTorch Linear layers use (out_features, in_features),
while NumPy implementation uses (in_features, out_features).
Must transpose weight matrices!
"""

import torch
import numpy as np
from pathlib import Path

# Import the PyTorch model architecture
from train_pytorch import MLP


def export_weights(pytorch_path: str = 'weights/pytorch_model.pth',
                   numpy_path: str = 'weights/model_weights.npz'):
    """Convert PyTorch weights to NumPy format."""

    # Load PyTorch model
    model = MLP()
    model.load_state_dict(torch.load(pytorch_path, map_location='cpu', weights_only=True))
    model.eval()

    state_dict = model.state_dict()

    # Map PyTorch layer names to our layer indices
    # layers.0 = first Linear (784 -> 128)
    # layers.2 = second Linear (128 -> 64)
    # layers.4 = third Linear (64 -> 10)
    layer_mapping = {
        0: ('layers.0.weight', 'layers.0.bias'),
        1: ('layers.2.weight', 'layers.2.bias'),
        2: ('layers.4.weight', 'layers.4.bias'),
    }

    weights = {}
    for layer_idx, (weight_key, bias_key) in layer_mapping.items():
        # CRITICAL: Transpose weights!
        # PyTorch: (out_features, in_features)
        # NumPy:   (in_features, out_features)
        W = state_dict[weight_key].T.numpy().astype('float32')
        b = state_dict[bias_key].numpy().reshape(1, -1).astype('float32')

        weights[f'layer{layer_idx}_W'] = W
        weights[f'layer{layer_idx}_b'] = b

        print(f"Layer {layer_idx}: W shape {W.shape}, b shape {b.shape}")

    # Save as compressed NumPy archive
    np.savez_compressed(numpy_path, **weights)
    print(f"\nWeights saved to {numpy_path}")

    return weights


def validate_conversion():
    """Validate that NumPy model produces same predictions as PyTorch."""
    import sys
    sys.path.insert(0, '.')
    from main import NeuralNetwork
    from torchvision import datasets, transforms

    # Load test data using torchvision (already downloaded)
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=False,
                                   transform=transforms.ToTensor())

    # Get raw pixel data (no normalization - matches how we'll use it in production)
    x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
    y_test = test_dataset.targets.numpy()

    # PyTorch predictions (without normalization to match NumPy)
    model_pt = MLP()
    model_pt.load_state_dict(torch.load('weights/pytorch_model.pth', map_location='cpu', weights_only=True))
    model_pt.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x_test[:100], dtype=torch.float32)
        pt_logits = model_pt(x_tensor)
        pt_preds = pt_logits.argmax(dim=1).numpy()

    # NumPy predictions (using from-scratch implementation)
    model_np = NeuralNetwork([784, 128, 64, 10])
    model_np.load_weights('weights/model_weights.npz')
    np_preds = model_np.predict(x_test[:100])

    # Compare
    matches = np.sum(pt_preds == np_preds)
    print(f"\nValidation: {matches}/100 predictions match between PyTorch and NumPy")

    if matches == 100:
        print("SUCCESS: Perfect match!")
    else:
        print(f"WARNING: {100 - matches} predictions differ")
        # Show some differences
        diff_idx = np.where(pt_preds != np_preds)[0][:5]
        for idx in diff_idx:
            print(f"  Sample {idx}: PyTorch={pt_preds[idx]}, NumPy={np_preds[idx]}, True={y_test[idx]}")

    # Also check full test set accuracy
    np_full_preds = model_np.predict(x_test)
    np_acc = np.mean(np_full_preds == y_test) * 100
    print(f"\nNumPy from-scratch accuracy on full test set: {np_acc:.2f}%")


if __name__ == '__main__':
    export_weights()
    print("\n" + "="*50)
    print("Running validation...")
    print("="*50)
    validate_conversion()

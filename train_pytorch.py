"""
PyTorch Training Script for MNIST
=================================
Trains a PyTorch MLP on MNIST, then exports weights for use with
the NumPy from-scratch implementation.

Architecture matches main.py: 784 → 128 → 64 → 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


class MLP(nn.Module):
    """MLP matching the from-scratch architecture: 784 → 128 → 64 → 10"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        return self.layers(x)


def train():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data - no normalization for simpler frontend integration
    transform = transforms.ToTensor()  # Just converts to [0,1] range

    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # Model
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 20
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100. * correct / total
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Test Acc: {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            Path('weights').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'weights/pytorch_model.pth')
            print(f"  → Saved new best model (acc: {acc:.2f}%)")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to weights/pytorch_model.pth")

    return model


if __name__ == '__main__':
    train()

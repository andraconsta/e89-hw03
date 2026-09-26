"""Step 2: pick a device and define the ImageClassifier MLP and its loss.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Follows "Building an Image Classifier with PyTorch" in Geron, Hands-On
Machine Learning, chapter 10 (10_neural_nets_with_pytorch.ipynb).
"""

import torch
import torch.nn as nn

from load_data import train_loader

# Use an NVIDIA GPU (cuda) or Apple GPU (mps) if available, else the CPU.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# The classifier is a multilayer perceptron:
# Flatten turns each [1, 28, 28] image into a 784-value vector, two hidden
# layers (300 and 100 units, ReLU) learn features, and the output layer
# returns one raw score (logit) per class. There is no softmax at the end:
# CrossEntropyLoss applies log-softmax internally, which is more stable.
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


# Build the model (784 -> 300 -> 100 -> 10) on the chosen device.
# Seeding makes the initial weights reproducible, as in Geron.
torch.manual_seed(42)
model = ImageClassifier(n_inputs=28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)

# Loss for multiclass classification with integer class labels.
xentropy = nn.CrossEntropyLoss()

if __name__ == "__main__":
    print(f"Device: {device}")
    print(model)

    # Expected: 784*300+300 + 300*100+100 + 100*10+10 = 266,610 parameters.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # One forward pass on a batch: output [32, 10] (one logit per class).
    # An untrained model should give a loss near ln(10) = 2.30 (random guess).
    X_batch, y_batch = next(iter(train_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        y_pred = model(X_batch)
    print(f"Output shape: {y_pred.shape}")
    print(f"Initial loss: {xentropy(y_pred, y_batch).item():.4f}")

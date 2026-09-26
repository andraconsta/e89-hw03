"""Define the FashionMNIST image classifier (an MLP) and pick a device.

Follows "Building an Image Classifier with PyTorch" > "Building the
Classifier" in Geron, Hands-On Machine Learning, chapter 10.
"""

import torch
import torch.nn as nn

# Step 1: pick the fastest available device: an NVIDIA GPU (cuda), an Apple
# GPU (mps), or the CPU as a fallback.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Step 2: define the classifier as a multilayer perceptron.
# Flatten turns each [1, 28, 28] image into a 784-value vector, two hidden
# layers with ReLU activations learn features, and the output layer produces
# one raw score (logit) per class. No softmax here: CrossEntropyLoss applies
# it internally, which is more numerically stable.
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


if __name__ == "__main__":
    from load_data import train_loader

    # Step 3: build the model with Geron's sizes (784 -> 300 -> 100 -> 10) and
    # move it to the device. Seeding makes the initial weights reproducible.
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(device)

    # Step 4: the loss for multiclass classification with integer labels.
    xentropy = nn.CrossEntropyLoss()

    # Step 5: sanity-check the architecture and parameter count
    # (784*300+300 + 300*100+100 + 100*10+10 = 266,610).
    print(f"Device: {device}")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")

    # Step 6: run one batch through the untrained model to confirm the output
    # shape is [32, 10] (one logit per class) and the loss is about
    # ln(10) = 2.30, what random guessing over 10 classes gives.
    X_batch, y_batch = next(iter(train_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        logits = model(X_batch)
    print(f"Logits shape: {logits.shape}")
    print(f"Initial loss: {xentropy(logits, y_batch).item():.4f}")

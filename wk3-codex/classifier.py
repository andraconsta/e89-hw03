"""Define Geron's FashionMNIST MLP and choose an available compute device."""
import torch
import torch.nn as nn

# Prefer NVIDIA CUDA, then Apple MPS, and otherwise keep execution on the CPU.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Flatten 28x28 grayscale images, learn two hidden representations, and output 10 logits.
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
    # Import one training batch, seed initialization, and build the chapter-sized model.
    from data import train_loader
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300,
                            n_hidden2=100, n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()

    # Print the model, parameter count, logits shape, and initial loss for validation.
    print(f"Device: {device}")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    X_batch, y_batch = next(iter(train_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        logits = model(X_batch)
    print(f"Logits shape: {logits.shape}")
    print(f"Initial loss: {xentropy(logits, y_batch).item():.4f}")

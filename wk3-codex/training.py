"""Train and evaluate the FashionMNIST classifier with Geron's helpers."""
import json
import torch
import torch.nn as nn
import torchmetrics
from data import train_loader, valid_loader
from classifier import ImageClassifier, device

# Evaluate a model on a complete data loader without computing gradients.
def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

# Train with mini-batches, recording average loss and train/validation accuracy each epoch.
def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train accuracy: {history['train_metrics'][-1]:.4f}, "
              f"validation accuracy: {history['valid_metrics'][-1]:.4f}")
    return history

if __name__ == "__main__":
    # Seed and initialize the Geron model, cross-entropy objective, SGD, and accuracy metric.
    n_epochs = 20
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300,
                            n_hidden2=100, n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    # Train for all epochs and preserve model weights and the complete metric history.
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs)
    torch.save(model.state_dict(), "model_weights.pt")
    with open("train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Saved model_weights.pt and train_history.json")

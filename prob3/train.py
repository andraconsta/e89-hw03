"""Step 3: train the ImageClassifier with SGD and track accuracy.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Uses evaluate_tm() and train2() from "Building an Image Classifier with
PyTorch" in Geron, Hands-On Machine Learning, chapter 10.
Saves the weights to fashion_mnist_mlp.pt and the history to history.json.
"""

import json

import torch
import torchmetrics

from load_data import train_loader, valid_loader
from model import device, model, xentropy


# Evaluate a model on a whole DataLoader with a torchmetrics metric.
# The metric accumulates counts over all batches, so the result is exact
# for the full dataset (not an average of per-batch values).
def evaluate_tm(model, data_loader, metric):
    model.eval()        # evaluation mode (no dropout/batch-norm updates)
    metric.reset()      # clear state from any earlier call
    with torch.no_grad():   # no gradients needed during evaluation
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


# Training loop: for each epoch, run mini-batch gradient descent on the
# training set, then measure the metric on the validation set.
# Returns a history dict of per-epoch train loss, train metric, valid metric.
def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()   # evaluate_tm() switches to eval mode, switch back
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)             # forward pass (logits)
            loss = criterion(y_pred, y_batch)   # loss vs. true labels
            total_loss += loss.item()
            loss.backward()                     # backprop: gradients
            optimizer.step()                    # gradient descent step
            optimizer.zero_grad()               # reset gradients
            metric.update(y_pred, y_batch)      # running train accuracy
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history


if __name__ == "__main__":
    # Plain SGD with learning rate 0.1, and multiclass accuracy as the metric.
    n_epochs = 20
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    # Train for 20 epochs.
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs)

    # Save the trained weights and the history for the next scripts.
    torch.save(model.state_dict(), "fashion_mnist_mlp.pt")
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved fashion_mnist_mlp.pt and history.json")

"""Train the FashionMNIST ImageClassifier and record its learning history.

Follows "Building an Image Classifier with PyTorch" in Geron, Hands-On Machine
Learning, chapter 10, reusing the chapter's evaluate_tm() and train2().
"""

import json

import torch
import torch.nn as nn
import torchmetrics

from load_data import train_loader, valid_loader
from model import ImageClassifier, device


# Step 1: evaluation helper. Runs the model over a whole DataLoader and returns
# one metric value for the full dataset (not an average of per-batch values).
def evaluate_tm(model, data_loader, metric):
    model.eval()     # evaluation mode (matters for dropout/batch norm layers)
    metric.reset()   # clear any state left over from a previous call
    with torch.no_grad():  # no gradients needed, saves memory and time
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # accumulate counts batch by batch
    return metric.compute()  # final metric over every sample seen


# Step 2: training loop. For each epoch it runs mini-batch gradient descent
# over the training set, then measures accuracy on the validation set.
# Returns a history dict of per-epoch training loss, training metric and
# validation metric, which is what the learning curves are plotted from.
def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            # evaluate_tm() switches to eval mode, so switch back every batch
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)             # forward pass: logits
            loss = criterion(y_pred, y_batch)   # compare with true labels
            total_loss += loss.item()
            loss.backward()                     # backprop: compute gradients
            optimizer.step()                    # update the weights
            optimizer.zero_grad()               # reset gradients for next batch
            metric.update(y_pred, y_batch)      # running training accuracy
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
    n_epochs = 20

    # Step 3: build the model and loss exactly as in model.py. Seeding makes
    # the initial weights, and so the whole run, reproducible.
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()

    # Step 4: plain stochastic gradient descent with learning rate 0.1, and
    # multiclass accuracy (fraction of images whose top logit is the true class)
    # as the metric to track.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    # Step 5: train for 20 epochs.
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs)

    # Step 6: save the trained weights and the history so later scripts can
    # evaluate the model and plot the learning curves without retraining.
    torch.save(model.state_dict(), "fashion_mnist_mlp.pt")
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved fashion_mnist_mlp.pt and history.json")

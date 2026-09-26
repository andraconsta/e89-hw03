"""Step 4: plot the training accuracy per epoch from history.json.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Reads the history saved by train.py and saves training_accuracy.png.
"""

import json

import matplotlib.pyplot as plt

# Load the per-epoch history written by train.py.
with open("history.json") as f:
    history = json.load(f)

# Epoch numbers 1..n for the x-axis.
train_acc = history["train_metrics"]
epochs = range(1, len(train_acc) + 1)

# Line plot of training accuracy, with the final value labeled.
plt.figure(figsize=(7, 4))
plt.plot(epochs, train_acc, "b.-", label="Training accuracy")
plt.annotate(f"{train_acc[-1]:.4f}", (epochs[-1], train_acc[-1]),
             textcoords="offset points", xytext=(-10, -15), ha="center")
plt.xticks(list(epochs))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FashionMNIST MLP: training accuracy per epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure for the report.
plt.savefig("training_accuracy.png", dpi=150)
print(f"Epochs: {len(train_acc)}, first: {train_acc[0]:.4f}, "
      f"last: {train_acc[-1]:.4f}")
print("Saved training_accuracy.png")

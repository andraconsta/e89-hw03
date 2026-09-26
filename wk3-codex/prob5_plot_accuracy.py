"""Plot the training accuracy per epoch of the FashionMNIST ImageClassifier.

Reads history.json, which train.py writes after training, so the model does
not need to be retrained to draw the plot.
"""
# Resolve paths from the script location so the project can be run from any working directory.
from pathlib import Path
import os
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import json

import matplotlib.pyplot as plt
import numpy as np

# Step 1: load the per-epoch history saved by train.py. It holds three lists
# with one value per epoch: train_losses, train_metrics and valid_metrics.
with open("history_prob5.json") as f:
    history = json.load(f)

# Step 2: pull out the training accuracy and number the epochs from 1, not 0.
train_acc = np.array(history["train_metrics"])
epochs = np.arange(1, len(train_acc) + 1)

# Step 3: draw the curve, one marker per epoch.
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, "o-", label="Training accuracy")

# Step 4: label the final value so the end point can be read off directly.
plt.annotate(f"{train_acc[-1]:.4f}", xy=(epochs[-1], train_acc[-1]),
             xytext=(-10, -18), textcoords="offset points", ha="center")

# Step 5: axes, title and grid. Integer ticks, since epochs are whole numbers.
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FashionMNIST ImageClassifier: training accuracy per epoch")
plt.xticks(epochs)
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()

# Step 6: save the figure as a PNG (for the report) and show it on screen.
plt.savefig("training_accuracy_prob5.png", dpi=150)
print("Saved training_accuracy.png")
plt.show()

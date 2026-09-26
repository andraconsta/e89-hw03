"""Plot training accuracy across the 20 recorded epochs."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the training accuracies saved by training.py.
with open("train_history.json", encoding="utf-8") as f:
    history = json.load(f)
train_accuracy = np.asarray(history["train_metrics"])
epochs = np.arange(1, len(train_accuracy) + 1)

# Draw the learning curve, annotate its final value, and label each epoch.
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracy, "o-", label="Training accuracy")
plt.annotate(f"{train_accuracy[-1]:.4f}",
             xy=(epochs[-1], train_accuracy[-1]),
             xytext=(-10, -18), textcoords="offset points", ha="center")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FashionMNIST ImageClassifier: training accuracy per epoch")
plt.xticks(epochs)
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()

# Save the requested figure and display it for inspection.
plt.savefig("accuracy_plot.png", dpi=150)
print("Saved accuracy_plot.png")
plt.show()

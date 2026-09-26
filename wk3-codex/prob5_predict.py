"""Use the trained ImageClassifier to predict the class of 3 validation images.

Follows the prediction step of "Building an Image Classifier with PyTorch" in
Geron, Hands-On Machine Learning, chapter 10. Loads the weights saved by
train.py, so run that first.
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


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from prob5_load_data import train_and_valid_data, valid_loader
from prob5_model import ImageClassifier, device

# Step 1: rebuild the same architecture and load the trained weights into it.
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
model.load_state_dict(torch.load("fashion_mnist_mlp_prob5.pt", map_location=device))

# Step 2: switch to evaluation mode for inference.
model.eval()

# Step 3: take the first 3 images of the first validation batch. The
# validation loader is not shuffled, so these are the same 3 images every run.
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]

# Step 4: run the model without tracking gradients. The output is one logit
# (raw score) per class for each image.
with torch.no_grad():
    y_pred_logits = model(X_new)

# Step 5: the predicted class is the index of the largest logit. Softmax turns
# the logits into probabilities, giving the model's confidence in that class.
y_pred = y_pred_logits.argmax(dim=1)
y_proba = F.softmax(y_pred_logits, dim=1).cpu()

# Step 6: map class indices to names (e.g. 0 -> "T-shirt/top") and compare
# each prediction with the true label.
class_names = train_and_valid_data.classes
print("Predicted class indices:", y_pred.tolist())
print("Predicted class names:  ", [class_names[i] for i in y_pred])
print("True class names:       ", [class_names[i] for i in y_new])
print()
for i, (pred, true) in enumerate(zip(y_pred.tolist(), y_new.tolist())):
    verdict = "correct" if pred == true else "WRONG"
    print(f"Image {i + 1}: predicted {class_names[pred]} "
          f"({y_proba[i, pred]:.1%} confidence), "
          f"true {class_names[true]} -> {verdict}")

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
for i, ax in enumerate(axes):
    pred, true = y_pred[i].item(), y_new[i].item()
    ax.imshow(X_new[i].cpu().squeeze(), cmap="binary")
    ax.set_title(f"Pred: {class_names[pred]} ({y_proba[i, pred]:.0%})\n"
                 f"True: {class_names[true]}",
                 color="green" if pred == true else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

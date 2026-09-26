"""Step 5: load the trained weights and classify 3 validation images.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Follows the prediction code in "Building an Image Classifier with PyTorch"
in Geron, Hands-On Machine Learning, chapter 10.
"""

import torch
import torch.nn.functional as F

from load_data import train_and_valid_data, valid_loader
from model import device, model

# Load the weights saved by train.py into the ImageClassifier.
model.load_state_dict(torch.load("fashion_mnist_mlp.pt", map_location=device))

# Take the first 3 images of the (unshuffled) validation set.
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]

# Predict: the class with the highest logit wins. Softmax turns the logits
# into probabilities, so the winning probability is the model's confidence.
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()
y_proba = F.softmax(y_pred_logits, dim=1).cpu()

# Print the predicted class, confidence, and true class for each image.
class_names = train_and_valid_data.classes
for i in range(3):
    pred, true = class_names[y_pred[i]], class_names[y_new[i]]
    status = "correct" if y_pred[i] == y_new[i] else "wrong"
    print(f"Image {i + 1}: predicted {pred} ({y_proba[i, y_pred[i]]:.1%}), "
          f"true {true} ({status})")

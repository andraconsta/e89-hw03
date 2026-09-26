"""Predict FashionMNIST classes for three validation images."""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import train_and_valid_data, valid_loader
from classifier import ImageClassifier, device

# Rebuild the trained architecture, load weights, and enable inference mode.
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300,
                        n_hidden2=100, n_classes=10).to(device)
model.load_state_dict(torch.load("model_weights.pt", map_location=device))
model.eval()

# Take the first three validation images and predict without gradient tracking.
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)
y_proba = F.softmax(y_pred_logits, dim=1).cpu()
class_names = train_and_valid_data.classes

# Print each predicted class, the matching true class, and predicted confidence.
print("Predicted class names:", [class_names[i] for i in y_pred.tolist()])
print("True class names:     ", [class_names[i] for i in y_new.tolist()])
for i, (pred, true) in enumerate(zip(y_pred.tolist(), y_new.tolist())):
    verdict = "correct" if pred == true else "WRONG"
    print(f"Image {i + 1}: predicted {class_names[pred]} "
          f"({y_proba[i, pred]:.1%} confidence), true {class_names[true]} "
          f"-> {verdict}")

# Show all three images with predicted labels, confidence, and true labels.
fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
for i, ax in enumerate(axes):
    pred, true = y_pred[i].item(), y_new[i].item()
    ax.imshow(X_new[i].cpu().squeeze(), cmap="binary")
    ax.set_title(f"Pred: {class_names[pred]} ({y_proba[i, pred]:.0%})\n"
                 f"True: {class_names[true]}",
                 color="green" if pred == true else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("inference.png", dpi=150)
print("Saved inference.png")
plt.show()

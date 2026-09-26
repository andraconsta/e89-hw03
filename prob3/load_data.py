"""Step 1: load FashionMNIST, split it into train/valid/test, build DataLoaders.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Follows "Building an Image Classifier with PyTorch" in Geron, Hands-On
Machine Learning, chapter 10 (10_neural_nets_with_pytorch.ipynb).
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# Preprocessing pipeline, as in Geron:
# ToImage() turns each PIL image into an image tensor of shape [1, 28, 28]
# (channels, height, width) with uint8 pixels in 0-255.
# ToDtype(float32, scale=True) casts to float32 and rescales pixels to [0, 1].
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Load FashionMNIST (downloaded into ./datasets on the first run).
# The official training set has 60,000 images, the test set 10,000.
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

# Split the training set into 55,000 train and 5,000 validation images.
# Seeding first makes random_split produce the same split on every run.
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

# Wrap each dataset in a DataLoader serving mini-batches of 32 images.
# Geron re-seeds here so the training loader's shuffle order is reproducible.
# Only the training data is shuffled (a new order each epoch).
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    # Dataset sizes: expected 55,000 / 5,000 / 10,000.
    print(f"Train size: {len(train_data)}")
    print(f"Valid size: {len(valid_data)}")
    print(f"Test size:  {len(test_data)}")

    # One sample is an (image, target) tuple. Grayscale, so shape [1, 28, 28].
    X_sample, y_sample = train_data[0]
    print(f"Sample shape: {X_sample.shape}, dtype: {X_sample.dtype}")
    print(f"Sample class: {y_sample} ({train_and_valid_data.classes[y_sample]})")

    # One batch: images [32, 1, 28, 28] float32, labels [32] int64.
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch images: {X_batch.shape}, dtype: {X_batch.dtype}")
    print(f"Batch labels: {y_batch.shape}, dtype: {y_batch.dtype}")

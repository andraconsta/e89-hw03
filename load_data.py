"""Load FashionMNIST, split it into train/valid/test, and build DataLoaders.

Follows "Building an Image Classifier with PyTorch" > "Using TorchVision to
Load the Dataset" in Geron, Hands-On Machine Learning, chapter 10.
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# Step 1: define the preprocessing pipeline.
# ToImage() turns each PIL image into a tensor-backed image of shape [1, 28, 28]
# (channels, rows, columns) with uint8 pixel values 0-255.
# ToDtype(float32, scale=True) converts to float32 and rescales pixels to [0, 1].
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Step 2: load FashionMNIST (downloads into ./datasets on the first run).
# The official training set has 60,000 images and the test set has 10,000.
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

# Step 3: carve a validation set out of the training set.
# Seeding first makes random_split pick the same 55,000/5,000 split every run.
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

# Step 4: wrap each dataset in a DataLoader that serves mini-batches of 32.
# Seeding again fixes the shuffle order of the training loader, so training
# runs are reproducible. Only the training set is shuffled (a new order each
# epoch); validation and test order doesn't matter.
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    # Step 5: sanity-check the sizes of each split.
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"Test:  {len(test_data)}")

    # Step 6: inspect one sample. Each entry is an (image, target) tuple;
    # grayscale images have a single channel, so the shape is [1, 28, 28].
    X_sample, y_sample = train_data[0]
    print(f"Sample image: {X_sample.shape}, dtype {X_sample.dtype}")
    print(f"Sample class: {train_and_valid_data.classes[y_sample]}")

    # Step 7: pull one batch to confirm the loader's output shapes:
    # images [32, 1, 28, 28] float32 and labels [32] (class ids 0-9).
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch images: {X_batch.shape}, dtype {X_batch.dtype}")
    print(f"Batch labels: {y_batch.shape}")

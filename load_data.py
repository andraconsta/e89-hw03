"""Load FashionMNIST, split it into train/valid/test, and build DataLoaders."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

BATCH_SIZE = 32        # number of images per mini-batch
DATA_DIR = "datasets"  # where torchvision downloads and caches the dataset

# Step 1: define the preprocessing pipeline.
# ToImage() turns each PIL image into a tensor-backed image of shape [1, 28, 28]
# (channels, height, width) with uint8 pixel values 0-255.
# ToDtype(float32, scale=True) converts to float32 and rescales pixels to [0, 1].
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Step 2: load FashionMNIST (downloads on first run).
# The official training set has 60,000 images and the test set has 10,000.
train_full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

# Step 3: carve a validation set out of the training set.
# Setting the seed first makes random_split pick the same 55,000/5,000 split
# every time the script runs, so results are reproducible.
torch.manual_seed(42)
train_set, valid_set = random_split(train_full, [55_000, 5_000])

# Step 4: wrap each dataset in a DataLoader that serves mini-batches.
# The training loader shuffles every epoch so the model sees images in a new
# order each pass; validation and test order doesn't matter, so no shuffling.
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    # Step 5: sanity-check the sizes of each split.
    print(f"Train: {len(train_set)}")
    print(f"Valid: {len(valid_set)}")
    print(f"Test:  {len(test_set)}")

    # Step 6: pull one batch to confirm shapes: images should be
    # [32, 1, 28, 28] float32 and labels [32] (class ids 0-9).
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch images: {X_batch.shape}, dtype {X_batch.dtype}")
    print(f"Batch labels: {y_batch.shape}")

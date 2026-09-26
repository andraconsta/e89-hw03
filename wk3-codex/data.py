"""Load FashionMNIST and create reproducible data splits and mini-batches."""
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# Convert each grayscale PIL image to a float tensor scaled to [0, 1].
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Load the 60,000-image training set and the separate 10,000-image test set.
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

# Split training data into 55,000 training examples and 5,000 validation examples.
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

# Build batch-size-32 loaders and shuffle only the training examples.
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    # Print split sizes, a sample, and a batch to verify the input pipeline.
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"Test:  {len(test_data)}")
    X_sample, y_sample = train_data[0]
    print(f"Sample image: {X_sample.shape}, dtype {X_sample.dtype}")
    print(f"Sample class: {train_and_valid_data.classes[y_sample]}")
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch images: {X_batch.shape}, dtype {X_batch.dtype}")
    print(f"Batch labels: {y_batch.shape}")

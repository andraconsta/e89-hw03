import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

BATCH_SIZE = 32
DATA_DIR = "datasets"

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

torch.manual_seed(42)
train_set, valid_set = random_split(train_full, [55_000, 5_000])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    print(f"Train: {len(train_set)}")
    print(f"Valid: {len(valid_set)}")
    print(f"Test:  {len(test_set)}")

    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch images: {X_batch.shape}, dtype {X_batch.dtype}")
    print(f"Batch labels: {y_batch.shape}")

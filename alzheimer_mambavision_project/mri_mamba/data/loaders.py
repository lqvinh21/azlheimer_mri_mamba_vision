import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from ..config import CLASS_NAMES
from .transforms import make_eval_transform, make_train_transform


def create_loaders(split_root, img_size, batch_size, num_workers):
    train_ds = datasets.ImageFolder(
        split_root / "train",
        transform=make_train_transform(img_size),
    )
    val_ds = datasets.ImageFolder(
        split_root / "val",
        transform=make_eval_transform(img_size),
    )
    test_ds = datasets.ImageFolder(
        split_root / "test",
        transform=make_eval_transform(img_size),
    )
    if train_ds.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {train_ds.classes}")

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, train_ds

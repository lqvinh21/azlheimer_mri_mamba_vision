from pathlib import Path

import torch


def save_best_checkpoint(model, path: Path, class_names, class_to_idx, img_size, val_macro_f1):
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "best_val_macro_f1": val_macro_f1,
    }, path)


def load_checkpoint(model, path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

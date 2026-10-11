import time

import torch

from .checkpoint import save_best_checkpoint
from .losses import make_weighted_cross_entropy
from .train_epoch import run_epoch


def fit_model(model, train_loader, val_loader, train_dataset, device, output_dir, args):
    criterion = make_weighted_cross_entropy(train_dataset, len(train_dataset.classes), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_path = output_dir / "Alzheimer_MambaVision_Tiny_best.pth"
    best_f1 = -1.0
    bad_epochs = 0
    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        "train_macro_f1": [], "val_macro_f1": [],
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True, epoch, args.epochs
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, False, epoch, args.epochs
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"train_macro_f1={train_f1:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f} | time={(time.time() - epoch_start) / 60:.1f} min"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            save_best_checkpoint(
                model, best_path, train_dataset.classes, train_dataset.class_to_idx,
                args.img_size, best_f1,
            )
            print(f"Saved best checkpoint: {best_path}")
        else:
            bad_epochs += 1
            print(f"Validation did not improve: {bad_epochs}/{args.patience}")
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break
    return best_path, history

import numpy as np
import torch
from tqdm.auto import tqdm

from .metrics import classification_metrics


def run_epoch(model, loader, criterion, optimizer, scaler, device, training, epoch, total_epochs):
    model.train(training)
    losses = []
    y_true = []
    y_pred = []
    phase = "Train" if training else "Validation"
    progress = tqdm(loader, desc=f"{phase} {epoch}/{total_epochs}", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{np.mean(losses):.4f}")

    accuracy, macro_f1 = classification_metrics(y_true, y_pred)
    return float(np.mean(losses)), accuracy, macro_f1, y_true, y_pred

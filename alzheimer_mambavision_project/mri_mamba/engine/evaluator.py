gitimport matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from ..config import CLASS_NAMES
from .checkpoint import load_checkpoint
from .metrics import classification_metrics


def evaluate_test(model, test_loader, checkpoint_path, device, output_dir):
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    accuracy, macro_f1 = classification_metrics(y_true, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    save_confusion_matrix(y_true, y_pred, output_dir)
    return accuracy, macro_f1


def save_confusion_matrix(y_true, y_pred, output_dir):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Test confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.show()

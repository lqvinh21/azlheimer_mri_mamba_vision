import matplotlib.pyplot as plt


def save_training_curves(history, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.show()

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import mamba_ssm  # noqa: F401 - Mamba SSM dependency used by MambaVision.
from mri_mamba.data.discovery import resolve_data_root
from mri_mamba.data.loaders import create_loaders
from mri_mamba.data.split import create_grouped_splits
from mri_mamba.engine.evaluator import evaluate_test
from mri_mamba.engine.trainer import fit_model
from mri_mamba.models.classifier import MRI_MambaFusionClassifier
from mri_mamba.utils.plots import save_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaVision-Tiny for Alzheimer MRI classification")
    parser.add_argument("--data-root", required=True, help="Dataset folder or ZIP archive")
    parser.add_argument("--output-dir", default="./mri_mambavision_output")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    class_root = resolve_data_root(Path(args.data_root), output_dir)
    split_root = create_grouped_splits(class_root, output_dir, args.seed)
    train_loader, val_loader, test_loader, train_ds = create_loaders(
        split_root, args.img_size, args.batch_size, args.num_workers
    )
    print(f"Class mapping: {train_ds.class_to_idx}")
    print(f"Samples: train={len(train_ds)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = MRI_MambaFusionClassifier(
        num_classes=len(train_ds.classes),
        pretrained=not args.no_pretrained,
        model_path=output_dir / "mamba_vision_tiny_1k.pth.tar",
    ).to(device)
    checkpoint_path, history = fit_model(model, train_loader, val_loader, train_ds, device, output_dir, args)
    save_training_curves(history, output_dir)
    evaluate_test(model, test_loader, checkpoint_path, device, output_dir)
    print(f"All outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()

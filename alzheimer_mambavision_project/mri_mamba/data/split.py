import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from ..config import CLASS_NAMES
from .dedup import scan_images, unique_groups_without_label_conflicts


def _assign_splits(records, seed):
    labels = np.asarray([row["label"] for row in records])
    groups = np.asarray([row["group"] for row in records])
    group_counts = [len(set(groups[labels == name])) for name in CLASS_NAMES]
    n_splits = min(20, min(group_counts))
    if n_splits < 4:
        raise ValueError("Need at least 4 unique image groups in every class for grouped splitting")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(records)), labels, groups))
    train_folds = max(1, int(0.70 * n_splits))
    val_folds = max(1, int(0.15 * n_splits))
    if train_folds + val_folds >= n_splits:
        train_folds = n_splits - 2
        val_folds = 1

    split_indices = {
        "train": np.concatenate([folds[i][1] for i in range(train_folds)]),
        "val": np.concatenate([folds[i][1] for i in range(train_folds, train_folds + val_folds)]),
        "test": np.concatenate([folds[i][1] for i in range(train_folds + val_folds, n_splits)]),
    }
    split_by_index = {}
    for split_name, indices in split_indices.items():
        for index in indices:
            split_by_index[int(index)] = split_name
    return split_by_index, n_splits


def create_grouped_splits(class_root: Path, output_dir: Path, seed: int = 42) -> Path:
    """Hash-deduplicate and create approximately stratified 70/15/15 splits."""
    all_records = scan_images(class_root)
    records, conflicting_hashes = unique_groups_without_label_conflicts(all_records)
    split_by_index, n_splits = _assign_splits(records, seed)

    split_root = output_dir / "Split_Alzheimer_MRI"
    if split_root.exists():
        shutil.rmtree(split_root)
    for split_name in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            (split_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "sha256", "split"])
        writer.writeheader()
        for index, row in enumerate(tqdm(records, desc="Copying unique images")):
            split_name = split_by_index[index]
            destination = split_root / split_name / row["label"] / f"{row['group']}{row['path'].suffix.lower()}"
            shutil.copy2(row["path"], destination)
            writer.writerow({
                "path": str(row["path"]),
                "label": row["label"],
                "sha256": row["group"],
                "split": split_name,
            })

    print(f"Image files scanned: {len(all_records)}")
    print(f"Unique images retained: {len(records)}")
    print(f"Conflicting-label duplicate groups removed: {len(conflicting_hashes)}")
    print(f"Grouped folds used: {n_splits} (approximately 70/15/15)")
    print(f"Split output: {split_root}")
    return split_root

import hashlib
from pathlib import Path

from tqdm.auto import tqdm

from ..config import CLASS_NAMES, IMAGE_EXTENSIONS


def scan_images(class_root: Path):
    """Collect image paths and exact SHA-256 hashes from each class folder."""
    records = []
    for class_name in CLASS_NAMES:
        class_dir = class_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        image_paths = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in tqdm(image_paths, desc=f"Hashing {class_name}", leave=False):
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            records.append({"path": image_path, "label": class_name, "group": image_hash})
    if not records:
        raise ValueError(f"No supported images found under {class_root}")
    return records


def unique_groups_without_label_conflicts(records):
    """Keep one copy per byte-identical image; discard hashes with conflicting labels."""
    labels_by_hash = {}
    row_by_hash = {}
    for row in records:
        labels_by_hash.setdefault(row["group"], set()).add(row["label"])
        row_by_hash.setdefault(row["group"], row)

    conflicting_hashes = {
        image_hash for image_hash, labels in labels_by_hash.items() if len(labels) > 1
    }
    unique_records = [
        row for image_hash, row in row_by_hash.items()
        if image_hash not in conflicting_hashes
    ]
    if not unique_records:
        raise ValueError("No images remain after removing conflicting duplicate labels")
    return unique_records, conflicting_hashes

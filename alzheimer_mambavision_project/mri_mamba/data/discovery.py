import os
from pathlib import Path

from ..config import CLASS_NAMES


def find_class_root(root: Path) -> Path:
    """Find the directory that directly contains all four class folders."""
    if all((root / name).is_dir() for name in CLASS_NAMES):
        return root
    for current, _, _ in os.walk(root):
        current_path = Path(current)
        if all((current_path / name).is_dir() for name in CLASS_NAMES):
            return current_path
    raise FileNotFoundError(f"Could not find all four class folders under: {root}")


def resolve_data_root(data_root: Path, output_dir: Path) -> Path:
    """Accept either an extracted dataset folder or a ZIP archive."""
    data_root = data_root.expanduser().resolve()
    if data_root.is_file() and data_root.suffix.lower() == ".zip":
        from .archive import safe_extract

        extracted_root = output_dir / "raw_data"
        if not extracted_root.exists() or not any(extracted_root.iterdir()):
            print(f"Extracting dataset archive: {data_root}")
            safe_extract(data_root, extracted_root)
        return find_class_root(extracted_root)
    if data_root.is_dir():
        return find_class_root(data_root)
    raise FileNotFoundError(f"Data path does not exist: {data_root}")

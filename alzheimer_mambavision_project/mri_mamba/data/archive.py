import zipfile
from pathlib import Path


def safe_extract(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP while rejecting paths that escape the destination."""
    extract_to.mkdir(parents=True, exist_ok=True)
    root = extract_to.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_to / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(extract_to)

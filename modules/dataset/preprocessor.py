import hashlib
from pathlib import Path

import cv2
import numpy as np


def validate_dataset_integrity(data_dir: Path) -> dict:
    """
    Walks data_dir subdirectories.
    Returns {
        'families': list[str],
        'counts': dict[str, int],
        'total': int,
        'corrupt_files': list[Path],
        'duplicate_hashes': list[str],
    }
    Raises FileNotFoundError if data_dir does not exist.
    SRS ref: FE-4 of Module 3
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    families = sorted(
        [d.name for d in data_dir.iterdir() if d.is_dir()]
    )
    counts: dict[str, int] = {}
    corrupt_files: list[Path] = []
    seen_hashes: dict[str, Path] = {}
    duplicate_hashes: list[str] = []

    for family in families:
        family_dir = data_dir / family
        image_files = list(family_dir.glob("*.png")) + list(family_dir.glob("*.bmp"))
        counts[family] = len(image_files)
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                corrupt_files.append(img_path)
                continue
            file_bytes = img_path.read_bytes()
            sha = hashlib.sha256(file_bytes).hexdigest()
            if sha in seen_hashes:
                if sha not in duplicate_hashes:
                    duplicate_hashes.append(sha)
            else:
                seen_hashes[sha] = img_path

    total = sum(counts.values())
    return {
        "families": families,
        "counts": counts,
        "total": total,
        "corrupt_files": corrupt_files,
        "duplicate_hashes": duplicate_hashes,
    }


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Converts uint8 [0,255] to float32 [0.0, 1.0].
    SRS ref: FE-2 of Module 3
    """
    return img.astype(np.float32) / 255.0


def encode_labels(families: list) -> dict:
    """
    Returns {family_name: integer_class_index} sorted alphabetically.
    Deterministic — same input always produces same mapping.
    """
    sorted_families = sorted(families)
    return {name: idx for idx, name in enumerate(sorted_families)}

"""
Tests for Module 3 — Dataset Collection & Preprocessing.
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.dataset.preprocessor import (
    encode_labels,
    normalize_image,
    validate_dataset_integrity,
)


def _create_fake_dataset(tmp_path: Path, families: list, samples_per_family: int = 5):
    """Helper: creates a fake Malimg-style dataset with tiny PNG images."""
    for family in families:
        family_dir = tmp_path / family
        family_dir.mkdir()
        for i in range(samples_per_family):
            img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            cv2.imwrite(str(family_dir / f"sample_{i:03d}.png"), img)
    return tmp_path


@pytest.fixture()
def fake_data_dir(tmp_path):
    families = ["FamilyA", "FamilyB", "FamilyC"]
    return _create_fake_dataset(tmp_path, families, samples_per_family=10)


def test_validate_integrity_detects_missing_dir():
    """validate_dataset_integrity raises FileNotFoundError for missing directory."""
    with pytest.raises(FileNotFoundError):
        validate_dataset_integrity(Path("/nonexistent/path/xyz"))


def test_validate_integrity_counts_samples(fake_data_dir):
    """validate_dataset_integrity returns correct family counts."""
    result = validate_dataset_integrity(fake_data_dir)
    assert set(result["families"]) == {"FamilyA", "FamilyB", "FamilyC"}
    assert result["total"] == 30
    for family in result["families"]:
        assert result["counts"][family] == 10


def test_split_ratios_sum_to_one():
    """Train + val + test ratios from config sum to 1.0."""
    import config
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-9


def test_stratified_split_no_class_missing_from_val(fake_data_dir):
    """All classes appear in val split after stratified split."""
    from modules.dataset.loader import MalimgDataset
    val_dataset = MalimgDataset(fake_data_dir, img_size=32, split="val")
    val_class_counts = val_dataset.class_counts
    # All families should have at least 1 sample in val (10 samples per family, 15% val)
    for family, count in val_class_counts.items():
        assert count >= 1, f"Family {family} missing from val split"


def test_getitem_returns_correct_tensor_shape(fake_data_dir):
    """MalimgDataset.__getitem__ returns tensor of shape (1, img_size, img_size)."""
    from modules.dataset.loader import MalimgDataset
    dataset = MalimgDataset(fake_data_dir, img_size=32, split="train")
    tensor, label = dataset[0]
    assert tensor.shape == (1, 32, 32)
    assert isinstance(label, int)


def test_label_encoding_deterministic():
    """encode_labels always produces same mapping for same input regardless of input order."""
    families = ["Zbot", "Allaple.A", "Rbot", "Yuner.A"]
    mapping1 = encode_labels(families)
    import random
    shuffled = families[:]
    random.shuffle(shuffled)
    mapping2 = encode_labels(shuffled)
    assert mapping1 == mapping2


def test_normalize_image_range():
    """normalize_image converts uint8 array to float32 in [0, 1]."""
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = normalize_image(img)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

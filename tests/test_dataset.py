"""
test_dataset.py — matches actual MalTwin signatures.

Key facts from source:
- encode_labels(list[str]) takes a list of family name strings, not a Path
- get_dataloaders() returns (train_dl, val_dl, test_dl, class_names) — 4 values
- MalimgDataset(data_dir, split) — data_dir must be a Path
- ClassAwareOversampler(dataset, strategy) takes a MalimgDataset, not a label list
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


FAMILIES = ["Adialer.C", "Agent.FYI", "Allaple.A", "Skintrim.N", "Yuner.A"]
SAMPLES_PER_FAMILY = 10


@pytest.fixture(scope="module")
def synthetic_malimg(tmp_path_factory):
    """Build a minimal synthetic Malimg folder structure."""
    root = tmp_path_factory.mktemp("malimg")
    for family in FAMILIES:
        fdir = root / family
        fdir.mkdir()
        for i in range(SAMPLES_PER_FAMILY):
            img = Image.fromarray(
                np.random.randint(0, 256, (128, 128), dtype=np.uint8), mode="L"
            )
            img.save(fdir / f"sample_{i:04d}.png")
    return root  # Path


@pytest.fixture(scope="module")
def loader_mod():
    from modules.dataset import loader, preprocessor
    return loader, preprocessor


# ===========================================================================
# encode_labels — takes list[str], returns dict[str, int]
# ===========================================================================

class TestLabelEncoding:

    def test_encode_returns_dict(self, loader_mod):
        _, preprocessor = loader_mod
        mapping = preprocessor.encode_labels(FAMILIES)
        assert isinstance(mapping, dict)

    def test_encode_correct_class_count(self, loader_mod):
        _, preprocessor = loader_mod
        mapping = preprocessor.encode_labels(FAMILIES)
        assert len(mapping) == len(FAMILIES)

    def test_encode_indices_are_contiguous(self, loader_mod):
        _, preprocessor = loader_mod
        mapping = preprocessor.encode_labels(FAMILIES)
        indices = sorted(mapping.values())
        assert indices == list(range(len(FAMILIES)))

    def test_encode_keys_match_families(self, loader_mod):
        _, preprocessor = loader_mod
        mapping = preprocessor.encode_labels(FAMILIES)
        assert set(mapping.keys()) == set(FAMILIES)


# ===========================================================================
# MalimgDataset — direct instantiation
# ===========================================================================

class TestMalimgDataset:

    def test_train_split_instantiates(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        assert len(ds) > 0

    def test_val_split_instantiates(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="val")
        assert len(ds) >= 0

    def test_test_split_instantiates(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="test")
        assert len(ds) >= 0

    def test_splits_sum_to_total(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        train_ds = loader.MalimgDataset(synthetic_malimg, split="train")
        val_ds   = loader.MalimgDataset(synthetic_malimg, split="val")
        test_ds  = loader.MalimgDataset(synthetic_malimg, split="test")
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(FAMILIES) * SAMPLES_PER_FAMILY

    def test_train_larger_than_val_and_test(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        train_ds = loader.MalimgDataset(synthetic_malimg, split="train")
        val_ds   = loader.MalimgDataset(synthetic_malimg, split="val")
        test_ds  = loader.MalimgDataset(synthetic_malimg, split="test")
        assert len(train_ds) > len(val_ds)
        assert len(train_ds) > len(test_ds)

    def test_getitem_returns_tensor_and_label(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(label, int)

    def test_item_tensor_shape(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        tensor, _ = ds[0]
        assert tensor.shape == (1, 128, 128)

    def test_item_tensor_dtype(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32

    def test_label_in_valid_range(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        for _, label in ds:
            assert 0 <= label < len(FAMILIES)

    def test_get_labels_returns_list(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        labels = ds.get_labels()
        assert isinstance(labels, list)
        assert len(labels) == len(ds)

    def test_missing_dir_raises(self, loader_mod, tmp_path):
        loader, _ = loader_mod
        with pytest.raises(FileNotFoundError):
            loader.MalimgDataset(tmp_path / "no_such_dir", split="train")

    def test_invalid_split_raises(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        with pytest.raises(ValueError):
            loader.MalimgDataset(synthetic_malimg, split="banana")

    def test_class_names_sorted(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        ds = loader.MalimgDataset(synthetic_malimg, split="train")
        assert ds.class_names == sorted(ds.class_names)


# ===========================================================================
# get_dataloaders — returns (train_dl, val_dl, test_dl, class_names)
# ===========================================================================

class TestGetDataloaders:

    def test_returns_four_values(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        result = loader.get_dataloaders(
            data_dir=synthetic_malimg, batch_size=4, num_workers=0
        )
        assert len(result) == 4, \
            f"Expected 4 return values (train, val, test, class_names), got {len(result)}"

    def test_class_names_is_list_of_strings(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        _, _, _, class_names = loader.get_dataloaders(
            data_dir=synthetic_malimg, batch_size=4, num_workers=0
        )
        assert isinstance(class_names, list)
        assert all(isinstance(n, str) for n in class_names)

    def test_class_names_count(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        _, _, _, class_names = loader.get_dataloaders(
            data_dir=synthetic_malimg, batch_size=4, num_workers=0
        )
        assert len(class_names) == len(FAMILIES)

    def test_train_batch_shape(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        train_dl, _, _, _ = loader.get_dataloaders(
            data_dir=synthetic_malimg, batch_size=4, num_workers=0
        )
        images, labels = next(iter(train_dl))
        assert images.ndim == 4
        assert images.shape[1] == 1
        assert images.dtype == torch.float32

    def test_label_values_in_range(self, loader_mod, synthetic_malimg):
        loader, _ = loader_mod
        train_dl, _, _, class_names = loader.get_dataloaders(
            data_dir=synthetic_malimg, batch_size=4, num_workers=0
        )
        images, labels = next(iter(train_dl))
        assert labels.min() >= 0
        assert labels.max() < len(class_names)


# ===========================================================================
# Integration tests (require real Malimg at data/malimg/)
# ===========================================================================

@pytest.mark.integration
class TestMalimgIntegration:
    MALIMG_PATH = Path("data/malimg")

    def test_malimg_has_25_families(self, loader_mod):
        from modules.dataset.preprocessor import encode_labels
        families = [d.name for d in self.MALIMG_PATH.iterdir() if d.is_dir()]
        mapping = encode_labels(families)
        assert len(mapping) == 25

    def test_full_dataloader_one_batch(self, loader_mod):
        loader, _ = loader_mod
        train_dl, _, _, class_names = loader.get_dataloaders(
            data_dir=self.MALIMG_PATH, batch_size=32, num_workers=0
        )
        images, labels = next(iter(train_dl))
        assert images.shape[1] == 1
        assert len(class_names) == 25
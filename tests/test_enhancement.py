"""
Tests for Module 4 — Data Enhancement & Balancing.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.enhancement.augmentor import GaussianNoise, get_train_transforms, get_val_transforms


def _create_fake_dataset(tmp_path: Path, families: list, samples_per_family: int = 10):
    for family in families:
        family_dir = tmp_path / family
        family_dir.mkdir()
        for i in range(samples_per_family):
            img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            cv2.imwrite(str(family_dir / f"sample_{i:03d}.png"), img)
    return tmp_path


@pytest.fixture()
def fake_data_dir(tmp_path):
    return _create_fake_dataset(tmp_path, ["FamilyA", "FamilyB", "FamilyC"])


def test_gaussian_noise_clamps_output_to_01():
    """GaussianNoise output is clamped to [0.0, 1.0]."""
    noise_transform = GaussianNoise()
    # Tensor at boundary values
    tensor = torch.ones(1, 32, 32) * 0.95
    out = noise_transform(tensor)
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0

    tensor_low = torch.ones(1, 32, 32) * 0.05
    out_low = noise_transform(tensor_low)
    assert float(out_low.min()) >= 0.0
    assert float(out_low.max()) <= 1.0


def test_augmented_tensor_same_shape_as_input():
    """Training transforms preserve tensor shape."""
    transforms = get_train_transforms(32)
    tensor = torch.rand(1, 32, 32)
    out = transforms(tensor)
    assert out.shape == tensor.shape


def test_val_transforms_preserve_shape():
    """Val transforms preserve tensor shape."""
    transforms = get_val_transforms(32)
    tensor = torch.rand(1, 32, 32)
    out = transforms(tensor)
    assert out.shape == tensor.shape


def test_oversampler_returns_weighted_random_sampler(fake_data_dir):
    """ClassAwareOversampler.get_sampler() returns a WeightedRandomSampler."""
    from torch.utils.data import WeightedRandomSampler
    from modules.dataset.loader import MalimgDataset
    from modules.enhancement.balancer import ClassAwareOversampler

    dataset = MalimgDataset(fake_data_dir, img_size=32, split="train")
    oversampler = ClassAwareOversampler(dataset, strategy="oversample_minority")
    sampler = oversampler.get_sampler()
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == len(dataset)

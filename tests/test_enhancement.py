"""
test_enhancement.py — matches actual MalTwin signatures.

Key facts from source:
- ClassAwareOversampler(dataset, strategy) takes a MalimgDataset (not a list)
  dataset must expose get_labels() -> list[int]
- ClassAwareOversampler.get_sampler() returns WeightedRandomSampler
- GaussianNoise — inspect signature dynamically for parameter name
"""

import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler


@pytest.fixture(scope="module")
def enhancement_mod():
    from modules.enhancement import augmentor, balancer
    return augmentor, balancer


def _make_pil_image(h: int = 128, w: int = 128):
    from PIL import Image
    arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_tensor(h: int = 128, w: int = 128) -> torch.Tensor:
    return torch.rand(1, h, w)


def _noise_kwargs(augmentor_mod, value: float) -> dict:
    """Find the first non-self parameter of GaussianNoise.__init__."""
    sig = inspect.signature(augmentor_mod.GaussianNoise.__init__)
    params = [p for p in sig.parameters if p != "self"]
    return {params[0]: value} if params else {}


class _FakeDataset:
    """Minimal object that mimics MalimgDataset for oversampler tests."""
    def __init__(self, labels: list):
        self._labels = labels

    def get_labels(self) -> list:
        return self._labels


# ===========================================================================
# Training transforms
# ===========================================================================

class TestTrainTransforms:

    def test_returns_tensor(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        t = augmentor.get_train_transforms()(_make_pil_image())
        assert isinstance(t, torch.Tensor)

    def test_output_shape(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        t = augmentor.get_train_transforms()(_make_pil_image())
        assert t.shape == (1, 128, 128)

    def test_output_dtype_float32(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        t = augmentor.get_train_transforms()(_make_pil_image())
        assert t.dtype == torch.float32

    def test_augmentation_produces_variation(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        transform = augmentor.get_train_transforms()
        img = _make_pil_image()
        results = [transform(img) for _ in range(10)]
        unique = {r.numpy().tobytes() for r in results}
        assert len(unique) > 1, "Train transforms appear deterministic"


# ===========================================================================
# Validation transforms
# ===========================================================================

class TestValTransforms:

    def test_returns_tensor(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        t = augmentor.get_val_transforms()(_make_pil_image())
        assert isinstance(t, torch.Tensor)

    def test_output_shape(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        t = augmentor.get_val_transforms()(_make_pil_image())
        assert t.shape == (1, 128, 128)

    def test_is_deterministic(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        transform = augmentor.get_val_transforms()
        img = _make_pil_image()
        torch.testing.assert_close(transform(img), transform(img))


# ===========================================================================
# GaussianNoise
# ===========================================================================

class TestGaussianNoise:

    def test_does_not_change_shape(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        if not hasattr(augmentor, "GaussianNoise"):
            pytest.skip("GaussianNoise not in augmentor")
        noise = augmentor.GaussianNoise(**_noise_kwargs(augmentor, 0.05))
        t = _make_tensor()
        assert noise(t).shape == t.shape

    def test_does_not_change_dtype(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        if not hasattr(augmentor, "GaussianNoise"):
            pytest.skip("GaussianNoise not in augmentor")
        noise = augmentor.GaussianNoise(**_noise_kwargs(augmentor, 0.05))
        t = _make_tensor()
        assert noise(t).dtype == t.dtype

    def test_nonzero_noise_modifies_tensor(self, enhancement_mod):
        augmentor, _ = enhancement_mod
        if not hasattr(augmentor, "GaussianNoise"):
            pytest.skip("GaussianNoise not in augmentor")
        noise = augmentor.GaussianNoise(**_noise_kwargs(augmentor, 0.5))
        t = _make_tensor()
        assert not torch.equal(noise(t), t)


# ===========================================================================
# ClassAwareOversampler
# ===========================================================================

class TestClassAwareOversampler:

    def _make_dataset(self, labels):
        return _FakeDataset(labels)

    def test_get_sampler_returns_weighted_random_sampler(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10 + [2] * 30
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds).get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_num_samples_equals_dataset(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10 + [2] * 30
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds).get_sampler()
        assert sampler.num_samples == len(labels)

    def test_sampler_weights_length_matches_dataset(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10 + [2] * 30
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds).get_sampler()
        assert len(sampler.weights) == len(labels)

    def test_minority_weight_greater_than_majority(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 1000 + [1] * 50
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds).get_sampler()
        weights = sampler.weights.tolist()
        majority_w = weights[0]      # class 0
        minority_w = weights[1000]   # class 1
        assert minority_w > majority_w

    def test_all_weights_positive(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10 + [2] * 30
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds).get_sampler()
        assert sampler.weights.min().item() > 0

    def test_sqrt_inverse_strategy(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds, strategy="sqrt_inverse").get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_uniform_strategy(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 200 + [1] * 10
        ds = self._make_dataset(labels)
        sampler = balancer.ClassAwareOversampler(ds, strategy="uniform").get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_invalid_strategy_raises(self, enhancement_mod):
        _, balancer = enhancement_mod
        labels = [0] * 10
        ds = self._make_dataset(labels)
        with pytest.raises(ValueError):
            balancer.ClassAwareOversampler(ds, strategy="bad_strategy")
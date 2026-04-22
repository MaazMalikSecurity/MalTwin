# MalTwin — Phase 2: Dataset Module
### Agent Instruction Document | `modules/dataset/` + `tests/test_dataset.py`

> **Read this entire document before writing a single line of code.**
> Every class, method, signature, and behavioral rule is specified completely.
> Do not infer, guess, or deviate from what is written here.

---

## Mandatory Rules (from PRD Section 16)

These are the most commonly hallucinated bugs. Violating any of them causes test failures.

- **Read `MALTWIN_PRD_COMPLETE.md`** before writing any code.
- All CNN tensors are **single-channel** `(batch, 1, H, W)` — NEVER RGB `(batch, 3, H, W)`.
- `cv2.imread()` is **always** called with `cv2.IMREAD_GRAYSCALE` flag. Never load as BGR.
- `cv2.resize()` target is `(width, height)` — i.e. `(IMG_SIZE, IMG_SIZE)`. OpenCV uses `(width, height)` convention. **This is the most common bug.**
- `transforms.Normalize(mean=[0.5], std=[0.5])` uses **single-element lists** (not scalars, not 3-element lists).
- `encode_labels()` **always sorts alphabetically** — same input always produces the same mapping.
- All `train_test_split()` calls use `random_state=config.RANDOM_SEED` (42).
- `get_val_transforms` is used for **val, test, and inference** — never `get_train_transforms` for inference.
- `drop_last=True` on train DataLoader to prevent single-sample batches breaking BatchNorm.
- All paths use `pathlib.Path`, never string concatenation.
- Tests requiring Malimg dataset are marked `@pytest.mark.integration`.
- `pytest tests/test_dataset.py -v -m "not integration"` must pass with **zero failures** without any dataset present.

---

## Phase 2 Scope

Phase 2 implements the dataset loading, preprocessing, and (partially) the enhancement module that the loader depends on. It does **not** implement the full enhancement module (that is Phase 3) — but the loader imports `get_train_transforms` and `get_val_transforms` from `modules/enhancement/augmentor.py`, so a minimal version of those two functions must exist.

### Files to create

| File | Description |
|------|-------------|
| `modules/dataset/__init__.py` | Package exports |
| `modules/dataset/preprocessor.py` | `validate_dataset_integrity`, `normalize_image`, `encode_labels`, `save_class_names`, `load_class_names` |
| `modules/dataset/loader.py` | `MalimgDataset`, `get_dataloaders` |
| `modules/enhancement/__init__.py` | Package exports (minimal) |
| `modules/enhancement/augmentor.py` | `GaussianNoise`, `get_train_transforms`, `get_val_transforms` |
| `modules/enhancement/balancer.py` | `ClassAwareOversampler` |
| `tests/test_dataset.py` | Full test suite (exactly as specified below) |

> **Note:** Phase 2 produces the full `modules/enhancement/` module because `loader.py` imports from it. The enhancement tests (`tests/test_enhancement.py`) are left for Phase 3 — but the implementation must be complete and correct here.

---

## File 1: `modules/dataset/__init__.py`

```python
# modules/dataset/__init__.py
from .loader import MalimgDataset, get_dataloaders
from .preprocessor import validate_dataset_integrity
```

---

## File 2: `modules/dataset/preprocessor.py`

```python
# modules/dataset/preprocessor.py
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional


def validate_dataset_integrity(data_dir: Path) -> dict:
    """
    Scans the Malimg dataset directory and produces an integrity report.

    Args:
        data_dir: Path to the Malimg root directory (config.DATA_DIR).

    Returns:
        {
            'valid':            bool,         # True if no corrupt files found
            'families':         list[str],    # sorted list of family folder names
            'counts':           dict[str,int],# {family: sample_count}
            'total':            int,          # sum of all counts
            'min_class':        str,          # family with fewest samples
            'max_class':        str,          # family with most samples
            'imbalance_ratio':  float,        # max_count / min_count
            'corrupt_files':    list[str],    # str(path) of unreadable files
            'missing_dirs':     list[str],    # always [] — see notes
        }

    Raises:
        FileNotFoundError: if data_dir does not exist
        FileNotFoundError: if data_dir has no subdirectories

    Implementation notes:
        - Iterate over data_dir.iterdir(), keeping only directories.
        - For each family dir, iterate over *.png files (case-insensitive via glob('*.png') + glob('*.PNG')).
        - For each PNG, attempt cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).
          If result is None, add str(path) to corrupt_files list.
        - Sort families alphabetically.
        - corrupt_files contains str representations of Path objects.
        - missing_dirs = [] (we cannot know expected names without hardcoding).
        - imbalance_ratio = max_count / min_count. Handle divide-by-zero if min_count == 0.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"Dataset directory is empty: {data_dir}")

    families = sorted([d.name for d in subdirs])
    counts = {}
    corrupt_files = []

    for family_dir in sorted(subdirs, key=lambda d: d.name):
        family = family_dir.name
        png_files = list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG'))
        # Deduplicate (glob may overlap on case-insensitive filesystems)
        png_files = list({str(p): p for p in png_files}.values())
        count = 0
        for path in png_files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                corrupt_files.append(str(path))
            else:
                count += 1
        counts[family] = count

    total = sum(counts.values())
    max_class = max(counts, key=lambda k: counts[k]) if counts else ''
    min_class = min(counts, key=lambda k: counts[k]) if counts else ''
    max_count = counts.get(max_class, 0)
    min_count = counts.get(min_class, 1)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    return {
        'valid':           len(corrupt_files) == 0,
        'families':        families,
        'counts':          counts,
        'total':           total,
        'min_class':       min_class,
        'max_class':       max_class,
        'imbalance_ratio': imbalance_ratio,
        'corrupt_files':   corrupt_files,
        'missing_dirs':    [],
    }


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image [0, 255] to float32 [0.0, 1.0].

    Args:
        img: numpy array, dtype uint8.

    Returns:
        numpy array, same shape, dtype float32, values in [0.0, 1.0].

    Implementation:
        return img.astype(np.float32) / 255.0

    Notes:
        - Do NOT use cv2.normalize here. Simple division is exact and fast.
        - The output of this function feeds directly into PyTorch tensors.
    """
    return img.astype(np.float32) / 255.0


def encode_labels(families: list[str]) -> dict[str, int]:
    """
    Create a deterministic string→integer label mapping.

    Args:
        families: list of family names.

    Returns:
        Dict mapping each family name to a unique integer [0, len(families)-1].
        Sorted alphabetically so the mapping is always the same for the same input.

    Implementation:
        return {name: idx for idx, name in enumerate(sorted(families))}

    Example:
        encode_labels(['Yuner.A', 'Allaple.A', 'VB.AT'])
        → {'Allaple.A': 0, 'VB.AT': 1, 'Yuner.A': 2}
    """
    return {name: idx for idx, name in enumerate(sorted(families))}


def save_class_names(class_names: list[str], output_path: Path) -> None:
    """
    Persist the ordered class name list to JSON for dashboard use.

    Args:
        class_names: sorted list of family names (index = label integer).
        output_path: destination JSON path (config.CLASS_NAMES_PATH).

    File format:
        {"class_names": ["Adialer.C", "Agent.FYI", ...]}

    Notes:
        - Creates parent directory if it does not exist.
        - Overwrites if file already exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)


def load_class_names(input_path: Path) -> list[str]:
    """
    Load class names from JSON file written by save_class_names.

    Args:
        input_path: path to class_names.json (config.CLASS_NAMES_PATH).

    Returns:
        list[str] of family names in index order.

    Raises:
        FileNotFoundError: if file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"class_names.json not found at {input_path}. "
            "Run scripts/train.py first."
        )
    with open(input_path) as f:
        return json.load(f)['class_names']
```

---

## File 3: `modules/dataset/loader.py`

```python
# modules/dataset/loader.py
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from typing import Optional, Callable
from PIL import Image

import config
from .preprocessor import encode_labels, save_class_names


class MalimgDataset(Dataset):
    """
    PyTorch Dataset for the Malimg malware image dataset.

    Loads grayscale PNG images from directory structure:
        data_dir/FamilyName/image.png

    Each image is:
        1. Loaded as grayscale (single channel) with cv2.IMREAD_GRAYSCALE
        2. Resized to (img_size, img_size) using cv2.resize(img, (img_size, img_size))
           NOTE: cv2.resize takes (width, height), so (img_size, img_size) is correct for square images.
        3. Converted to PIL Image mode 'L'
        4. Transform applied (returns float32 tensor shape (1, H, W))

    Internal data structure:
        self.samples: list[tuple[Path, int]]
        self.class_names: list[str]  — sorted alphabetically, index = label integer
        self.label_map: dict[str, int]
        self.class_counts: dict[str, int]  — counts for THIS split only

    Split algorithm (must be implemented exactly as specified):
        Step 1: Gather all (path, label) pairs for entire dataset across all families.
        Step 2: Extract label list for stratification.
        Step 3: train_test_split(all_samples, test_size=(val_ratio + test_ratio),
                                 stratify=labels, random_state=random_seed)
                → produces train_samples, temp_samples
        Step 4: relative_val = val_ratio / (val_ratio + test_ratio)
                train_test_split(temp_samples, test_size=(1 - relative_val),
                                 stratify=temp_labels, random_state=random_seed)
                → produces val_samples, test_samples
        Step 5: self.samples = train_samples / val_samples / test_samples per requested split.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        img_size: int = config.IMG_SIZE,
        transform: Optional[Callable] = None,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio: float = config.VAL_RATIO,
        test_ratio: float = config.TEST_RATIO,
        random_seed: int = config.RANDOM_SEED,
    ):
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.random_seed = random_seed

        # Build label map from all family subdirectories
        family_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        all_families = [d.name for d in family_dirs]
        self.label_map = encode_labels(all_families)
        self.class_names = sorted(all_families)  # sorted alphabetically

        # Gather all (path, label) pairs
        all_samples: list[tuple[Path, int]] = []
        for family_dir in family_dirs:
            label = self.label_map[family_dir.name]
            png_files = sorted(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))
            # Deduplicate on case-insensitive filesystems
            seen = set()
            deduped = []
            for p in png_files:
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(p)
            for path in deduped:
                all_samples.append((path, label))

        # Stratified split
        labels = [s[1] for s in all_samples]
        train_samples, temp_samples = train_test_split(
            all_samples,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_seed,
        )
        temp_labels = [s[1] for s in temp_samples]
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1.0 - relative_val),
            stratify=temp_labels,
            random_state=random_seed,
        )

        split_map = {'train': train_samples, 'val': val_samples, 'test': test_samples}
        self.samples = split_map[split]

        # Compute class counts for this split
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.samples)
        self.class_counts = {self.class_names[lbl]: cnt.get(lbl, 0) for lbl in range(len(self.class_names))}

        # Default transform: val transforms (no augmentation)
        if transform is None:
            from modules.enhancement.augmentor import get_val_transforms
            self.transform = get_val_transforms(img_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        # cv2.resize takes (width, height) — for square images this is (img_size, img_size)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(img, mode='L')
        tensor = self.transform(pil_img)   # shape: (1, img_size, img_size), float32
        return tensor, label

    def get_labels(self) -> list[int]:
        """Returns list of integer labels for all samples in this split."""
        return [label for _, label in self.samples]


def get_dataloaders(
    data_dir: Path = config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    oversample_strategy: str = config.OVERSAMPLE_STRATEGY,
    augment_train: bool = True,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build all three DataLoaders and return (train_loader, val_loader, test_loader, class_names).

    - Train loader uses oversampling sampler + optional augmentation.
    - Val and test loaders use val transforms, shuffle=False, no sampler.
    - Persists class_names to config.CLASS_NAMES_PATH for dashboard use.
    - drop_last=True on train loader prevents incomplete final batches.
    - pin_memory=True when CUDA is available.
    """
    from modules.enhancement.augmentor import get_train_transforms, get_val_transforms
    from modules.enhancement.balancer import ClassAwareOversampler

    val_transform = get_val_transforms(img_size)
    train_transform = get_train_transforms(img_size) if augment_train else val_transform

    train_ds = MalimgDataset(data_dir, 'train', img_size, train_transform,
                              random_seed=random_seed)
    val_ds   = MalimgDataset(data_dir, 'val',   img_size, val_transform,
                              random_seed=random_seed)
    test_ds  = MalimgDataset(data_dir, 'test',  img_size, val_transform,
                              random_seed=random_seed)

    sampler = ClassAwareOversampler(train_ds, strategy=oversample_strategy).get_sampler()
    use_pin = (config.DEVICE.type == 'cuda')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,         # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,          # avoid incomplete final batch during training
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    # Persist class names for dashboard
    save_class_names(train_ds.class_names, config.CLASS_NAMES_PATH)

    return train_loader, val_loader, test_loader, train_ds.class_names
```

---

## File 4: `modules/enhancement/__init__.py`

```python
# modules/enhancement/__init__.py
from .augmentor import get_train_transforms, get_val_transforms, GaussianNoise
from .balancer import ClassAwareOversampler
```

---

## File 5: `modules/enhancement/augmentor.py`

```python
# modules/enhancement/augmentor.py
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class GaussianNoise:
    """
    Custom torchvision-compatible transform that adds Gaussian noise to a tensor.

    MUST be placed AFTER transforms.ToTensor() in the pipeline.
    Operates on torch.Tensor, not PIL.Image.

    Constructor args:
        mean (float):      noise mean, default 0.0
        std_range (tuple): (min_std, max_std), std sampled uniformly each call.
                           Default (0.01, 0.05).

    __call__:
        1. Sample std = random.uniform(std_range[0], std_range[1])
        2. Generate noise = torch.randn_like(tensor) * std + mean
        3. result = tensor + noise
        4. Clamp result to [0.0, 1.0]
        5. Return clamped tensor (same shape and dtype as input)
    """

    def __init__(self, mean: float = 0.0, std_range: tuple = (0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussianNoise(mean={self.mean}, std_range={self.std_range})"


def get_train_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the augmentation pipeline for training data.

    Transform order (CRITICAL — do not reorder):
        1. RandomRotation(degrees=15, fill=0)      ← PIL stage
        2. RandomHorizontalFlip(p=0.5)             ← PIL stage
        3. RandomVerticalFlip(p=0.5)               ← PIL stage
        4. ColorJitter(brightness=0.2)             ← PIL stage (MUST be before ToTensor)
        5. ToTensor()                              ← converts PIL 'L' → (1, H, W) float32
        6. GaussianNoise(mean=0.0, std=(0.01,0.05))← Tensor stage (MUST be after ToTensor)
        7. Normalize(mean=[0.5], std=[0.5])        ← Tensor stage (single-element lists!)

    Args:
        img_size: not used directly (resizing done in Dataset.__getitem__).

    Returns:
        transforms.Compose instance
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),    # PIL stage — before ToTensor
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, std_range=(0.01, 0.05)),  # Tensor stage — after ToTensor
        transforms.Normalize(mean=[0.5], std=[0.5]),       # single-element lists
    ])


def get_val_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the inference/validation transform pipeline (NO augmentation).

    Transform order:
        1. ToTensor()                       ← PIL 'L' → (1, H, W) float32
        2. Normalize(mean=[0.5], std=[0.5]) ← maps [0,1] to [-1,1]

    Used for val, test, and inference. NEVER use get_train_transforms for inference.

    Args:
        img_size: kept for API consistency, not used here.

    Returns:
        transforms.Compose instance
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
```

---

## File 6: `modules/enhancement/balancer.py`

```python
# modules/enhancement/balancer.py
import math
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


class ClassAwareOversampler:
    """
    Produces a WeightedRandomSampler to address class imbalance in Malimg.

    Malimg is severely imbalanced (Allaple.A has ~2949 samples, Skintrim.N has ~80).
    Without balancing, the CNN learns to predict majority classes and performs
    poorly on rare families.

    Constructor args:
        dataset:  a MalimgDataset instance (train split).
                  Must expose a get_labels() method returning list[int].
        strategy: one of 'oversample_minority', 'sqrt_inverse', 'uniform'.

    Strategies:
        'oversample_minority' — weight = 1 / class_count (pure inverse frequency)
        'sqrt_inverse'        — weight = 1 / sqrt(class_count) (softer balancing)
        'uniform'             — weight = 1.0 for all samples (effectively random sampling)

    Properties set after get_sampler() call:
        self.class_weights: dict[int, float]
        self.effective_class_counts: dict[int, float]
    """

    def __init__(self, dataset, strategy: str = 'oversample_minority'):
        self.dataset = dataset
        self.strategy = strategy
        self.class_weights: dict[int, float] = {}
        self.effective_class_counts: dict[int, float] = {}

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Compute per-sample weights and return a WeightedRandomSampler.

        Implementation:
            labels = dataset.get_labels()
            class_counts = Counter(labels)

            if strategy == 'oversample_minority':
                class_weights = {c: 1.0 / count for c, count in class_counts.items()}
            elif strategy == 'sqrt_inverse':
                class_weights = {c: 1.0 / math.sqrt(count) for c, count in class_counts.items()}
            elif strategy == 'uniform':
                class_weights = {c: 1.0 for c in class_counts}
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            sample_weights = [class_weights[label] for label in labels]
            return WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(labels),
                replacement=True,
            )
        """
        labels = self.dataset.get_labels()
        class_counts = Counter(labels)

        if self.strategy == 'oversample_minority':
            self.class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        elif self.strategy == 'sqrt_inverse':
            self.class_weights = {c: 1.0 / math.sqrt(count) for c, count in class_counts.items()}
        elif self.strategy == 'uniform':
            self.class_weights = {c: 1.0 for c in class_counts}
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. "
                             "Choose from: oversample_minority, sqrt_inverse, uniform")

        total_weight = sum(self.class_weights.values())
        n = len(labels)
        self.effective_class_counts = {
            c: self.class_weights[c] / total_weight * n
            for c in self.class_weights
        }

        sample_weights = torch.tensor(
            [self.class_weights[label] for label in labels],
            dtype=torch.float32,
        )

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )
```

---

## File 7: `tests/test_dataset.py`

Write this file **exactly** as shown. Do not add, remove, or rename any test.

```python
"""
Test suite for modules/dataset/
NOTE: Tests that require the Malimg dataset are marked @pytest.mark.integration.
Unit tests (no dataset needed) run without the dataset.

Run unit tests only (CI-safe):
    pytest tests/test_dataset.py -v -m "not integration"

Run all tests (requires Malimg at config.DATA_DIR):
    pytest tests/test_dataset.py -v
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from modules.dataset.preprocessor import (
    normalize_image, encode_labels, validate_dataset_integrity,
    save_class_names, load_class_names,
)


class TestNormalizeImage:
    def test_output_range(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.dtype == np.float32

    def test_zero_maps_to_zero(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        assert normalize_image(arr).max() == 0.0

    def test_255_maps_to_one(self):
        arr = np.full((4, 4), 255, dtype=np.uint8)
        np.testing.assert_almost_equal(normalize_image(arr).min(), 1.0, decimal=6)

    def test_shape_preserved(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.shape == sample_grayscale_array.shape

    def test_midpoint_maps_correctly(self):
        arr = np.full((4, 4), 128, dtype=np.uint8)
        result = normalize_image(arr)
        np.testing.assert_almost_equal(result[0, 0], 128 / 255.0, decimal=6)


class TestEncodeLabels:
    def test_sorted_alphabetically(self):
        result = encode_labels(['Yuner.A', 'Allaple.A', 'VB.AT'])
        assert result == {'Allaple.A': 0, 'VB.AT': 1, 'Yuner.A': 2}

    def test_unique_integers(self):
        families = ['A', 'B', 'C', 'D']
        result = encode_labels(families)
        assert len(set(result.values())) == 4

    def test_range_correct(self):
        families = ['X', 'Y', 'Z']
        result = encode_labels(families)
        assert set(result.values()) == {0, 1, 2}

    def test_deterministic(self):
        f = ['Yuner.A', 'Allaple.A']
        assert encode_labels(f) == encode_labels(f)

    def test_single_family(self):
        assert encode_labels(['OnlyOne']) == {'OnlyOne': 0}

    def test_order_independent(self):
        f1 = ['C', 'A', 'B']
        f2 = ['A', 'B', 'C']
        assert encode_labels(f1) == encode_labels(f2)

    def test_returns_dict(self):
        result = encode_labels(['X', 'Y'])
        assert isinstance(result, dict)

    def test_all_values_are_ints(self):
        result = encode_labels(['A', 'B', 'C'])
        assert all(isinstance(v, int) for v in result.values())


class TestSaveLoadClassNames:
    def test_roundtrip(self, tmp_path):
        names = ['Allaple.A', 'Agent.FYI', 'VB.AT']
        path = tmp_path / 'class_names.json'
        save_class_names(names, path)
        loaded = load_class_names(path)
        assert loaded == names

    def test_creates_parent_dirs(self, tmp_path):
        names = ['A', 'B']
        path = tmp_path / 'subdir' / 'nested' / 'class_names.json'
        save_class_names(names, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        missing = tmp_path / 'nonexistent.json'
        with pytest.raises(FileNotFoundError, match="class_names.json not found"):
            load_class_names(missing)

    def test_file_format_correct(self, tmp_path):
        import json
        names = ['A', 'B', 'C']
        path = tmp_path / 'class_names.json'
        save_class_names(names, path)
        with open(path) as f:
            data = json.load(f)
        assert 'class_names' in data
        assert data['class_names'] == names

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / 'class_names.json'
        save_class_names(['X'], path)
        save_class_names(['A', 'B'], path)
        loaded = load_class_names(path)
        assert loaded == ['A', 'B']


class TestValidateDatasetIntegrity:
    def test_missing_dir_raises(self, tmp_path):
        missing = tmp_path / 'nonexistent'
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_dataset_integrity(missing)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="empty"):
            validate_dataset_integrity(tmp_path)

    def test_returns_required_keys(self, tmp_path):
        """Create a minimal fake dataset with one family and one valid PNG."""
        import cv2
        family_dir = tmp_path / 'FamilyA'
        family_dir.mkdir()
        # Create a minimal valid grayscale PNG
        img = np.zeros((16, 16), dtype=np.uint8)
        img_path = family_dir / 'sample.png'
        cv2.imwrite(str(img_path), img)

        report = validate_dataset_integrity(tmp_path)
        required_keys = {
            'valid', 'families', 'counts', 'total',
            'min_class', 'max_class', 'imbalance_ratio',
            'corrupt_files', 'missing_dirs'
        }
        assert required_keys.issubset(report.keys())

    def test_counts_and_total_correct(self, tmp_path):
        import cv2
        for family in ['FamilyA', 'FamilyB']:
            d = tmp_path / family
            d.mkdir()
            for i in range(3):
                img = np.zeros((16, 16), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i}.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert report['total'] == 6
        assert report['counts']['FamilyA'] == 3
        assert report['counts']['FamilyB'] == 3

    def test_corrupt_file_detection(self, tmp_path):
        """A corrupt PNG (not a valid image) should appear in corrupt_files."""
        family_dir = tmp_path / 'FamilyA'
        family_dir.mkdir()
        bad_png = family_dir / 'bad.png'
        bad_png.write_bytes(b'not an image')  # cv2.imread will return None

        report = validate_dataset_integrity(tmp_path)
        assert len(report['corrupt_files']) == 1
        assert report['valid'] is False

    def test_families_sorted(self, tmp_path):
        import cv2
        for name in ['Zebra', 'Alligator', 'Monkey']:
            d = tmp_path / name
            d.mkdir()
            img = np.zeros((8, 8), dtype=np.uint8)
            cv2.imwrite(str(d / 'img.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert report['families'] == ['Alligator', 'Monkey', 'Zebra']

    def test_imbalance_ratio(self, tmp_path):
        import cv2
        # FamilyA has 1 sample, FamilyB has 4 samples → ratio = 4.0
        for name, count in [('FamilyA', 1), ('FamilyB', 4)]:
            d = tmp_path / name
            d.mkdir()
            for i in range(count):
                img = np.zeros((8, 8), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i}.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert abs(report['imbalance_ratio'] - 4.0) < 1e-6

    def test_missing_dirs_always_empty_list(self, tmp_path):
        import cv2
        d = tmp_path / 'FamilyA'
        d.mkdir()
        img = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(d / 'img.png'), img)
        report = validate_dataset_integrity(tmp_path)
        assert report['missing_dirs'] == []

    @pytest.mark.integration
    def test_malimg_dataset_valid(self):
        """Requires real Malimg dataset at config.DATA_DIR."""
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found at DATA_DIR")
        report = validate_dataset_integrity(config.DATA_DIR)
        assert report['total'] > 0
        assert len(report['families']) == 25
        assert len(report['corrupt_files']) == 0


class TestMalimgDataset:
    """Unit tests that build a tiny fake dataset (no real Malimg needed)."""

    @pytest.fixture
    def fake_data_dir(self, tmp_path):
        """3 families × 5 samples each = 15 total images."""
        import cv2
        for family in ['FamilyA', 'FamilyB', 'FamilyC']:
            d = tmp_path / family
            d.mkdir()
            for i in range(5):
                img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i:03d}.png'), img)
        return tmp_path

    def test_invalid_split_raises(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        with pytest.raises(ValueError, match="split"):
            MalimgDataset(fake_data_dir, 'invalid_split')

    def test_missing_data_dir_raises(self, tmp_path):
        from modules.dataset.loader import MalimgDataset
        missing = tmp_path / 'does_not_exist'
        with pytest.raises(FileNotFoundError):
            MalimgDataset(missing, 'train')

    def test_split_sizes_sum_to_total(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        train_ds = MalimgDataset(fake_data_dir, 'train')
        val_ds   = MalimgDataset(fake_data_dir, 'val')
        test_ds  = MalimgDataset(fake_data_dir, 'test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 15

    def test_getitem_tensor_shape(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        tensor, label = ds[0]
        assert tensor.shape == (1, 128, 128)

    def test_getitem_tensor_dtype(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        tensor, label = ds[0]
        assert tensor.dtype == torch.float32

    def test_getitem_label_is_int(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        _, label = ds[0]
        assert isinstance(label, int)

    def test_class_names_sorted(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert ds.class_names == sorted(ds.class_names)

    def test_label_map_keys_match_class_names(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert set(ds.label_map.keys()) == set(ds.class_names)

    def test_label_map_values_are_unique_ints(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        values = list(ds.label_map.values())
        assert len(set(values)) == len(values)
        assert all(isinstance(v, int) for v in values)

    def test_get_labels_length_matches_len(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert len(ds.get_labels()) == len(ds)

    def test_splits_are_reproducible(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds1 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        ds2 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        paths1 = [str(p) for p, _ in ds1.samples]
        paths2 = [str(p) for p, _ in ds2.samples]
        assert paths1 == paths2

    def test_different_seeds_produce_different_splits(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds1 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        ds2 = MalimgDataset(fake_data_dir, 'train', random_seed=99)
        # With 15 samples, different seeds should produce different orderings
        paths1 = set(str(p) for p, _ in ds1.samples)
        paths2 = set(str(p) for p, _ in ds2.samples)
        # Not guaranteed to differ but very likely with different seeds
        # At minimum, verify no crash
        assert isinstance(paths1, set)
        assert isinstance(paths2, set)

    @pytest.mark.integration
    def test_malimg_split_sizes_sum_correctly(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        train_ds = MalimgDataset(config.DATA_DIR, 'train')
        val_ds   = MalimgDataset(config.DATA_DIR, 'val')
        test_ds  = MalimgDataset(config.DATA_DIR, 'test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert 9000 < total < 9500

    @pytest.mark.integration
    def test_malimg_getitem_tensor_shape(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(config.DATA_DIR, 'train')
        tensor, label = ds[0]
        assert tensor.shape == (1, 128, 128)
        assert tensor.dtype == torch.float32
        assert isinstance(label, int)

    @pytest.mark.integration
    def test_malimg_all_splits_contain_all_classes(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        for split in ['train', 'val', 'test']:
            ds = MalimgDataset(config.DATA_DIR, split)
            labels_in_split = set(ds.get_labels())
            assert len(labels_in_split) == 25, \
                f"Split '{split}' missing classes: {25 - len(labels_in_split)} absent"
```

---

## Definition of Done

Run these commands after implementing. All must pass before Phase 2 is complete.

```bash
# Unit tests only (no dataset required — these must all pass clean)
pytest tests/test_dataset.py -v -m "not integration"

# Expected output (all unit tests):
# tests/test_dataset.py::TestNormalizeImage::test_output_range PASSED
# tests/test_dataset.py::TestNormalizeImage::test_output_dtype_float32 PASSED
# tests/test_dataset.py::TestNormalizeImage::test_zero_maps_to_zero PASSED
# tests/test_dataset.py::TestNormalizeImage::test_255_maps_to_one PASSED
# tests/test_dataset.py::TestNormalizeImage::test_shape_preserved PASSED
# tests/test_dataset.py::TestNormalizeImage::test_midpoint_maps_correctly PASSED
# tests/test_dataset.py::TestEncodeLabels::test_sorted_alphabetically PASSED
# tests/test_dataset.py::TestEncodeLabels::test_unique_integers PASSED
# tests/test_dataset.py::TestEncodeLabels::test_range_correct PASSED
# tests/test_dataset.py::TestEncodeLabels::test_deterministic PASSED
# tests/test_dataset.py::TestEncodeLabels::test_single_family PASSED
# tests/test_dataset.py::TestEncodeLabels::test_order_independent PASSED
# tests/test_dataset.py::TestEncodeLabels::test_returns_dict PASSED
# tests/test_dataset.py::TestEncodeLabels::test_all_values_are_ints PASSED
# tests/test_dataset.py::TestSaveLoadClassNames::test_roundtrip PASSED
# tests/test_dataset.py::TestSaveLoadClassNames::test_creates_parent_dirs PASSED
# tests/test_dataset.py::TestSaveLoadClassNames::test_load_nonexistent_raises PASSED
# tests/test_dataset.py::TestSaveLoadClassNames::test_file_format_correct PASSED
# tests/test_dataset.py::TestSaveLoadClassNames::test_overwrites_existing PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_missing_dir_raises PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_empty_dir_raises PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_returns_required_keys PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_counts_and_total_correct PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_corrupt_file_detection PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_families_sorted PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_imbalance_ratio PASSED
# tests/test_dataset.py::TestValidateDatasetIntegrity::test_missing_dirs_always_empty_list PASSED
# tests/test_dataset.py::TestMalimgDataset::test_invalid_split_raises PASSED
# tests/test_dataset.py::TestMalimgDataset::test_missing_data_dir_raises PASSED
# tests/test_dataset.py::TestMalimgDataset::test_split_sizes_sum_to_total PASSED
# tests/test_dataset.py::TestMalimgDataset::test_getitem_tensor_shape PASSED
# tests/test_dataset.py::TestMalimgDataset::test_getitem_tensor_dtype PASSED
# tests/test_dataset.py::TestMalimgDataset::test_getitem_label_is_int PASSED
# tests/test_dataset.py::TestMalimgDataset::test_class_names_sorted PASSED
# tests/test_dataset.py::TestMalimgDataset::test_label_map_keys_match_class_names PASSED
# tests/test_dataset.py::TestMalimgDataset::test_label_map_values_are_unique_ints PASSED
# tests/test_dataset.py::TestMalimgDataset::test_get_labels_length_matches_len PASSED
# tests/test_dataset.py::TestMalimgDataset::test_splits_are_reproducible PASSED
# tests/test_dataset.py::TestMalimgDataset::test_different_seeds_produce_different_splits PASSED
#
# ====== X passed, 3 deselected (integration) ======

# Phase 1 tests must still pass (no regressions)
pytest tests/test_converter.py -v
```

If you have the Malimg dataset available:
```bash
# Run all tests including integration
pytest tests/test_dataset.py -v
```

### Checklist

- [ ] `pytest tests/test_dataset.py -v -m "not integration"` passes with zero failures
- [ ] `pytest tests/test_converter.py -v` still passes (no regressions)
- [ ] `modules/dataset/__init__.py` exports `MalimgDataset`, `get_dataloaders`, `validate_dataset_integrity`
- [ ] `modules/enhancement/__init__.py` exports `get_train_transforms`, `get_val_transforms`, `GaussianNoise`, `ClassAwareOversampler`
- [ ] `cv2.imread()` uses `cv2.IMREAD_GRAYSCALE` everywhere — never loads as BGR
- [ ] `cv2.resize()` uses `(img_size, img_size)` — NOT `(height, width)` or `(width, height)` reversed
- [ ] `Normalize` uses `mean=[0.5], std=[0.5]` (single-element lists, not scalars)
- [ ] `ColorJitter` is before `ToTensor()` in the train pipeline
- [ ] `GaussianNoise` is after `ToTensor()` in the train pipeline
- [ ] `encode_labels()` sorts alphabetically before enumerating
- [ ] `train_test_split` uses `random_state=config.RANDOM_SEED`
- [ ] No external network calls anywhere in these modules

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|-----|---------|-----|
| `cv2.resize((height, width))` | Images load with transposed dimensions | Use `(width, height)` — i.e. `(img_size, img_size)` for square |
| `Normalize(mean=0.5, std=0.5)` scalar | Runtime error or wrong normalization | Must be `mean=[0.5], std=[0.5]` (lists) |
| `ColorJitter` after `ToTensor` | TypeError: expects PIL Image | Move `ColorJitter` before `ToTensor` |
| `GaussianNoise` before `ToTensor` | TypeError: tensor expected | Move `GaussianNoise` after `ToTensor` |
| Forgetting `sorted()` in `encode_labels` | Non-deterministic label mapping | Always sort before enumerating |
| `shuffle=True` on val/test loader | Non-reproducible evaluation | Only train loader uses sampler; val/test use `shuffle=False` |
| `sampler=` AND `shuffle=True` together | DataLoader raises ValueError | Use either `sampler=` or `shuffle=True`, never both |
| Forgetting `drop_last=True` on train loader | Single-sample final batch crashes BatchNorm | Set `drop_last=True` on train DataLoader only |
| `WeightedRandomSampler(replacement=False)` | Cannot oversample minority classes | Must use `replacement=True` |

---

*Phase 2 complete → proceed to Phase 3: Enhancement tests + Detection model.*

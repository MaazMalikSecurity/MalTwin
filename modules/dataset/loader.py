import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from modules.dataset.preprocessor import encode_labels, normalize_image


class MalimgDataset(Dataset):
    """
    PyTorch Dataset wrapping the Malimg directory structure.
    SRS ref: FE-1, FE-2, FE-3 of Module 3
    """

    def __init__(
        self,
        data_dir: Path,
        img_size: int = None,
        transform: Optional[Callable] = None,
        split: str = "train",
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size if img_size is not None else config.IMG_SIZE
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
        self.val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
        self.test_ratio = test_ratio if test_ratio is not None else config.TEST_RATIO

        families = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self._label_map = encode_labels(families)
        self._class_names = sorted(self._label_map.keys())

        all_samples: list[tuple[Path, int]] = []
        for family in families:
            family_dir = self.data_dir / family
            label = self._label_map[family]
            for img_path in sorted(family_dir.glob("*.png")) + sorted(family_dir.glob("*.bmp")):
                all_samples.append((img_path, label))

        all_paths = [s[0] for s in all_samples]
        all_labels = [s[1] for s in all_samples]

        # First split off test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths,
            all_labels,
            test_size=self.test_ratio,
            stratify=all_labels,
            random_state=42,
        )

        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=42,
        )

        if split == "train":
            self._samples = list(zip(train_paths, train_labels))
        elif split == "val":
            self._samples = list(zip(val_paths, val_labels))
        elif split == "test":
            self._samples = list(zip(test_paths, test_labels))
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        img_path, label = self._samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = normalize_image(img)  # [0, 1] float32
        # Add channel dim → (1, H, W)
        img = np.expand_dims(img, axis=0)

        if self.transform is not None:
            # Transform expects (H, W, C) PIL or (H, W) array depending on pipeline;
            # Our transforms operate on tensors post ToTensor(), so pass tensor directly
            tensor = torch.from_numpy(img)
            tensor = self.transform(tensor)
        else:
            tensor = torch.from_numpy(img)

        return tensor, label

    @property
    def class_names(self) -> list:
        return self._class_names

    @property
    def class_counts(self) -> dict:
        counts: dict[str, int] = {name: 0 for name in self._class_names}
        inv_map = {v: k for k, v in self._label_map.items()}
        for _, label in self._samples:
            counts[inv_map[label]] += 1
        return counts


def get_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool = True,
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader).
    SRS ref: FE-3 of Module 3
    """
    from modules.enhancement.augmentor import get_train_transforms, get_val_transforms

    train_transform = get_train_transforms(img_size) if augment_train else get_val_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    train_dataset = MalimgDataset(data_dir, img_size=img_size, transform=train_transform, split="train")
    val_dataset = MalimgDataset(data_dir, img_size=img_size, transform=val_transform, split="val")
    test_dataset = MalimgDataset(data_dir, img_size=img_size, transform=val_transform, split="test")

    from modules.enhancement.balancer import ClassAwareOversampler
    sampler = ClassAwareOversampler(train_dataset).get_sampler()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Save class_names.json for dashboard use
    class_names_path = Path(config.PROCESSED_DIR) / "class_names.json"
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(train_dataset.class_names, f)

    return train_loader, val_loader, test_loader

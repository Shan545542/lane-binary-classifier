from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

CLASS_TO_TARGET = {"out_lane": 0, "in_lane": 1}
TARGET_TO_CLASS = {v: k for k, v in CLASS_TO_TARGET.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageConfig:
    width: int = 160
    height: int = 96
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class LaneImageFolder(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_config: ImageConfig,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_config = image_config
        self.augment = augment
        self.samples = self._collect_samples(self.root)
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}")

    @staticmethod
    def _collect_samples(root: Path) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name, target in CLASS_TO_TARGET.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_config.width, self.image_config.height), Image.BILINEAR)

        if self.augment:
            image = self._augment(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array(self.image_config.mean)) / np.array(self.image_config.std)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        label = torch.tensor(float(target), dtype=torch.float32)
        return tensor, label

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.8:
            factor = random.uniform(0.65, 1.35)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image


def make_loaders(
    data_dir: str | Path,
    image_config: ImageConfig,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    train_dataset = LaneImageFolder(data_dir / "train", image_config=image_config, augment=True)
    val_dataset = LaneImageFolder(data_dir / "val", image_config=image_config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def describe_class_balance(samples: Iterable[tuple[Path, int]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_TO_TARGET}
    for _, target in samples:
        counts[TARGET_TO_CLASS[target]] += 1
    return counts

"""
MVTec AD Dataset loader for PatchCore.
Supports train (normal only) and test (all defect types + GT masks).
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MASK_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.NEAREST),
    T.CenterCrop(224),
    T.ToTensor(),
])


class MVTecTrainDataset(Dataset):
    """Loads train/good/ images for a single MVTec category."""

    def __init__(self, root: str, category: str):
        self.root = Path(root)
        self.category = category
        good_dir = self.root / category / "train" / "good"
        self.image_paths = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No training images found in {good_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return IMAGE_TRANSFORM(img)


class MVTecTestDataset(Dataset):
    """
    Loads all test images for a single MVTec category.
    Returns (image_tensor, label, gt_mask_tensor, defect_type).
    label: 0 = good, 1 = defective.
    gt_mask: [1, H, W] binary tensor; zeros for 'good' samples.
    """

    def __init__(self, root: str, category: str):
        self.root = Path(root)
        self.category = category
        test_dir = self.root / category / "test"
        gt_dir   = self.root / category / "ground_truth"

        self.samples = []  # (img_path, label, mask_path_or_None, defect_type)

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted(list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))):
                mask_path = None
                if label == 1:
                    # GT masks use same stem with _mask suffix or same name
                    candidate = gt_dir / defect_type / (img_path.stem + "_mask.png")
                    if not candidate.exists():
                        candidate = gt_dir / defect_type / img_path.name
                    if candidate.exists():
                        mask_path = candidate
                self.samples.append((img_path, label, mask_path, defect_type))

        if len(self.samples) == 0:
            raise RuntimeError(f"No test images found in {test_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
        img_path, label, mask_path, defect_type = self.samples[idx]

        img = Image.open(img_path)
        img_tensor = IMAGE_TRANSFORM(img)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = MASK_TRANSFORM(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros(1, 224, 224)

        return img_tensor, label, mask_tensor, defect_type


def get_dataloaders(
    root: str,
    category: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MVTecTrainDataset(root, category)
    test_ds  = MVTecTestDataset(root, category)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader

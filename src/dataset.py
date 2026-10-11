"""
Dataset loading, augmentations, and color perturbation utilities for Wood Grain Classification.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COLOR_SHIFTS = {
    "original":   lambda im: im,
    "grayscale":  lambda im: TF.to_grayscale(im, num_output_channels=3),
    "desaturate": lambda im: TF.adjust_saturation(im, 0.3),
    "hue_+0.2":   lambda im: TF.adjust_hue(im, 0.2),
    "hue_-0.2":   lambda im: TF.adjust_hue(im, -0.2),
    "dark":       lambda im: TF.adjust_brightness(im, 0.6),
    "bright":     lambda im: TF.adjust_brightness(im, 1.4),
}

def build_train_transform(
    img_size: int = 224,
    color_aug: bool = True,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
    grayscale_p: float = 0.2
) -> T.Compose:
    """Build data augmentation pipeline for training."""
    ops = [
        T.Resize((256, 256)),
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ]
    if color_aug:
        ops.extend([
            T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            T.RandomGrayscale(p=grayscale_p),
        ])
    ops.extend([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return T.Compose(ops)

def build_eval_transform(img_size: int = 224) -> T.Compose:
    """Build deterministic transform for validation and test."""
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

class WoodDataset(Dataset):
    """
    PyTorch Dataset for wood grain macroscopic images.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        color_shift_name: str = "original"
    ):
        self.samples = samples  # List of (filepath, class_idx)
        self.transform = transform
        self.color_shift_fn = COLOR_SHIFTS.get(color_shift_name, lambda im: im)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = self.color_shift_fn(im)
            if self.transform is not None:
                tensor = self.transform(im)
            else:
                tensor = TF.to_tensor(im)
        return tensor, label

def get_dataloader(
    samples: List[Tuple[str, int]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    color_shift_name: str = "original",
    drop_last: bool = False
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    dataset = WoodDataset(samples, transform=transform, color_shift_name=color_shift_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )

def load_split_samples(
    splits_json_path: str,
    split_name: str,
    base_dir: Optional[Path] = None
) -> Tuple[List[Tuple[str, int]], Dict]:
    """
    Load samples for a given split ('train', 'val', or 'test') from splits.json.
    Returns:
        samples: List of (absolute_image_path, label_idx)
        metadata: Metadata dictionary containing class mappings
    """
    import json
    p = Path(splits_json_path)
    if not p.exists():
        raise FileNotFoundError(f"splits.json not found at: {splits_json_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split_name not in data:
        raise ValueError(f"Split {split_name} not found in splits.json")
    
    if base_dir is None:
        base_dir = p.resolve().parent.parent

    samples = []
    for item in data[split_name]:
        full_path = str(base_dir / item["path"])
        samples.append((full_path, int(item["label"])))
    return samples, data.get("metadata", {})


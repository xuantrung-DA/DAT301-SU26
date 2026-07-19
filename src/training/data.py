from __future__ import annotations

from pathlib import Path
import csv
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox_tensor(image: Image.Image, image_size: int, fill: int = 114) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio and symmetric padding."""
    width, height = image.size
    ratio = min(image_size / width, image_size / height)
    resized_width, resized_height = int(round(width * ratio)), int(round(height * ratio))
    resized = TF.resize(image, [resized_height, resized_width], antialias=True)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas = Image.new("RGB", (image_size, image_size), color=(fill, fill, fill))
    canvas.paste(resized, (pad_x, pad_y))
    return TF.to_tensor(canvas), ratio, (pad_x, pad_y)


class PairedImageDataset(Dataset):
    """LOL-style paired dataset with matching filenames in low/high folders."""
    def __init__(self, low_dir: Path, high_dir: Path, image_size: int = 640, augment: bool = False, max_samples: int | None = None):
        self.low_dir, self.high_dir = Path(low_dir), Path(high_dir)
        self.image_size, self.augment = image_size, augment
        self.samples = []
        for low in sorted(self.low_dir.iterdir() if self.low_dir.exists() else []):
            if low.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            high = self.high_dir / low.name
            if high.exists():
                self.samples.append((low, high))
        if not self.samples:
            raise FileNotFoundError(f"No paired images found in {self.low_dir} and {self.high_dir}")
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        low_path, high_path = self.samples[index]
        low = Image.open(low_path).convert("RGB"); high = Image.open(high_path).convert("RGB")
        low = TF.resize(low, [self.image_size, self.image_size], antialias=True)
        high = TF.resize(high, [self.image_size, self.image_size], antialias=True)
        if self.augment and random.random() < 0.5:
            low, high = TF.hflip(low), TF.hflip(high)
        return {"low": TF.to_tensor(low), "high": TF.to_tensor(high), "path": str(low_path)}


class YoloEnhancementDataset(Dataset):
    """YOLO images/labels plus optional aligned clean targets for joint training."""
    def __init__(self, image_dir: Path, label_dir: Path, clean_dir: Path | None = None, image_size: int = 640, max_samples: int | None = None):
        self.image_dir, self.label_dir = Path(image_dir), Path(label_dir)
        self.clean_dir, self.image_size = Path(clean_dir) if clean_dir else None, image_size
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.clean_flags: dict[str, bool] = {}
        manifest = self.image_dir.parents[1] / "manifest.csv"
        if manifest.exists():
            with manifest.open("r", encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    value = str(row.get("is_clean", "")).strip().lower()
                    self.clean_flags[Path(row.get("output", "")).name] = value in {"1", "true", "yes"}
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        if max_samples:
            self.images = self.images[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        pil_image = Image.open(path).convert("RGB")
        original_width, original_height = pil_image.size
        image, ratio, (pad_x, pad_y) = letterbox_tensor(pil_image, self.image_size)
        clean = image
        if self.clean_dir and (self.clean_dir / path.name).exists():
            clean, _, _ = letterbox_tensor(Image.open(self.clean_dir / path.name).convert("RGB"), self.image_size)
        labels = []
        label_path = self.label_dir / f"{path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = [float(x) for x in line.split()[:5]]
                if len(values) == 5:
                    cls, cx, cy, width, height = values
                    cx = (cx * original_width * ratio + pad_x) / self.image_size
                    cy = (cy * original_height * ratio + pad_y) / self.image_size
                    width = width * original_width * ratio / self.image_size
                    height = height * original_height * ratio / self.image_size
                    labels.append([cls, cx, cy, width, height])
        is_clean = self.clean_flags.get(path.name, bool(torch.mean(torch.abs(image - clean)) < 1.5 / 255.0))
        return {
            "low": image,
            "high": clean,
            "labels": torch.tensor(labels, dtype=torch.float32).reshape(-1, 5),
            "is_clean": is_clean,
            "original_size": (original_height, original_width),
            "path": str(path),
        }


def detection_collate(batch):
    return {"low": torch.stack([x["low"] for x in batch]), "high": torch.stack([x["high"] for x in batch]),
            "labels": [x["labels"] for x in batch], "is_clean": torch.tensor([x["is_clean"] for x in batch], dtype=torch.bool),
            "original_size": [x["original_size"] for x in batch], "path": [x["path"] for x in batch]}

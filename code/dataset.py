"""DeepWeeds dataset utilities.

The module deliberately keeps the CSV files authoritative: it never re-splits
the data. This makes every experiment comparable with the course baseline.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The author repository's split CSVs contain Filename/Label; labels.csv also
# carries Species.  Do not require Species from the split files.
_REQUIRED_COLUMNS = {"Filename", "Label"}


def load_split(labels_dir: str | Path, fold: int = 0):
    """Load the original train/validation/test CSVs for one fold unchanged."""
    if not isinstance(fold, int) or fold < 0 or fold > 4:
        raise ValueError("fold must be an integer from 0 to 4")
    root = Path(labels_dir)
    frames = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        frame = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Validate S1--S4 and return counts suitable for a report."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    sets: dict[str, set[str]] = {}
    per_class: dict[str, dict[int, int]] = {}
    counts: dict[str, int] = {}
    for name, frame in frames.items():
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{name} split contains an empty Filename or Label")
        if not frame["Label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{name} split has labels outside 0..{NUM_CLASSES - 1}")
        names = frame["Filename"].astype(str)
        if names.duplicated().any():
            raise ValueError(f"{name} split contains duplicate filenames")
        sets[name] = set(names)
        counts[name] = len(frame)
        per_class[name] = {int(k): int(v) for k, v in frame["Label"].value_counts().sort_index().items()}

    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Data leakage: split overlap found: {overlaps}")
    all_names = sets["train"] | sets["val"] | sets["test"]
    if len(all_names) != 17_509:
        raise ValueError(f"Expected 17,509 unique DeepWeeds images, found {len(all_names)}")
    image_root = Path(images_dir)
    missing_files = [name for name in all_names if not (image_root / name).is_file()]
    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"{len(missing_files)} CSV images are absent from {image_root}: {preview}")
    result = {"n": counts, "per_class": per_class, "overlap": overlaps, "union": len(all_names)}
    print(result)
    return result


class _FallbackTransform:
    """Small torchvision-free transform used in minimal notebook environments."""
    def __init__(self, train: bool, img_size: int, aug: str):
        self.train, self.img_size, self.aug = train, img_size, aug

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        w, h = image.size
        if self.train:
            scale = random.uniform(0.80, 1.0)
            side = max(1, int(min(w, h) * scale))
            left, top = random.randint(0, w - side), random.randint(0, h - side)
            image = image.crop((left, top, left + side, top + side))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.aug in {"color", "trivial", "randaug"}:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        else:
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            image = image.crop((left, top, left + side, top + side))
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic evaluation and stochastic training transforms."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError("aug must be one of: basic, color, trivial, randaug")
    try:
        from torchvision import transforms as T
        ops = []
        if train:
            ops += [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
            if aug == "color":
                ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
            elif aug == "trivial":
                ops.append(T.TrivialAugmentWide())
            elif aug == "randaug":
                ops.append(T.RandAugment())
        else:
            ops += [T.CenterCrop(img_size)]
        return T.Compose([*ops, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    except ImportError:
        return _FallbackTransform(train, img_size, aug)


class DeepWeedsDataset(Dataset):
    """A PIL-backed dataset returning ``(image, int_label, filename)``."""
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row.Filename)
        path = self.images_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image = self.transform(image) if self.transform is not None else image
        return image, int(row.Label), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Create a reproducible loader, optionally using inverse-frequency sampling."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if sampler not in {None, "balanced"}:
        raise ValueError("sampler must be None or 'balanced'")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    sample_sampler = None
    if sampler == "balanced":
        labels = df["Label"].astype(int)
        class_count = labels.value_counts()
        weights = labels.map(lambda label: 1.0 / class_count[label]).to_numpy(dtype=np.float64)
        sample_sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train and sample_sampler is None,
        sampler=sample_sampler, drop_last=train, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )

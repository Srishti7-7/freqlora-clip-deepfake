"""FaceForensics++ frame dataloaders with reproducible splits and compression support."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.compression import validate_compression


class FaceForensicsDataset(Dataset):
    """Load extracted FF++ frames.

    Layout:
        image_dir/real/*.png
        image_dir/fake/*.png

    If ``compression_level`` is C23/C40, frames are read from
    ``image_dir/C23`` or ``image_dir/C40`` when those directories exist.
    C0 reads the original ``image_dir``.
    """

    def __init__(self, image_dir, split="train", ratio=0.8,
                 compression_level="C0", seed=0):
        self.base_dir = Path(image_dir)
        self.compression_level = validate_compression(compression_level)
        self.preprocess = self._get_preprocess()

        candidate = self.base_dir if self.compression_level == "C0" else self.base_dir / self.compression_level
        if not (candidate / "real").exists() or not (candidate / "fake").exists():
            if self.compression_level != "C0":
                raise FileNotFoundError(
                    f"Missing {self.compression_level} frames at {candidate}. "
                    "Generate them with data.compression.build_compressed_image_dataset()."
                )
            raise FileNotFoundError(f"Expected real/ and fake/ under {candidate}")

        real_images = sorted(candidate.glob("real/*.png"))
        fake_images = sorted(candidate.glob("fake/*.png"))
        all_samples = [(p, 0) for p in real_images] + [(p, 1) for p in fake_images]
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        if split == "all":
            self.samples = all_samples
        elif split == "train":
            split_idx = int(len(all_samples) * ratio)
            self.samples = all_samples[:split_idx]
        elif split == "test":
            split_idx = int(len(all_samples) * ratio)
            self.samples = all_samples[split_idx:]
        else:
            raise ValueError("split must be 'train', 'test', or 'all'")

    @staticmethod
    def _get_preprocess():
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        return preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.preprocess(img)
        return tensor, label


class FewShotSampler:
    """Sample K examples per class reproducibly."""

    def __init__(self, image_dir, k=5, seed=0):
        if k < 1:
            raise ValueError("k must be >= 1")
        root = Path(image_dir)
        rng = random.Random(seed)
        real = sorted(root.glob("real/*.png"))
        fake = sorted(root.glob("fake/*.png"))
        rng.shuffle(real)
        rng.shuffle(fake)
        real, fake = real[:k], fake[:k]
        self.images = real + fake
        self.labels = [0] * len(real) + [1] * len(fake)
        self.seed = seed

    def get_batch(self, device="cuda"):
        if not self.images:
            raise RuntimeError("No few-shot images found")
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        batch = []
        for img_path in self.images:
            with Image.open(img_path) as img:
                batch.append(preprocess(img.convert("RGB")))
        return torch.stack(batch).to(device), torch.tensor(self.labels, device=device)


def get_dataloader(image_dir, batch_size=64, split="train", num_workers=4,
                   compression_level="C0", seed=0):
    dataset = FaceForensicsDataset(
        image_dir, split=split, compression_level=compression_level, seed=seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_few_shot_data(image_dir, k=5, seed=0, device="cuda"):
    return FewShotSampler(image_dir, k=k, seed=seed).get_batch(device=device)

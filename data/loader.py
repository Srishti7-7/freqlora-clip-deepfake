"""
data/loader.py - FaceForensics++ dataloader with compression
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import cv2
import numpy as np

class FaceForensicsDataset(Dataset):
    """Load FaceForensics++ with compression levels"""
    
    def __init__(self, image_dir, split='train', ratio=0.8, 
                 compression_level="C23"):
        """
        image_dir: path to extracted frames
        compression_level: "C0" (original), "C23", "C40"
        """
        self.preprocess = self._get_preprocess()
        self.compression_level = compression_level
        
        # Load all images
        real_images = list(Path(image_dir).glob("real/*.png"))
        fake_images = list(Path(image_dir).glob("fake/*.png"))
        
        all_samples = [(img, 0) for img in real_images] + \
                     [(img, 1) for img in fake_images]
        
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * ratio)
        
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
    
    def _get_preprocess(self):
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        return preprocess
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.preprocess(img), label


class FewShotSampler:
    """Few-shot data sampler"""
    
    def __init__(self, image_dir, k=5, seed=0):
        """Sample K examples per class"""
        random.seed(seed)
        
        real = list(Path(image_dir).glob("real/*.png"))[:k]
        fake = list(Path(image_dir).glob("fake/*.png"))[:k]
        
        self.images = real + fake
        self.labels = [0]*len(real) + [1]*len(fake)
        self.seed = seed
    
    def get_batch(self, device="cuda"):
        """Return K examples as single batch"""
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        
        batch = []
        for img_path in self.images:
            img = Image.open(img_path).convert('RGB')
            batch.append(preprocess(img))
        
        images = torch.stack(batch).to(device)
        labels = torch.tensor(self.labels).to(device)
        return images, labels


def get_dataloader(image_dir, batch_size=64, split='train', num_workers=4):
    """Get standard dataloader"""
    dataset = FaceForensicsDataset(image_dir, split=split)
    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=(split == 'train'), num_workers=num_workers)
    return loader


def get_few_shot_data(image_dir, k=5, seed=0, device="cuda"):
    """Get few-shot K examples"""
    sampler = FewShotSampler(image_dir, k=k, seed=seed)
    return sampler.get_batch(device=device)

"""
training/trainer.py - Few-shot trainer with seed management
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np

class FewShotTrainer:
    """Train model on few-shot support set"""
    
    def __init__(self, model, device="cuda", lr=1e-3, epochs=100):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
    
    def get_trainable_params(self):
        """Get only trainable parameters"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def train(self, support_images, support_labels):
        """Few-shot training loop"""
        
        trainable = self.get_trainable_params()
        if not trainable:
            # Model has no trainable params, just return
            return None
        
        optimizer = Adam(trainable, lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, self.epochs)
        
        losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            
            logits = self.model(support_images)
            loss = self.criterion(logits, support_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
        
        return np.mean(losses[-10:])  # Last 10 epoch average


def train_model(model, support_images, support_labels, device="cuda", epochs=100, lr=1e-3):
    """Train model on few-shot data."""
    trainer = FewShotTrainer(model, device=device, lr=lr, epochs=epochs)
    return trainer.train(support_images, support_labels)

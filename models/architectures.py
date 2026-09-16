"""
models/architectures.py - All 4 models
M1: CLIP only
M2: CLIP + Low-Rank Adapter
M3: CLIP + Frequency Features
M4: FreqLoRA-CLIP (Full)
"""

import torch
import torch.nn as nn
import clip
import numpy as np
from scipy.fftpack import dct

class M1_CLIPOnly(nn.Module):
    """M1: CLIP ViT-B/32 (frozen) + linear classifier"""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        
        # Simple classifier on top of CLIP
        self.classifier = nn.Linear(512, 2)
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            feat = self.clip_model.visual(x)  # [B, 512]
        return self.classifier(feat)
    
    def get_preprocess(self):
        return self.preprocess


class M2_CLIPAdapter(nn.Module):
    """M2: CLIP + Low-Rank Adapter (like LoRA)"""
    
    def __init__(self, rank=8, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        
        # Low-rank adapter
        self.rank = rank
        self.lora_down = nn.Linear(512, rank)
        self.lora_up = nn.Linear(rank, 512)
        
        # Classifier
        self.classifier = nn.Linear(512, 2)
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            feat = self.clip_model.visual(x)  # [B, 512]
        
        # Apply adapter
        adapter_out = self.lora_up(self.lora_down(feat))
        feat_adapted = feat + adapter_out  # Residual connection
        
        return self.classifier(feat_adapted)
    
    def get_preprocess(self):
        return self.preprocess


class FrequencyEncoder(nn.Module):
    """Extract DCT features robust to H.264 compression"""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Frequency attention
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
    
    def forward(self, x, freq_band="full"):
        """
        x: [B, 3, H, W]
        freq_band: "low", "mid", or "full"
        """
        B = x.size(0)
        gray = x.mean(dim=1)  # [B, H, W]
        
        freq_feats = []
        
        for i in range(B):
            img_np = gray[i].cpu().detach().numpy()
            
            # 2D DCT
            dct_2d = dct(dct(img_np, axis=0, norm='ortho'), 
                        axis=1, norm='ortho')
            
            # Extract frequency bands
            if freq_band in ["low", "full"]:
                low_freq = dct_2d[:56, :56].flatten()  # Top-left 56×56
            else:
                low_freq = np.zeros(56*56)
            
            if freq_band in ["mid", "full"]:
                mid_freq = dct_2d[56:112, 56:112].flatten()  # [56:112]×[56:112]
            else:
                mid_freq = np.zeros(56*56)
            
            combined = np.concatenate([low_freq, mid_freq])
            combined = combined[:self.feature_dim]
            
            if len(combined) < self.feature_dim:
                combined = np.pad(combined, (0, self.feature_dim - len(combined)))
            
            freq_feats.append(torch.tensor(combined, dtype=torch.float32))
        
        freq_feats = torch.stack(freq_feats).to(x.device)
        
        # Apply attention
        return self.attention(freq_feats)


class M3_CLIPDCTOnly(nn.Module):
    """M3: CLIP + DCT Frequency Features (no adapter)"""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        
        # Frequency encoder
        self.freq_encoder = FrequencyEncoder(feature_dim=256)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            spatial_feat = self.clip_model.visual(x)  # [B, 512]
        
        freq_feat = self.freq_encoder(x, freq_band="full")  # [B, 256]
        
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion(combined)
    
    def get_preprocess(self):
        return self.preprocess


class M4_FreqLoRACLIP(nn.Module):
    """M4: FreqLoRA-CLIP (Full model: CLIP + Adapter + Frequency)"""
    
    def __init__(self, rank=8, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        
        # Low-rank adapter
        self.rank = rank
        self.lora_down = nn.Linear(512, rank)
        self.lora_up = nn.Linear(rank, 512)
        
        # Frequency encoder
        self.freq_encoder = FrequencyEncoder(feature_dim=256)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            spatial_feat = self.clip_model.visual(x)  # [B, 512]
        
        # Apply adapter
        adapter_out = self.lora_up(self.lora_down(spatial_feat))
        spatial_feat_adapted = spatial_feat + adapter_out
        
        # Get frequency features
        freq_feat = self.freq_encoder(x, freq_band="full")  # [B, 256]
        
        # Fuse
        combined = torch.cat([spatial_feat_adapted, freq_feat], dim=1)
        return self.fusion(combined)
    
    def get_preprocess(self):
        return self.preprocess


def get_model(model_name, rank=8, device="cuda"):
    """Factory function to get model"""
    models = {
        "M1": M1_CLIPOnly,
        "M2": lambda: M2_CLIPAdapter(rank=rank, device=device),
        "M3": M3_CLIPDCTOnly,
        "M4": lambda: M4_FreqLoRACLIP(rank=rank, device=device)
    }
    
    if model_name == "M1":
        return M1_CLIPOnly(device=device)
    elif model_name == "M2":
        return M2_CLIPAdapter(rank=rank, device=device)
    elif model_name == "M3":
        return M3_CLIPDCTOnly(device=device)
    elif model_name == "M4":
        return M4_FreqLoRACLIP(rank=rank, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_trainable_params(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model):
    """Count all parameters"""
    return sum(p.numel() for p in model.parameters())

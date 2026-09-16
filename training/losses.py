"""Loss functions used by the few-shot training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        alpha: scalar class weight or a tensor of per-class weights.
        gamma: focusing parameter; gamma=0 reduces to weighted CE.
        reduction: ``mean``, ``sum`` or ``none``.
    """

    def __init__(self, alpha=1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device, logits.dtype)[targets]
            loss = loss * alpha
        else:
            loss = loss * float(self.alpha)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / n_classes)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / n_classes)
        loss = -(true_dist * log_probs).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

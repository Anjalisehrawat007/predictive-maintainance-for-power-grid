"""Loss functions for imbalanced binary classification."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        at = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (at * (1 - pt) ** self.gamma * bce).mean()


def build_loss(name: str, pos_weight: float = 1.0, gamma: float = 2.0, alpha: float = 0.75) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        pw = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(float(pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if name == "focal":
        return FocalLoss(gamma, alpha)
    raise ValueError(f"unknown loss {name}")

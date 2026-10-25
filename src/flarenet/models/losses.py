"""Loss functions: focal BCE for the rare M/X forecast, label-smoothed weighted CE for nowcast."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCE(nn.Module):
    """Binary focal loss on logits with optional positive-class weight."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        return ((1 - pt) ** self.gamma * bce).mean()


class LabelSmoothedCE(nn.Module):
    """Cross-entropy with label smoothing and per-class weights (normalised per batch)."""

    def __init__(self, n_classes: int, smoothing: float = 0.08, weight: torch.Tensor | None = None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            soft = torch.full_like(log_p, self.smoothing / (self.n_classes - 1))
            soft.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(soft * log_p).sum(-1)
        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w / w.mean()
        return loss.mean()


def class_weights(
    y: np.ndarray, n_classes: int, cap: float = 6.0, power: float = 0.5
) -> torch.Tensor:
    """(inverse-frequency ** power) class weights, capped; ``power=0.5`` is sqrt-inverse-frequency.
    Absent classes get count 1."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    ratio = counts.sum() / (n_classes * counts)
    w = np.clip(np.sqrt(ratio) if power == 0.5 else ratio**power, None, cap)
    return torch.tensor(w, dtype=torch.float32)

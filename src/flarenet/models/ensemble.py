"""Probability-averaging ensemble of FlareNet members, usable wherever a single model is."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """Averages member probabilities and returns them as logits, so the usual
    ``softmax(now_logits)`` / ``sigmoid(fc_logit)`` downstream code gives the averaged probabilities."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if not members:
            raise ValueError("EnsembleModel needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor):
        outs = [m(x) for m in self.members]
        now_p = torch.stack([F.softmax(o[0], dim=-1) for o in outs]).mean(0)
        fc_p = torch.stack([torch.sigmoid(o[1]) for o in outs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(now_p.clamp_min(1e-8)), torch.log(fc_p) - torch.log1p(-fc_p), outs[0][2]

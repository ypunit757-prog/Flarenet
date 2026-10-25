"""FlareNet: shared Conv1D + residual encoder feeding a nowcast head and a forecast head."""

from __future__ import annotations

import torch
import torch.nn as nn

from flarenet.core.constants import N_CHANNELS, N_CLASSES
from flarenet.core.registry import MODELS


class ResBlock(nn.Module):
    """Two Conv1d layers with BatchNorm, dropout, and an identity skip connection."""

    def __init__(self, channels: int, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class ChannelDropout(nn.Module):
    """Drop whole feature channels at train time (inverted scaling), no-op at eval."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep = torch.bernoulli(torch.full((x.size(0), x.size(1), 1), 1 - self.p, device=x.device))
        return x * keep / (1 - self.p)


class AttentionPool(nn.Module):
    """Additive attention over time; returns the pooled vector and the attention weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1), w


@MODELS.register("flarenet")
class FlareNet(nn.Module):
    """Dual-head network.

    Input  : (batch, 12 channels, window_min)
    Output : nowcast logits (batch, 5), forecast logit (batch,), attention weights (batch, T/2)
    """

    def __init__(
        self,
        cnn_channels: int = 32,
        nowcast_hidden: int = 32,
        forecast_hidden: int = 40,
        dropout: float = 0.35,
        in_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        nowcast_skip: bool = False,
        nowcast_pool: str = "last",
    ):
        super().__init__()
        cc, nh, fh, dp = cnn_channels, nowcast_hidden, forecast_hidden, dropout
        self.nh = nh
        self.nowcast_skip = nowcast_skip
        self.nowcast_pool = nowcast_pool
        # Shared encoder
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cc, 7, padding=3), nn.BatchNorm1d(cc), nn.ReLU()
        )
        self.res1 = ResBlock(cc, 5, dp * 0.4)
        self.res2 = ResBlock(cc, 3, dp * 0.4)
        self.chan_drop = ChannelDropout(dp * 0.3)
        self.pool = nn.MaxPool1d(2)
        self.res3 = ResBlock(cc, 3, dp * 0.4)
        self.drop = nn.Dropout(dp)
        # Nowcast head: 1-layer BiLSTM -> last step -> MLP
        self.now_lstm = nn.LSTM(cc, nh, 1, batch_first=True, bidirectional=True)
        self.now_norm = nn.LayerNorm(nh * 2)
        # The skip path appends the newest minute and the mean of the newest 5 minutes of the
        # (normalised) input channels to the LSTM summary.
        head_in = nh * 2 + (2 * in_channels if nowcast_skip else 0)
        self.now_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dp), nn.Linear(64, n_classes)
        )
        # Forecast head: 2-layer BiLSTM -> attention pooling -> MLP
        self.fc_lstm = nn.LSTM(cc, fh, 2, batch_first=True, bidirectional=True, dropout=dp)
        self.fc_norm = nn.LayerNorm(fh * 2)
        self.fc_attn = AttentionPool(fh * 2)
        self.fc_head = nn.Sequential(
            nn.Linear(fh * 2, 64), nn.ReLU(), nn.Dropout(dp), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        f = self.res2(self.res1(self.stem(x)))
        f = self.res3(self.pool(self.chan_drop(f)))
        f = self.drop(f).transpose(1, 2)  # (B, T/2, C)

        now_out = self.now_norm(self.now_lstm(f)[0])
        if self.nowcast_pool == "final_states":
            # forward direction at the last step + backward direction at the first step: the two
            # states that have each read the whole window (the backward state at the last step
            # has seen a single time step).
            h = torch.cat([now_out[:, -1, : self.nh], now_out[:, 0, self.nh :]], dim=-1)
        else:
            h = now_out[:, -1, :]
        if self.nowcast_skip:
            h = torch.cat([h, x[:, :, -1], x[:, :, -5:].mean(dim=-1)], dim=-1)
        now_logits = self.now_head(h)

        fc_out, _ = self.fc_lstm(f)
        pooled, attn = self.fc_attn(self.fc_norm(fc_out))
        fc_logit = self.fc_head(pooled).squeeze(-1)
        return now_logits, fc_logit, attn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Names of the shared-encoder sub-modules (frozen when fine-tuning with ``freeze_encoder``).
ENCODER_MODULES = ("stem", "res1", "res2", "chan_drop", "res3")


def build_model(mcfg, in_channels: int = N_CHANNELS) -> nn.Module:
    """Instantiate the architecture named by ``cfg.model.name`` from a ``ModelConfig``."""
    cls = MODELS.get(mcfg.name)
    return cls(
        mcfg.cnn_channels, mcfg.nowcast_hidden, mcfg.forecast_hidden, mcfg.dropout,
        in_channels=in_channels,
        nowcast_skip=getattr(mcfg, "nowcast_skip", False),
        nowcast_pool=getattr(mcfg, "nowcast_pool", "last"),
    )  # fmt: skip

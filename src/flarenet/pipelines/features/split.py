"""Leakage-safe split of a ``WindowSet`` into train / val / test by contiguous multi-day blocks."""

from __future__ import annotations

import numpy as np

from flarenet.core.schema import WindowSet


def block_split(
    ws: WindowSet,
    block_days: int = 3,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, WindowSet]:
    """Split by contiguous multi-day blocks so neighbouring (overlapping) windows never straddle
    train/val/test. Blocks are shuffled, then assigned 70/15/15 by default. Output windows are
    sorted by time within each split."""
    rng = np.random.default_rng(seed)
    order = np.argsort(ws.t_end, kind="stable")
    ws = ws.subset(order)
    t_end = ws.t_end
    blocks = np.floor((t_end - t_end[0]) / (block_days * 24 * 60)).astype(int)
    unique = np.unique(blocks)
    rng.shuffle(unique)
    n_blocks = len(unique)
    n_train = int(n_blocks * train_frac)
    n_val = int(n_blocks * val_frac)
    parts = {
        "train": unique[:n_train],
        "val": unique[n_train : n_train + n_val],
        "test": unique[n_train + n_val :],
    }
    return {k: ws.subset(np.isin(blocks, b)) for k, b in parts.items()}

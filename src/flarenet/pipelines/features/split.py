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


def chrono_split(
    ws: WindowSet, train_frac: float = 0.70, val_frac: float = 0.15, gap_min: float = 0.0
) -> dict[str, WindowSet]:
    """Split by time: the earliest ``train_frac`` of the series is train, the next ``val_frac`` is validation,
    the rest is test. Fractions are shares of the time span, not of the window count.

    ``gap_min`` (window length - 1 + forecast horizon) is cut off the END of the train and validation parts, so no
    training window's input or forecast label reaches into the following part. Test windows are never touched.
    Unlike :func:`block_split`, validation and test are later in time than everything the weights were fitted on,
    which is how the model is used in practice (and how a new satellite or year is scored)."""
    order = np.argsort(ws.t_end, kind="stable")
    ws = ws.subset(order)
    t = ws.t_end
    if len(t) == 0:
        return {"train": ws, "val": ws, "test": ws}
    t0, t1 = float(t[0]), float(t[-1])
    cut_train = t0 + train_frac * (t1 - t0)
    cut_val = t0 + (train_frac + val_frac) * (t1 - t0)
    train = t < cut_train - gap_min
    val = (t >= cut_train) & (t < cut_val - gap_min)
    test = t >= cut_val
    return {"train": ws.subset(train), "val": ws.subset(val), "test": ws.subset(test)}

"""Explain one forecast: how far each group of inputs moves the probability away from a typical window."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from flarenet.explain.shapley import masks, shapley

EPS = 1e-6


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def tree_effects(
    predict_proba: Callable[[pd.DataFrame], np.ndarray], row: pd.Series, reference: pd.Series,
    groups: dict[str, tuple[str, list[str]]],
) -> dict:  # fmt: skip
    """Shapley contribution (log-odds) of every feature group of a tree model to ``row``, against ``reference``.

    ``predict_proba`` takes a table with the columns of ``row`` and returns probabilities. Returns the contributions, the
    probability for ``row`` and for the all-reference input; the contributions sum to ``logit(p_row) - logit(p_ref)``."""
    gids = list(groups)
    cols = list(row.index)
    m = masks(len(gids))
    ref = reference.reindex(cols)
    batch = np.tile(ref.to_numpy(dtype=float), (len(m), 1))
    for k, gid in enumerate(gids):
        for c in groups[gid][1]:
            j = cols.index(c)
            batch[m[:, k], j] = row.iloc[j]
    p = np.asarray(predict_proba(pd.DataFrame(batch, columns=cols)), dtype=float)
    phi = shapley(logit(p), len(gids))
    return {
        "effects": dict(zip(gids, map(float, phi), strict=True)),
        "p_row": float(p[-1]),
        "p_ref": float(p[0]),
    }


def network_effects(
    predict_proba: Callable[[np.ndarray], np.ndarray], window: np.ndarray, reference: np.ndarray,
    groups: dict[str, tuple[str, list[int]]],
) -> dict:  # fmt: skip
    """The same for the sequence network: ``window`` and ``reference`` are (channels, minutes) arrays in raw units and a
    group is a set of channel indices; absent channels are replaced by the reference channels (the training average)."""
    gids = list(groups)
    m = masks(len(gids))
    batch = np.broadcast_to(reference, (len(m), *reference.shape)).copy()
    for k, gid in enumerate(gids):
        for c in groups[gid][1]:
            batch[m[:, k], c] = window[c]
    p = np.asarray(predict_proba(batch.astype(np.float32)), dtype=float)
    phi = shapley(logit(p), len(gids))
    return {
        "effects": dict(zip(gids, map(float, phi), strict=True)),
        "p_row": float(p[-1]),
        "p_ref": float(p[0]),
    }

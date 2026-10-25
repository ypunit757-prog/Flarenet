"""Per-channel normalisation fitted on the training windows only."""

from __future__ import annotations

import numpy as np


def fit_normaliser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over (samples, time), shaped (1, C, 1) for broadcasting."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sg = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sg.astype(np.float32)


def normalise(X: np.ndarray, mu: np.ndarray, sg: np.ndarray) -> np.ndarray:
    return ((X - mu) / sg).astype(np.float32)

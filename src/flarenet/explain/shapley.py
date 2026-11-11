"""Exact Shapley values over a small number of input groups.

A Shapley value is the average change in the model's output caused by adding one group of inputs to every possible subset
of the other groups. Here "absent" means "replaced by a reference value" (a typical window), and the output is measured in
log-odds, so the values add up exactly: ``sum(phi) == value(all groups present) - value(none present)``. With at most about
a dozen groups every subset can be evaluated (2^n model calls, batched), so the numbers are exact, not sampled.

This is the same quantity the SHAP library estimates with an interventional baseline, computed over groups of features.
"""

from __future__ import annotations

from math import factorial

import numpy as np

MAX_PLAYERS = 14


def masks(n: int) -> np.ndarray:
    """All 2**n subsets as a boolean matrix; row ``m`` holds the bits of the integer ``m`` (player 0 = lowest bit)."""
    if not 1 <= n <= MAX_PLAYERS:
        raise ValueError(f"exact Shapley values need 1 to {MAX_PLAYERS} groups, got {n}")
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)


def shapley(values: np.ndarray, n: int) -> np.ndarray:
    """``values[m]`` is the model output (log-odds) when exactly the groups in the bit set ``m`` take their real values.
    Returns one contribution per group."""
    values = np.asarray(values, dtype=float)
    if values.shape != (2**n,):
        raise ValueError(f"expected {2**n} coalition values, got {values.shape}")
    idx = np.arange(2**n)
    size = masks(n).sum(axis=1)
    weight = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        without = idx[((idx >> i) & 1) == 0]
        phi[i] = np.sum(weight[size[without]] * (values[without | (1 << i)] - values[without]))
    return phi

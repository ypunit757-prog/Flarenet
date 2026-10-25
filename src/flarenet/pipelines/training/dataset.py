"""Torch dataset over normalised windows with training-time augmentation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from flarenet.pipelines.features.normalise import fit_normaliser

# Channels whose value moves with the soft / hard detector's absolute level (log10 flux units).
_SOFT_LEVEL = ("log_sxr", "max_sxr_2h", "max_sxr_6h", "ewma_sxr_4h", "log_fluence_1h")


def instrument_augment(
    x: np.ndarray,
    channels: list[str],
    rng: np.random.Generator,
    level_jitter: float = 0.0,
    hxr_dropout: float = 0.0,
) -> np.ndarray:
    """Simulate a different detector on one raw (un-normalised) window ``x`` of shape (C, T).

    * ``level_jitter``: a random gain / background change, i.e. a constant shift (log10 units)
      of the soft level channels and, independently, of the hard channel (hardness follows).
    * ``hxr_dropout``: with this probability the hard channel is unavailable and is replaced by a
      copy of the soft one (hardness 0), as when a source has no usable hard X-ray band.
    """
    idx = {n: i for i, n in enumerate(channels)}
    if level_jitter > 0:
        s, h = rng.normal(0.0, level_jitter, size=2)
        for n in _SOFT_LEVEL:
            if n in idx:
                x[idx[n]] += s
        if "log_hxr" in idx:
            x[idx["log_hxr"]] += h
        if "hardness" in idx:
            x[idx["hardness"]] += h - s
    if hxr_dropout > 0 and rng.random() < hxr_dropout:
        for hard, soft in (("log_hxr", "log_sxr"), ("d_log_hxr", "d_log_sxr"),
                           ("d2_log_hxr", "d2_log_sxr")):  # fmt: skip
            if hard in idx and soft in idx:
                x[idx[hard]] = x[idx[soft]]
        if "hardness" in idx:
            x[idx["hardness"]] = 0.0
    return x


class WindowDataset(Dataset):
    """Windows with normalisation and augmentation.

    Augmentation (``augment=True``): Gaussian noise (sigma 0.03, plus an extra sigma drawn from
    ``[0, noise_max]`` per window), a short zeroed span, and the optional instrument
    simulation of :func:`instrument_augment` applied in raw units before normalising.
    """

    def __init__(
        self, X, y_now, y_fc, mu=None, sg=None, augment: bool = False,
        channels: list[str] | None = None, level_jitter: float = 0.0,
        hxr_dropout: float = 0.0, noise_max: float = 0.0,
    ):  # fmt: skip
        if mu is None or sg is None:
            mu, sg = fit_normaliser(X)
        self.raw = X
        self.mu, self.sg = mu, sg
        self.y_now = y_now
        self.y_fc = y_fc
        self.augment = augment
        self.channels = channels
        self.level_jitter, self.hxr_dropout, self.noise_max = level_jitter, hxr_dropout, noise_max
        self._inst = (level_jitter > 0 or hxr_dropout > 0) and channels is not None
        self._rng = np.random.default_rng()
        # plain evaluation and the standard augmentation work on a pre-normalised copy
        shape = (1, -1, 1)
        self._mu = np.asarray(mu, dtype=np.float32).reshape(shape)[0]
        self._sg = np.asarray(sg, dtype=np.float32).reshape(shape)[0]
        self.X = None if self._inst and augment else ((X - self._mu) / self._sg).astype(np.float32)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, i: int):
        if self.X is not None:
            x = self.X[i].copy()
        else:  # instrument simulation in raw units, then normalise
            x = instrument_augment(
                self.raw[i].astype(np.float32).copy(), self.channels, self._rng,
                self.level_jitter, self.hxr_dropout,
            )  # fmt: skip
            x = ((x - self._mu) / self._sg).astype(np.float32)
        if self.augment:
            sigma = 0.03
            if self.noise_max > 0:
                sigma += float(np.random.rand()) * self.noise_max
            x += np.random.randn(*x.shape).astype(np.float32) * sigma
            if np.random.rand() < 0.3:
                start = np.random.randint(0, x.shape[1] - 5)
                length = np.random.randint(2, 8)
                x[:, start : start + length] = 0.0
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_now[i], dtype=torch.long),
            torch.tensor(self.y_fc[i], dtype=torch.float32),
        )

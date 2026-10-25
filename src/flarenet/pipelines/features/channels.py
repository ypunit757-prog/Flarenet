"""The 12 engineered per-minute feature channels (pure functions on a labelled lightcurve)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.constants import N_CHANNELS

CHANNEL_NAMES = [
    "log_sxr",
    "log_hxr",
    "d_log_sxr",
    "d_log_hxr",
    "d2_log_sxr",
    "d2_log_hxr",
    "hardness",
    "max_sxr_2h",
    "max_sxr_6h",
    "ewma_sxr_4h",
    "log_fluence_1h",
    "recent_flare_rate_6h",
]


def fill_gaps(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs, then fill any leading NaNs with the series mean."""
    a = a.copy()
    missing = np.isnan(a)
    if not missing.any():
        return a
    idx = np.where(~missing, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a[missing] = a[idx[missing]]
    missing = np.isnan(a)
    if missing.any():
        a[missing] = np.nanmean(a)
    return a


def log10_clip(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 1e-11, None))


def rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=1).max().to_numpy()


def forward_any(flags: np.ndarray, horizon: int) -> np.ndarray:
    """out[i] is True if any flags[i+1 : i+1+horizon] is True (strictly future)."""
    n = len(flags)
    cum = np.concatenate([[0], np.cumsum(flags.astype(np.int32))])
    ii = np.arange(n)
    return (cum[np.minimum(n, ii + 1 + horizon)] - cum[np.minimum(n, ii + 1)]) > 0


def first_minute_flags(flare_id: np.ndarray, flare_class: np.ndarray, min_class: int) -> np.ndarray:
    """Flag the FIRST minute of each event where ``flare_class >= min_class``.

    Using a separate ``seen`` set that is only updated once the threshold class is actually
    observed marks the true escalation onset, not the start of the event group (which may begin
    at a lower class). This is the v5.1 fix for M/X onset detection.
    """
    flags = np.zeros(len(flare_id), dtype=bool)
    seen: set[int] = set()
    for i, fid in enumerate(flare_id):
        if fid != -1 and fid not in seen and flare_class[i] >= min_class:
            flags[i] = True
            seen.add(int(fid))
    return flags


def trailing_baseline(a: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Causal quiet-level estimate: the ``quantile`` of the last ``window`` samples (>= 60 needed).

    Uses only past samples, so the same value is produced when predicting live.
    """
    return (
        pd.Series(a)
        .rolling(window, min_periods=min(60, window))
        .quantile(quantile)
        .bfill()
        .to_numpy()
    )


def compute_channels(
    df: pd.DataFrame,
    relative: bool = False,
    baseline_min: int = 1440,
    baseline_quantile: float = 0.1,
) -> list[np.ndarray]:
    """Return the 12 per-minute feature channels (see CHANNEL_NAMES).

    ``relative=True`` expresses every level channel (log flux, rolling maxima, EWMA, fluence,
    hardness) as decades above the instrument's own trailing quiet baseline, so a different gain,
    background or unit does not shift the inputs. Derivative channels are already gain-free.

    Caveat: the last channel (recent flare rate) is computed from the flare labels in ``df``.
    On real data those labels come from a threshold detector on the SoLEXS flux, so it can be
    recomputed at inference time, but it is NOT a purely raw-flux feature.
    """
    sxr = fill_gaps(df["sxr"].to_numpy(dtype=float))
    hxr = fill_gaps(df["hxr"].to_numpy(dtype=float))
    flare_class = df["flare_class"].to_numpy()
    flare_id = df["flare_id"].to_numpy()

    ls, lh = log10_clip(sxr), log10_clip(hxr)
    ds = np.diff(ls, prepend=ls[0])
    dh = np.diff(lh, prepend=lh[0])
    d2s = np.diff(ds, prepend=ds[0])
    d2h = np.diff(dh, prepend=dh[0])
    hardness = lh - ls
    r2h = rolling_max(ls, 120)
    r6h = rolling_max(ls, 360)
    ewma = pd.Series(ls).ewm(alpha=1 - np.exp(-np.log(2) / 240), adjust=False).mean().to_numpy()
    fluence = log10_clip(
        pd.Series(np.clip(sxr, 1e-10, None)).rolling(60, min_periods=1).sum().to_numpy()
    )

    # Event onset: first minute of each labelled event (class >= B).
    onset = first_minute_flags(flare_id, flare_class, 1).astype(float)
    rate = np.log1p(pd.Series(onset).rolling(360, min_periods=1).sum().to_numpy())

    if relative:
        bs = trailing_baseline(ls, baseline_min, baseline_quantile)
        bh = trailing_baseline(lh, baseline_min, baseline_quantile)
        ls, lh = ls - bs, lh - bh
        hardness = lh - ls
        r2h, r6h, ewma = r2h - bs, r6h - bs, ewma - bs
        fluence = fluence - np.log10(60.0) - bs
    channels = [ls, lh, ds, dh, d2s, d2h, hardness, r2h, r6h, ewma, fluence, rate]
    assert len(channels) == N_CHANNELS
    return channels


def select_channels(
    channels: list[np.ndarray], names: list[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    """Keep only ``names`` (in that order); ``None`` keeps all 12."""
    if names is None:
        return channels, list(CHANNEL_NAMES)
    unknown = [n for n in names if n not in CHANNEL_NAMES]
    if unknown:
        raise KeyError(f"Unknown feature channels {unknown}. Available: {CHANNEL_NAMES}")
    if not names:
        raise ValueError("features.channels must not be empty")
    return [channels[CHANNEL_NAMES.index(n)] for n in names], list(names)

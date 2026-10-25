"""Sliding-window construction: lightcurve (1-minute, labelled) -> ``WindowSet``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.constants import CLASS_TO_IDX
from flarenet.core.schema import WindowSet
from flarenet.pipelines.features.channels import (
    compute_channels,
    first_minute_flags,
    forward_any,
    select_channels,
)


def build_windows(
    df: pd.DataFrame,
    window_min: int = 90,
    stride_min: int = 60,
    horizon_min: int = 360,
    for_inference: bool = False,
    channel_names: list[str] | None = None,
    max_missing_frac: float = 1.0,
    relative: bool = False,
    baseline_min: int = 1440,
    baseline_quantile: float = 0.1,
    starts: np.ndarray | None = None,
) -> WindowSet:
    """Slice the lightcurve into overlapping windows.

    X is (n, n_channels, window_min) float32 features, ``y_now`` the nowcast class at the last
    minute of each window, ``y_fc`` is 1 if an M/X escalation begins within ``horizon_min``
    after the window, ``t_end`` the minute index of the window's last sample and ``missing_frac``
    the share of missing soft X-ray samples inside the window (before gap filling).

    ``for_inference=True`` keeps windows that end inside the final ``horizon_min`` minutes (their
    forecast labels are then only partially observed). Use it when predicting, not for scoring.
    Windows with more than ``max_missing_frac`` missing samples are dropped.
    ``starts`` (minute indices) replaces the regular stride grid, e.g. to issue forecasts at
    exactly 00:00 UTC; windows whose label horizon runs past the data are then dropped.
    """
    channels, names = select_channels(
        compute_channels(df, relative, baseline_min, baseline_quantile), channel_names
    )
    flare_class = df["flare_class"].to_numpy()
    flare_id = df["flare_id"].to_numpy()
    missing = np.isnan(df["sxr"].to_numpy(dtype=float)).astype(np.float32)

    mx_onset = first_minute_flags(flare_id, flare_class, CLASS_TO_IDX["M"])
    future_mx = forward_any(mx_onset, horizon_min)

    n = len(df)
    last_start = n - window_min + 1 if for_inference else n - window_min - horizon_min
    if starts is None:
        starts = np.arange(0, max(last_start, 0), stride_min)
    else:
        starts = np.asarray(starts, dtype=np.int64)
        starts = starts[(starts >= 0) & (starts < max(last_start, 0))]
    if len(starts) == 0:
        raise ValueError(
            f"Series of {n} minutes is too short for window={window_min}"
            + ("" if for_inference else f" + horizon={horizon_min}")
        )
    ends = starts + window_min
    cum_missing = np.concatenate([[0.0], np.cumsum(missing)])
    miss_frac = ((cum_missing[ends] - cum_missing[starts]) / window_min).astype(np.float32)
    keep = miss_frac <= max_missing_frac
    starts, ends, miss_frac = starts[keep], ends[keep], miss_frac[keep]
    if len(starts) == 0:
        raise ValueError(f"All windows exceed max_missing_frac={max_missing_frac}")

    X = np.stack(
        [np.stack([c[s:e] for c in channels], axis=0) for s, e in zip(starts, ends, strict=True)]
    ).astype(np.float32)
    return WindowSet(
        X=X,
        y_now=flare_class[ends - 1].astype(np.int64),
        y_fc=future_mx[ends - 1].astype(np.float32),
        t_end=df["time_min"].to_numpy()[ends - 1].astype(np.float64),
        missing_frac=miss_frac,
        channels=names,
    )

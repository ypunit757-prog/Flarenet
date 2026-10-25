"""RawSeries (native cadence, per-channel time axes) -> regular 1-minute grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.schema import RawSeries, SchemaError

AL_FILL = {"flare_id": -1}  # fill value for aligned arrays on empty minutes (default 0)


def bin_mean(t_unix: np.ndarray, values: np.ndarray, t0: float, n: int) -> np.ndarray:
    """Mean of ``values`` inside each 1-minute bin starting at ``t0``; NaN for empty bins."""
    idx = np.floor((t_unix - t0) / 60.0).astype(np.int64)
    ok = (idx >= 0) & (idx < n) & np.isfinite(values)
    out = np.full(n, np.nan)
    if not ok.any():
        return out
    sums = np.bincount(idx[ok], weights=values[ok], minlength=n)
    cnt = np.bincount(idx[ok], minlength=n)
    nz = cnt > 0
    out[nz] = sums[nz] / cnt[nz]
    return out


def resample_to_minutes(raw: RawSeries) -> tuple[pd.DataFrame, float, dict[str, np.ndarray], dict]:
    """Return (df[time_min, sxr_raw, hxr_raw], t0_unix, aligned arrays on the grid, info).

    The grid starts at the first soft-channel sample's minute. When the source has no hard
    channel, ``hxr_raw`` is a copy of the soft channel (``info['hxr_substituted']``) so that the
    feature set keeps its shape; the hardness ratio is then uninformative.
    """
    sx = raw.channels["sxr"]
    t = sx.t_unix
    finite_t = np.isfinite(t)
    if finite_t.sum() < 2:
        raise SchemaError(f"[{raw.source}] fewer than 2 soft-channel samples")
    t0 = float(np.floor(t[finite_t][0] / 60.0) * 60.0)
    t_last = float(np.floor(t[finite_t][-1] / 60.0) * 60.0)
    n = int(round((t_last - t0) / 60.0)) + 1
    if n < 2:
        raise SchemaError(f"[{raw.source}] series spans less than 2 minutes")

    sxr = bin_mean(t, sx.values, t0, n)
    info: dict = {"hxr_substituted": False}
    if "hxr" in raw.channels:
        h = raw.channels["hxr"]
        hxr = bin_mean(h.t_unix, h.values, t0, n)
    else:
        hxr, info["hxr_substituted"] = sxr.copy(), True

    aligned: dict[str, np.ndarray] = {}
    if raw.aligned:
        idx = np.floor((t - t0) / 60.0).astype(np.int64)
        ok = finite_t & (idx >= 0) & (idx < n)
        for k, a in raw.aligned.items():
            a = np.asarray(a)
            out = np.full(n, AL_FILL.get(k, 0), dtype=a.dtype)
            out[idx[ok]] = a[ok]  # last sample in a bin wins
            aligned[k] = out

    df = pd.DataFrame({"time_min": np.arange(n, dtype=float), "sxr_raw": sxr, "hxr_raw": hxr})
    info["coverage_sxr"] = float(np.isfinite(sxr).mean())
    return df, t0, aligned, info

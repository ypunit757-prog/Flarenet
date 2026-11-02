"""Features at the end of each network window, built only from data that existed at that minute.

Reuses ``flarenet.daily.features.day_features`` (flux level over 3 h .. 7 days, flare counts and the time since the
last C/M/X flare), adds short-window terms the daily model does not need (last 30 min .. 2 h, slope, flux relative to
the quiet level of the last 3 days) and rotation recurrence at the hour scale: flares that started in the same 6-hour
slot one solar rotation (27 days) earlier.

A flare enters the history only ``DETECT_DELAY_S`` after its peak, because a peak is only known to be the peak once
the flux has fallen; without that delay a flare that peaked a minute ago would count as already known.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from flarenet.core.schema import LightcurveBundle, WindowSet
from flarenet.daily.features import ROT_DAYS, SINCE_CAP_H, day_features, events_table

DETECT_DELAY_S = 600.0
_UNUSED_DAY = date(1970, 1, 1)
_DAILY_ROT = (
    "rot_nC_27",
    "rot_nM_27",
    "rot_nM_27_wide",
    "rot_peak_27_wide",
)  # day-scale, not meaningful here


def _mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def _max(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.max()) if len(a) else np.nan


def prepare(bundle: LightcurveBundle) -> tuple[np.ndarray, pd.DataFrame]:
    """Log10 flux on the 1-minute grid and the event table, with each peak shifted by the detection delay."""
    df = bundle.df
    flux = (df["sxr_raw"] if "sxr_raw" in df.columns else df["sxr"]).to_numpy(float)
    lf = np.log10(np.clip(np.where(np.isfinite(flux) & (flux > 0), flux, np.nan), 1e-11, None))
    ev = events_table(bundle)
    ev = ev.assign(peak=ev["peak"] + DETECT_DELAY_S).sort_values("peak", ignore_index=True)
    return lf, ev


def window_features(
    i: int, lf: np.ndarray, ev: pd.DataFrame, t0: float, horizon_min: int
) -> dict[str, float]:
    """Features for the window whose last minute has index ``i`` on the grid starting at ``t0`` (unix seconds)."""
    t_issue = t0 + i * 60.0
    peak = ev["peak"].to_numpy()
    known = np.searchsorted(
        peak, t_issue, side="right"
    )  # events whose (delayed) peak is already known
    # day_features only looks at the last SINCE_CAP_H hours of known events: hand it that slice, not the whole table
    # (a pandas filter of the full table at every window dominates the run time on years of data)
    lo_cap = np.searchsorted(peak, t_issue - SINCE_CAP_H * 3600.0, side="right")
    f = day_features(
        t_issue, t_issue, lf, t0, ev.iloc[lo_cap:known], None, _UNUSED_DAY, rot_ok=False
    )
    for k in _DAILY_ROT:
        f.pop(k, None)
    cur = f["f_cur"]
    for m in (30, 60, 120):
        f[f"f_max_{m}m"] = _max(lf[max(0, i - m + 1) : i + 1])
    last10 = lf[max(0, i - 9) : i + 1]
    f["f_slope_30m"] = _mean(last10) - _mean(lf[max(0, i - 39) : max(0, i - 29)])
    f["f_slope_1h"] = _mean(last10) - _mean(lf[max(0, i - 69) : max(0, i - 59)])
    f["f_rel_q10_72h"] = cur - f["f_q10_72h"]
    f["f_rel_mean_24h"] = cur - f["f_mean_24h"]
    w6 = lf[max(0, i - 359) : i + 1]
    ok = np.isfinite(w6)
    f["f_since_max_6h"] = (
        float(len(w6) - 1 - np.nanargmax(np.where(ok, w6, -np.inf))) if ok.any() else np.nan
    )

    cls = ev["cls"].to_numpy()
    for h in (3, 6):
        lo = np.searchsorted(peak, t_issue - h * 3600.0, side="right")
        c = cls[lo:known]
        f[f"h_nC_{h}"] = float((c >= 2).sum())
        f[f"h_nM_{h}"] = float((c >= 3).sum())

    d0 = t_issue - ROT_DAYS * 86400.0
    if d0 - 12 * 3600.0 >= t0:
        on = ev["onset"].to_numpy()
        slot = (on > d0) & (on <= d0 + horizon_min * 60.0)
        wide = (on > d0 - 12 * 3600.0) & (on <= d0 + horizon_min * 60.0 + 12 * 3600.0)
        f["rot_nM_slot"] = float(((cls >= 3) & slot).sum())
        f["rot_nC_slot"] = float(((cls >= 2) & slot).sum())
        f["rot_nM_wide"] = float(((cls >= 3) & wide).sum())
    else:
        f["rot_nM_slot"] = f["rot_nC_slot"] = f["rot_nM_wide"] = np.nan
    return f


def set_features(
    ws: WindowSet, bundles: dict[str, LightcurveBundle], horizon_min: int, stride_min: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature table for the windows of ``ws`` and the row positions of ``ws`` it covers.

    ``stride_min`` > 0 keeps about one window per ``stride_min`` minutes of each source (neighbouring windows differ
    by one minute and would only slow the tree model down); 0 keeps every window. The returned index (``t_issue`` as
    UTC time) aligns row by row with ``ws.subset(rows)``."""
    prepared = {}
    frames, rows = [], []
    for k, name in enumerate(ws.source_names):
        sel = np.flatnonzero(ws.source_idx == k)
        if len(sel) == 0:
            continue
        if stride_min > 0:
            _, first = np.unique(np.floor(ws.t_end[sel] / stride_min), return_index=True)
            sel = sel[np.sort(first)]
        bundle = bundles[name]
        if name not in prepared:
            prepared[name] = prepare(bundle)
        lf, ev = prepared[name]
        feats = [
            window_features(int(round(t)), lf, ev, bundle.t0_unix, horizon_min)
            for t in ws.t_end[sel]
        ]
        tab = pd.DataFrame(feats)
        tab.index = pd.to_datetime(bundle.t0_unix + ws.t_end[sel] * 60.0, unit="s", utc=True)
        frames.append(tab)
        rows.append(sel)
    if not frames:
        return pd.DataFrame(), np.zeros(0, dtype=int)
    table, rows = pd.concat(frames), np.concatenate(rows)
    return table, rows

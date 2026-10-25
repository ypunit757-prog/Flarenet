"""Flux scaling policies: ``native`` keeps physical units, ``background_normalised`` maps counts
onto the synthetic flux scale (quiet-Sun level), as used for Aditya-L1 count rates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.registry import Registry

SCALERS: Registry = Registry("scaling policy")


def _positive(v: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


@SCALERS.register("native")
def scale_native(df: pd.DataFrame, params: dict | None = None, t0: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out["sxr_raw"], out["hxr_raw"] = (
        _positive(df.sxr_raw.to_numpy()),
        _positive(df.hxr_raw.to_numpy()),
    )
    out["sxr"], out["hxr"] = out["sxr_raw"], out["hxr_raw"]
    return out


@SCALERS.register("background_normalised")
def scale_background_normalised(
    df: pd.DataFrame, params: dict | None = None, t0: float = 0.0
) -> pd.DataFrame:
    """Exactly the v5.1 real-data recipe: divide by a quiet-Sun percentile, rescale to 3e-9 / 5e-10."""
    out = df.copy()
    s1m, h1m = df.sxr_raw.to_numpy(), df.hxr_raw.to_numpy()
    bg_s = np.nanpercentile(s1m, 10)
    bg_h = np.nanpercentile(h1m, 25)
    out["sxr"] = np.clip(s1m, 0.1, None) / max(bg_s, 1) * 3e-9
    out["hxr"] = np.clip(h1m, 0.1, None) / max(bg_h, 1) * 5e-10
    return out


def _fit_loglog(raw: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, int]:
    """Least squares ``log10(ref) = a + b*log10(raw)`` on finite positive pairs -> (a, b, R2, n)."""
    ok = np.isfinite(raw) & np.isfinite(ref) & (raw > 0) & (ref > 0)
    if ok.sum() < 200:
        raise ValueError(
            f"cross-calibration needs >= 200 overlapping minutes, found {int(ok.sum())}"
        )
    x, y = np.log10(raw[ok]), np.log10(ref[ok])
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1.0 - float(resid.var() / max(y.var(), 1e-12))
    return float(a), float(b), r2, int(ok.sum())


def _reference_on_grid(path: str, n: int, t0: float) -> np.ndarray:
    """Read ``time`` (ISO text or unix seconds) and ``flux`` columns of a CSV onto the 1-min grid."""
    ref = pd.read_csv(path)
    cols = {c.lower(): c for c in ref.columns}
    if "time" not in cols or "flux" not in cols:
        raise ValueError(
            f"{path}: reference CSV needs 'time' and 'flux' columns, has {list(ref.columns)}"
        )
    tc, fc = ref[cols["time"]], ref[cols["flux"]].to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(tc):
        unix = tc.to_numpy(dtype=float)
    else:
        unix = pd.to_datetime(tc, utc=True).astype("int64").to_numpy() / 1e9
    idx = np.floor((unix - t0) / 60.0).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    sums = np.bincount(idx[ok], weights=fc[ok], minlength=n)[:n]
    cnt = np.bincount(idx[ok], minlength=n)[:n]
    return np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)


@SCALERS.register("goes_calibrated")
def scale_goes_calibrated(
    df: pd.DataFrame, params: dict | None = None, t0: float = 0.0
) -> pd.DataFrame:
    """Map an instrument's soft channel onto GOES 1-8 A flux (W/m2) so that GOES class thresholds
    (C = 1e-6, M = 1e-5, X = 1e-4) can be applied to it.

    ``params`` (one of):
      * ``reference_csv`` (+ optional ``fit_range`` [start, end] unix seconds): fit
        ``log10 GOES = a + b log10(raw)`` on minutes where both exist, and apply it. The fit
        (a, b, R2, n) is returned in ``df.attrs['calibration']`` and ends up in the bundle meta.
      * ``gain`` / ``power`` / ``background``: ``flux = gain * max(raw - background, tiny)**power``.
      * ``quiet_flux`` (default 3e-9): divide by the instrument's 10th-percentile level and
        multiply by this quiet-Sun GOES level (a one-number anchor; class thresholds are then
        only approximate, since gain and spectral response differ).
    The hard channel is only rescaled to a fixed quiet level, as it has no GOES counterpart.
    """
    p = params or {}
    out = df.copy()
    s = df.sxr_raw.to_numpy(dtype=float)
    cal: dict = {}
    if "reference_csv" in p:
        ref = _reference_on_grid(p["reference_csv"], len(s), t0)
        mask = np.ones(len(s), bool)
        if "fit_range" in p:
            lo, hi = p["fit_range"]
            tt = t0 + df.time_min.to_numpy() * 60.0
            mask = (tt >= lo) & (tt <= hi)
        a, b, r2, n = _fit_loglog(np.where(mask, s, np.nan), ref)
        out["sxr"] = 10 ** (a + b * np.log10(np.clip(s, 1e-12, None)))
        cal = {"mode": "fit", "a": a, "b": b, "r2": r2, "n": n}
    elif "gain" in p or "power" in p or "background" in p:
        g, pw, bg = (
            float(p.get("gain", 1.0)),
            float(p.get("power", 1.0)),
            float(p.get("background", 0.0)),
        )
        out["sxr"] = g * np.clip(s - bg, 1e-12, None) ** pw
        cal = {"mode": "gain", "gain": g, "power": pw, "background": bg}
    else:
        q = float(p.get("quiet_flux", 3e-9))
        out["sxr"] = np.clip(s, 1e-12, None) / max(float(np.nanpercentile(s, 10)), 1e-12) * q
        cal = {"mode": "quiet_anchor", "quiet_flux": q}
    h = df.hxr_raw.to_numpy(dtype=float)
    out["hxr"] = np.clip(h, 1e-12, None) / max(float(np.nanpercentile(h, 25)), 1e-12) * 5e-10
    out["sxr_raw"] = out["sxr"]
    out.attrs["calibration"] = cal
    return out


def apply_scaling(
    name: str, df: pd.DataFrame, params: dict | None = None, t0: float = 0.0
) -> pd.DataFrame:
    return SCALERS.get(name)(df, params or {}, t0)

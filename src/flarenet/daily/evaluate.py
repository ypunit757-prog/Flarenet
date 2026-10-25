"""Scoring of daily probability forecasts: skill scores, paired block-bootstrap differences, reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc

NAN = float("nan")


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((y - p) ** 2)) if len(y) else NAN


def brier_skill(y, p) -> float:
    """1 - Brier / Brier(constant forecast of the observed rate). 0 = no better than climatology."""
    y = np.asarray(y, float)
    base = y.mean() if len(y) else NAN
    return 1.0 - brier(y, p) / (base * (1 - base)) if 0 < base < 1 else NAN


def scores(y, p, thr: float | np.ndarray = 0.5) -> dict:
    """All headline scores. ``thr`` may be one number or one per day (a threshold chosen per fold)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    thr_arr = np.broadcast_to(np.asarray(thr, float), p.shape)
    pred = (p >= thr_arr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else NAN
    pofd = fp / (fp + tn) if fp + tn else NAN
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "n_days": int(len(y)), "event_days": int(y.sum()), "TSS": pod - pofd if np.isfinite(pod) and np.isfinite(pofd) else NAN,
        "HSS": 2 * (tp * tn - fp * fn) / den if den else NAN, "POD": pod,
        "FAR": fp / (tp + fp) if tp + fp else NAN, "AUC": safe_auc(y, p), "Brier": brier(y, p),
        "BSS": brier_skill(y, p), "alert_rate": float(pred.mean()) if len(y) else NAN,
    }  # fmt: skip


def oracle_tss(y, p) -> float:
    return best_threshold(np.asarray(y, int), np.asarray(p, float))[1]


def paired_block_bootstrap(
    m: int, a, b, block: int = 14, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95 % interval of ``a(idx) - b(idx)`` over resamples of ``m`` days drawn in blocks of
    ``block`` consecutive days (flare activity is autocorrelated, single days are not independent).
    ``a`` and ``b`` are callables taking an index array and returning a score."""
    rng = np.random.default_rng(seed)
    nb = max(1, int(np.ceil(m / block)))
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, max(m - block + 1, 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, m)) for s in starts])[:m]
        va, vb = a(idx), b(idx)
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    if not diffs:
        return NAN, NAN, NAN
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def reliability(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.any():
            out.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                        "mean_forecast": float(p[m].mean()), "observed_rate": float(y[m].mean())})  # fmt: skip
    return out


def verdict(mean: float, lo: float, hi: float, higher_is_better: bool = True) -> str:
    if not np.isfinite(mean):
        return "n/a"
    good = lo > 0 if higher_is_better else hi < 0
    bad = hi < 0 if higher_is_better else lo > 0
    return (
        "FlareNet better (95% interval excludes 0)"
        if good
        else "NOAA better (95% interval excludes 0)"
        if bad
        else "no established difference"
    )


def by_year(df: pd.DataFrame, cols: dict[str, str]) -> list[dict]:
    """Per-calendar-year AUC / TSS(0.5) / Brier for each forecaster. ``cols`` maps name -> probability column."""
    rows = []
    for yr, g in df.groupby(df.index.year):
        row = {"year": int(yr), "days": int(len(g)), "event_days": int(g["y"].sum())}
        for name, c in cols.items():
            s = scores(g["y"], g[c], 0.5)
            row[f"{name}_AUC"], row[f"{name}_TSS"], row[f"{name}_Brier"] = (
                s["AUC"],
                s["TSS"],
                s["Brier"],
            )
        rows.append(row)
    return rows

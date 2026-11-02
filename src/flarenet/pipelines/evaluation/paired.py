"""Paired comparison of two forecasters scored on the same windows.

Two separate confidence intervals that overlap do not show that two models are equal: both move together with the
sample (the same quiet weeks, the same storms). Resampling whole time blocks of windows and scoring BOTH models on each
resample gives an interval for the DIFFERENCE, which is much tighter and is what decides whether a change helped.
Blocks keep neighbouring (overlapping) windows together, as in ``uncertainty.block_bootstrap_ci``.
"""

from __future__ import annotations

import warnings

import numpy as np

from flarenet.pipelines.evaluation.metrics import safe_auc, score_at_threshold

NAN = float("nan")


def _metrics(y, p, thr) -> dict[str, float]:
    out = {"AUC": safe_auc(y, p), "Brier": float(np.mean((np.asarray(p, dtype=float) - y) ** 2))}
    if thr is not None:
        s = score_at_threshold(y, p, thr)
        out.update({"TSS": float(s["TSS"]), "POD": float(s["POD"]), "FAR": float(s["FAR"])})
    return out


def paired_block_bootstrap(
    y, pa, pb, t_end, thr_a: float | None = None, thr_b: float | None = None,
    block_hours: float = 72.0, n: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, dict]:  # fmt: skip
    """Metrics of model A and model B on the same windows and the 95 % interval of ``B - A``.

    ``thr_a``/``thr_b`` are each model's own alert threshold (fixed beforehand, not re-tuned here); without them only
    AUC and Brier are compared. Returns ``{metric: {"a", "b", "diff", "ci95": [lo, hi]}}``; the interval is NaN when
    the metric is undefined in too many resamples."""
    y = np.asarray(y).astype(int)
    pa, pb, t_end = (np.asarray(v, dtype=float) for v in (pa, pb, t_end))
    if not (len(y) == len(pa) == len(pb) == len(t_end)):
        raise ValueError("y, pa, pb and t_end must have the same length")
    base_a, base_b = _metrics(y, pa, thr_a), _metrics(y, pb, thr_b)
    keys = list(base_a)
    out = {
        k: {"a": base_a[k], "b": base_b[k], "diff": base_b[k] - base_a[k], "ci95": [NAN, NAN]}
        for k in keys
    }
    if n <= 0 or len(y) < 2:
        return out
    blocks = np.floor((t_end - t_end.min()) / (block_hours * 60.0)).astype(int)
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a resample without flare windows leaves AUC undefined
        for _ in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            ma, mb = _metrics(y[idx], pa[idx], thr_a), _metrics(y[idx], pb[idx], thr_b)
            for k in keys:
                draws[k].append(mb[k] - ma[k])
    for k, v in draws.items():
        a = np.asarray(v, dtype=float)
        ok = a[np.isfinite(a)]
        if len(ok) >= max(10, n // 4):
            out[k]["ci95"] = [
                float(np.percentile(ok, 100 * alpha / 2)),
                float(np.percentile(ok, 100 * (1 - alpha / 2))),
            ]
    return out


def reading(metric: str, d: dict) -> str:
    """Plain-language reading of one row of :func:`paired_block_bootstrap` (B minus A)."""
    lo, hi = d["ci95"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undetermined"
    better_high = metric in ("AUC", "TSS", "POD")  # for Brier and FAR a lower value is better
    if lo > 0:
        return "B better" if better_high else "B worse"
    if hi < 0:
        return "B worse" if better_high else "B better"
    return "no established difference"

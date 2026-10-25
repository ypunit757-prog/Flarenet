"""Uncertainty and calibration of the forecast probabilities."""

from __future__ import annotations

import warnings

import numpy as np

from flarenet.pipelines.evaluation.metrics import safe_auc, score_at_threshold

NAN = float("nan")


def block_bootstrap_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    t_end: np.ndarray,
    threshold: float,
    n_boot: int = 200,
    block_hours: float = 72.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """95% intervals for TSS / HSS / POD / FAR / AUC by resampling whole time blocks.

    Windows overlap, so resampling single windows would understate the uncertainty; whole blocks
    of ``block_hours`` keep neighbouring windows together. Returns ``{metric: [lo, hi]}``
    (NaN when a metric is undefined in too many replicates, e.g. no positives).
    """
    y_true, prob, t_end = map(np.asarray, (y_true, prob, t_end))
    if n_boot <= 0 or len(y_true) < 2:
        return {}
    blocks = np.floor((t_end - t_end.min()) / (block_hours * 60.0)).astype(int)
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    keys = ("TSS", "HSS", "POD", "FAR", "AUC")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # replicates without positives make AUC undefined (NaN)
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            sc = score_at_threshold(y_true[idx], prob[idx], threshold)
            for k in ("TSS", "HSS", "POD", "FAR"):
                draws[k].append(float(sc[k]))
            draws["AUC"].append(float(safe_auc(y_true[idx], prob[idx])))
    out = {}
    for k, v in draws.items():
        a = np.asarray(v, dtype=float)
        ok = a[np.isfinite(a)]
        out[k] = (
            [
                float(np.percentile(ok, 100 * alpha / 2)),
                float(np.percentile(ok, 100 * (1 - alpha / 2))),
            ]
            if len(ok) >= max(10, n_boot // 4)
            else [NAN, NAN]
        )
    return out


def calibration_summary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability curve, Brier score and expected calibration error (equal-width bins)."""
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    if len(y_true) == 0:
        return {}
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    conf, freq, count = [], [], []
    for b in range(n_bins):
        m = which == b
        count.append(int(m.sum()))
        conf.append(float(prob[m].mean()) if m.any() else NAN)
        freq.append(float(y_true[m].mean()) if m.any() else NAN)
    cnt = np.asarray(count, dtype=float)
    gaps = np.abs(np.nan_to_num(conf) - np.nan_to_num(freq))
    return {
        "brier": float(np.mean((prob - y_true) ** 2)),
        "ece": float((cnt * gaps).sum() / cnt.sum()),
        "bin_confidence": conf,
        "bin_frequency": freq,
        "bin_count": count,
    }

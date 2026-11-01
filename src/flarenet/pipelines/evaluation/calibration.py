"""Probability calibration and alert levels, fitted on validation windows and stored in the checkpoint.

A model trained on re-balanced batches (``training.forecast_balance``, ``positive_oversample``) reports
probabilities that are much higher than the flare frequency it will meet, so "0.60" does not mean "60 % of
such windows were followed by a flare". Platt scaling (a logistic curve on the logit of the raw output, fitted on
validation windows that keep the natural flare frequency) repairs that. It is monotone, so AUC and the ranking of
windows do not change; what changes is what a probability means and where the alert threshold sits.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc

EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y, p, min_positives: int = 20) -> dict | None:
    """``{"kind": "platt", "a": a, "b": b}`` so that calibrated = sigmoid(a * logit(raw) + b), or None when the
    validation windows cannot support it (too few of either class, or a non-positive slope, which would reverse
    the model's ranking)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.sum() < min_positives or (1 - y).sum() < min_positives:
        return None
    m = LogisticRegression(C=1e4, max_iter=2000).fit(_logit(p)[:, None], y)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return {"kind": "platt", "a": a, "b": b}


def apply_calibration(cal: dict | None, p) -> np.ndarray:
    """Calibrated probabilities (unchanged when ``cal`` is None, as for older checkpoints)."""
    p = np.asarray(p, dtype=float)
    if not cal:
        return p
    if cal.get("kind") != "platt":
        raise ValueError(f"unknown calibration {cal.get('kind')!r}")
    z = float(cal["a"]) * _logit(p) + float(cal["b"])
    return 1.0 / (1.0 + np.exp(-z))


def _counts(y, p, thresholds: np.ndarray):
    """TP, FP, FN, TN at each threshold (alert when p >= threshold), in one pass over sorted scores."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return tp, fp, len(pos) - tp, len(neg) - fp


def alert_levels(
    y, p, sensitive_pod: float = 0.90, strict_far: float = 0.50, smooth: int = 0
) -> dict[str, dict | None]:
    """Three alert thresholds chosen on validation windows (``p`` already calibrated):

    * ``balanced``: the TSS-optimal threshold (what the checkpoint uses by default);
    * ``sensitive``: the highest threshold that still catches at least ``sensitive_pod`` of the positive windows;
    * ``strict``: the lowest threshold whose alerts are wrong at most ``strict_far`` of the time.

    Each entry is ``{"threshold", "POD", "FAR", "TSS"}`` on the validation windows, or None when no threshold
    with positive skill reaches the target. These are validation numbers; the evaluation report repeats them on held-out data."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    grid = np.arange(0.005, 0.9951, 0.0025)
    tp, fp, fn, tn = _counts(y, p, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        far = np.where(tp + fp > 0, fp / (tp + fp), np.nan)
        pofd = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
    tss = pod - pofd

    def entry(i: int) -> dict:
        return {
            "threshold": float(grid[i]),
            "POD": float(pod[i]),
            "FAR": float(far[i]),
            "TSS": float(tss[i]),
        }

    out: dict[str, dict | None] = {}
    thr, _ = best_threshold(y, p, smooth=smooth)
    j = {"threshold": float(thr)}
    tpb, fpb, fnb, tnb = _counts(y, p, np.array([thr]))
    j["POD"] = float(tpb[0] / (tpb[0] + fnb[0])) if tpb[0] + fnb[0] else float("nan")
    j["FAR"] = float(fpb[0] / (tpb[0] + fpb[0])) if tpb[0] + fpb[0] else float("nan")
    j["TSS"] = j["POD"] - (float(fpb[0] / (fpb[0] + tnb[0])) if fpb[0] + tnb[0] else 0.0)
    out["balanced"] = j

    # a level must also show skill (TSS > 0): "alert on everything" meets any POD target and, when flares are
    # the majority of windows, any false-alarm budget too, and would be no warning at all
    ok = np.flatnonzero(np.isfinite(pod) & (pod >= sensitive_pod) & (tss > 0))
    # the grid ascends, so the last hit is the highest threshold that still meets the POD target
    out["sensitive"] = entry(int(ok[-1])) if len(ok) else None
    ok = np.flatnonzero(np.isfinite(far) & (far <= strict_far) & (tp > 0) & (tss > 0))
    # the first hit is the lowest threshold within the false-alarm budget (the most flares caught)
    out["strict"] = entry(int(ok[0])) if len(ok) else None
    return out


def reliability_gap(y, p) -> dict:
    """Mean predicted probability against the observed flare frequency (calibration-in-the-large) and AUC."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {"mean_prob": float(np.mean(p)), "frequency": float(np.mean(y)), "AUC": safe_auc(y, p)}

"""Forecast-verification metrics (TSS, HSS, POD, FAR) and per-class detection scores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from flarenet.core.utils import get_logger

log = get_logger(__name__)
NAN = float("nan")


def contingency(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for binary arrays."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    return (
        int(np.sum((yp == 1) & (yt == 1))),
        int(np.sum((yp == 1) & (yt == 0))),
        int(np.sum((yp == 0) & (yt == 1))),
        int(np.sum((yp == 0) & (yt == 0))),
    )


def tss(tp: int, fp: int, fn: int, tn: int) -> float:
    """True Skill Statistic = POD - POFD, in [-1, 1]; insensitive to class imbalance."""
    pod = tp / (tp + fn) if tp + fn > 0 else 0.0
    pofd = fp / (fp + tn) if fp + tn > 0 else 0.0
    return pod - pofd


def hss(tp: int, fp: int, fn: int, tn: int) -> float:
    """Heidke Skill Score."""
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / den if den > 0 else NAN


def score_at_threshold(y_true, prob, threshold: float = 0.5) -> dict:
    """Contingency counts and skill scores after thresholding probabilities."""
    tp, fp, fn, tn = contingency(y_true, (np.asarray(prob) >= threshold).astype(int))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "POD": tp / (tp + fn) if tp + fn > 0 else NAN,
        "FAR": fp / (tp + fp) if tp + fp > 0 else NAN,
        "TSS": tss(tp, fp, fn, tn),
        "HSS": hss(tp, fp, fn, tn),
    }


def best_threshold(y_true, prob, n: int = 249, smooth: int = 0) -> tuple[float, float]:
    """Threshold in [0.01, 0.99] maximising TSS. Returns (threshold, TSS at that threshold).

    ``smooth=k`` averages the TSS curve over +-k neighbouring thresholds before taking the maximum,
    which avoids picking a spike that only exists because of sampling noise in the validation set.
    """
    prob = np.asarray(prob)
    grid = np.linspace(0.01, 0.99, n)
    curve = np.array([tss(*contingency(y_true, (prob >= thr).astype(int))) for thr in grid])
    pick = curve
    if smooth > 0:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        padded = np.pad(curve, smooth, mode="edge")
        pick = np.convolve(padded, k, mode="valid")
    i = int(np.argmax(pick))  # first maximum, like the original strict '>' loop
    return float(grid[i]), float(curve[i])


def safe_auc(y_true, prob) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    try:
        return float(roc_auc_score(y_true, prob))
    except ValueError:
        return NAN


def safe_roc(y_true, prob) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve, or the diagonal when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y_true, prob)
    return fpr, tpr


def per_class_metrics(true_cls, pred_cls, label: str = "") -> list[dict]:
    """Binary detection scores for 'at least C', 'at least M', 'at least X' nowcasts."""
    results = []
    lines = [
        f"{'Cls':>4} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'POD':>6} {'FAR':>6} {'TSS':>6} {'HSS':>6}"
    ]
    for idx, name in [(2, "C"), (3, "M"), (4, "X")]:
        yt = (np.asarray(true_cls) >= idx).astype(int)
        yp = (np.asarray(pred_cls) >= idx).astype(int)
        tp, fp, fn, tn = contingency(yt, yp)
        row = {
            "class": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "POD": tp / (tp + fn) if tp + fn > 0 else NAN,
            "FAR": fp / (tp + fp) if tp + fp > 0 else NAN,
            "TSS": tss(tp, fp, fn, tn),
            "HSS": hss(tp, fp, fn, tn),
        }
        results.append(row)
        lines.append(
            f"{name:>4} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>5d} "
            f"{row['POD']:>6.3f} {row['FAR']:>6.3f} {row['TSS']:>6.3f} {row['HSS']:>6.3f}"
        )
    log.info("Per-class detection [%s]\n%s", label, "\n".join(lines))
    return results

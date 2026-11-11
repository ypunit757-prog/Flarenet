"""The full set of scores for one binary forecaster at one alert threshold.

Accuracy, precision, recall and F1 are reported because they are the usual ones, but a forecast of "M/X flare within
6 hours" is imbalanced: a model that never alerts is right on every quiet window. Read TSS, ROC-AUC, PR-AUC (compare it with
``prevalence``, the score of a random ranking) and the confusion matrix, which cannot be inflated that way.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from flarenet.pipelines.evaluation.metrics import contingency, hss, safe_auc, safe_roc, tss

NAN = float("nan")


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else NAN


def full_metrics(y, p, thr: float) -> dict:
    """Scores of probabilities ``p`` against 0/1 outcomes ``y``, alerting when ``p >= thr``.

    ``confusion`` is ``[[TN, FP], [FN, TP]]`` (rows: truth quiet / flare, columns: no alert / alert)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tp, fp, fn, tn = contingency(y, (p >= thr).astype(int))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else NAN
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr_auc = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else NAN
    return {
        "threshold": float(thr),
        "accuracy": _ratio(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "ROC_AUC": safe_auc(y, p),
        "PR_AUC": pr_auc,
        "prevalence": float(y.mean()) if len(y) else NAN,
        "TSS": float(tss(tp, fp, fn, tn)),
        "HSS": float(hss(tp, fp, fn, tn)),
        "Brier": float(np.mean((p - y) ** 2)) if len(y) else NAN,
        "confusion": [[tn, fp], [fn, tp]],
    }


def roc_points(y, p) -> tuple[np.ndarray, np.ndarray]:
    """False-positive rate and true-positive rate for a ROC plot (empty when only one class is present)."""
    return safe_roc(np.asarray(y).astype(int), np.asarray(p, dtype=float))


def pr_points(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision for a precision-recall plot (empty when there are no positives)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return np.zeros(0), np.zeros(0)
    prec, rec, _ = precision_recall_curve(y, np.asarray(p, dtype=float))
    return rec, prec


def thin_curve(x, y, n: int = 80, digits: int = 4) -> list[list[float]]:
    """At most ``n`` points of a curve, evenly spaced along its length, always keeping both ends (for the web page)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0:
        return []
    if len(x) > n:
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        if d[-1] > 0:
            keep = np.unique(
                np.concatenate([[0], np.searchsorted(d, np.linspace(0, d[-1], n)), [len(x) - 1]])
            )
            keep = keep[keep < len(x)]
        else:
            keep = np.array([0, len(x) - 1])
        x, y = x[keep], y[keep]
    return [[round(float(a), digits), round(float(b), digits)] for a, b in zip(x, y, strict=True)]


def curves(y, p) -> dict:
    """ROC and precision-recall curves of one model, thinned, for the Results page."""
    y = np.asarray(y).astype(int)
    if (
        not 0 < y.sum() < len(y)
    ):  # one class only: there is no curve to draw (safe_roc would give the diagonal)
        return {"roc": [], "pr": []}
    fpr, tpr = roc_points(y, p)
    rec, prec = pr_points(y, p)
    return {"roc": thin_curve(fpr, tpr), "pr": thin_curve(rec, prec)}

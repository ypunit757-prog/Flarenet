"""Global importance: how much each group of inputs matters to a model over a whole set of windows.

* ``permutation_importance``: shuffle one group across the windows (so it no longer matches the outcome, but keeps its own
  distribution) and measure how much the ROC-AUC falls. A large fall means the model relied on the group; zero means it did
  not (or the group is duplicated by another). Several shuffles give a spread.
* ``mean_abs_shapley``: the average size of the per-window Shapley contributions over a sample of windows.
* ``shap_feature_importance`` (optional): the ``shap`` library's per-feature mean |SHAP|, when it is installed.

All of these describe the model, not the Sun: a feature the model ignores can still matter physically.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from flarenet.explain.local import network_effects, tree_effects
from flarenet.pipelines.evaluation.metrics import safe_auc


def _auc_drop(y, base: float, shuffled: list[float]) -> dict:
    d = base - np.asarray(shuffled, dtype=float)
    return {
        "auc_drop": float(np.nanmean(d)),
        "auc_drop_std": float(np.nanstd(d)),
        "repeats": len(shuffled),
    }


def permutation_importance_table(
    predict_proba: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y,
    groups: dict[str, tuple[str, list[str]]], repeats: int = 5, seed: int = 0,
) -> dict:  # fmt: skip
    """AUC fall when each feature group of the table ``X`` is shuffled (the columns of a group move together, keeping
    their relationships to each other)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    base = safe_auc(y, predict_proba(X))
    out = {"baseline_auc": base, "groups": {}}
    for gid, (label, cols) in groups.items():
        scores = []
        for _ in range(repeats):
            Xs = X.copy()
            perm = rng.permutation(len(X))
            Xs[cols] = X[cols].to_numpy()[perm]
            scores.append(safe_auc(y, predict_proba(Xs)))
        out["groups"][gid] = {"label": label, **_auc_drop(y, base, scores)}
    return out


def permutation_importance_windows(
    predict_proba: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y,
    groups: dict[str, tuple[str, list[int]]], repeats: int = 3, seed: int = 0,
) -> dict:  # fmt: skip
    """The same for the sequence network: a channel is shuffled between windows (each window keeps a whole, real channel)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    base = safe_auc(y, predict_proba(X))
    out = {"baseline_auc": base, "groups": {}}
    for gid, (label, chans) in groups.items():
        scores = []
        for _ in range(repeats):
            Xs = X.copy()
            perm = rng.permutation(len(X))
            for c in chans:
                Xs[:, c] = X[perm, c]
            scores.append(safe_auc(y, predict_proba(Xs)))
        out["groups"][gid] = {"label": label, **_auc_drop(y, base, scores)}
    return out


def mean_abs_shapley_table(
    predict_proba: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, reference: pd.Series,
    groups: dict[str, tuple[str, list[str]]], n: int = 200, seed: int = 0,
) -> dict[str, dict]:  # fmt: skip
    """Mean absolute and mean signed Shapley contribution (log-odds) of each group over ``n`` randomly chosen rows."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), size=min(n, len(X)), replace=False)
    acc = {g: [] for g in groups}
    for i in pick:
        eff = tree_effects(predict_proba, X.iloc[i], reference, groups)["effects"]
        for g, v in eff.items():
            acc[g].append(v)
    return {g: {"label": groups[g][0], "mean_abs": float(np.mean(np.abs(v))), "mean": float(np.mean(v)), "rows": len(v)}
            for g, v in acc.items()}  # fmt: skip


def mean_abs_shapley_windows(
    predict_proba: Callable[[np.ndarray], np.ndarray], X: np.ndarray, reference: np.ndarray,
    groups: dict[str, tuple[str, list[int]]], n: int = 60, seed: int = 0,
) -> dict[str, dict]:  # fmt: skip
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), size=min(n, len(X)), replace=False)
    acc = {g: [] for g in groups}
    for i in pick:
        eff = network_effects(predict_proba, X[i], reference, groups)["effects"]
        for g, v in eff.items():
            acc[g].append(v)
    return {g: {"label": groups[g][0], "mean_abs": float(np.mean(np.abs(v))), "mean": float(np.mean(v)), "rows": len(v)}
            for g, v in acc.items()}  # fmt: skip


def shap_feature_importance(
    predict_proba: Callable[[pd.DataFrame], np.ndarray], background: pd.DataFrame, X: pd.DataFrame,
    max_rows: int = 100, seed: int = 0,
) -> dict[str, float]:  # fmt: skip
    """Mean |SHAP| per feature from the ``shap`` library (model-agnostic permutation explainer). Raises ImportError when
    ``shap`` is not installed (``pip install shap``)."""
    import shap

    bg = background.sample(min(50, len(background)), random_state=seed)
    xs = X.sample(min(max_rows, len(X)), random_state=seed)
    explainer = shap.PermutationExplainer(
        lambda a: predict_proba(pd.DataFrame(a, columns=X.columns)), bg.to_numpy(float)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer(xs.to_numpy(float), max_evals=2 * X.shape[1] + 1)
    return dict(zip(X.columns, map(float, np.abs(sv.values).mean(axis=0)), strict=True))

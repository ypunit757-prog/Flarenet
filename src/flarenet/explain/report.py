"""Global explanation report for a trained network + tree model on one held-out dataset (``scripts/explain_model.py``)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.explain.groups import channel_groups, tree_groups
from flarenet.explain.importance import (
    mean_abs_shapley_table,
    mean_abs_shapley_windows,
    permutation_importance_table,
    permutation_importance_windows,
    shap_feature_importance,
)

METHOD = (
    "Permutation importance (fall in ROC-AUC when a group of inputs is shuffled between windows) and the mean size of "
    "Shapley contributions over a sample of windows, on a dataset the models were not trained on."
)


def _rows(perm: dict, shap_rows: dict) -> list[dict]:
    out = []
    for gid, d in perm["groups"].items():
        s = shap_rows.get(gid, {})
        out.append({"id": gid, "label": d["label"], "auc_drop": round(d["auc_drop"], 4), "auc_drop_std": round(d["auc_drop_std"], 4),
                    "mean_abs_shapley": round(s.get("mean_abs", float("nan")), 4)})  # fmt: skip
    out.sort(key=lambda r: -r["auc_drop"])
    return out


def build_global_report(
    dataset: str, period: dict | None, y, tab: pd.DataFrame, X: np.ndarray, net_proba, live, channels: list[str],
    reference_window: np.ndarray, repeats: int = 5, n_tree: int = 200, n_net: int = 60, use_shap: bool = False, seed: int = 0,
) -> dict:  # fmt: skip
    """``tab`` (feature rows) and ``X`` (network windows) are the same windows, ``y`` the outcome of each."""
    y = np.asarray(y).astype(int)
    cols = list(live.gbm.cols)
    tab = tab.reindex(columns=cols)
    tg, cg = tree_groups(cols), channel_groups(channels)
    ref = live.reference.reindex(cols)
    perm_t = permutation_importance_table(live.gbm.predict_proba, tab, y, tg, repeats, seed)
    shap_t = mean_abs_shapley_table(live.gbm.predict_proba, tab, ref, tg, n_tree, seed)
    perm_n = permutation_importance_windows(net_proba, X, y, cg, max(1, repeats // 2), seed)
    shap_n = mean_abs_shapley_windows(net_proba, X, reference_window, cg, n_net, seed)
    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "method": METHOD,
        "dataset": dataset, "period": period, "windows": int(len(y)), "flare_windows": int(y.sum()),
        "trees": {"baseline_auc": round(perm_t["baseline_auc"], 4), "groups": _rows(perm_t, shap_t)},
        "network": {"baseline_auc": round(perm_n["baseline_auc"], 4), "groups": _rows(perm_n, shap_n)},
        "shap_features": None, "shap_note": None,
    }  # fmt: skip
    if use_shap:
        try:
            imp = shap_feature_importance(live.gbm.predict_proba, tab, tab, seed=seed)
            report["shap_features"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
        except ImportError:
            report["shap_note"] = "the shap package is not installed (pip install shap)"
    return report


def render_markdown(rep: dict) -> str:
    L = [f"# What the models rely on: {rep['dataset']}", "",
         f"{rep['windows']:,} windows, {rep['flare_windows']:,} followed by an M/X flare within 6 h"
         + (f" ({rep['period']['start']} to {rep['period']['end']})" if rep.get("period") else "") + ". " + rep["method"], ""]  # fmt: skip
    for key, title in (
        ("trees", "Tree model (days of X-ray history)"),
        ("network", "Neural network (last 90 minutes)"),
    ):
        d = rep[key]
        L += [f"## {title}", "", f"AUC on these windows: {d['baseline_auc']:.3f}.", "",
              "| input group | AUC lost when shuffled | mean size of contribution (log-odds) |", "|---|---|---|"]  # fmt: skip
        L += [
            f"| {r['label']} | {r['auc_drop']:+.4f} (+-{r['auc_drop_std']:.4f}) | {r['mean_abs_shapley']:.3f} |"
            for r in d["groups"]
        ]
        L.append("")
    if rep.get("shap_features"):
        L += [
            "## SHAP, per feature (tree model)",
            "",
            "| feature | mean abs SHAP (probability) |",
            "|---|---|",
        ]
        L += [f"| {k} | {v:.4f} |" for k, v in rep["shap_features"].items()]
        L.append("")
    elif rep.get("shap_note"):
        L += [f"SHAP: {rep['shap_note']}.", ""]
    L += ["## How to read this", "",
          "- A group with a large AUC loss is one the model relies on. A loss near zero means the model does not use the group, "
          "or another group carries the same information. It is a statement about the model, not about the Sun.",
          "- Shuffling breaks the link between a group and the outcome but not the group's own distribution; strongly "
          "correlated groups can share credit.",
          "- The contributions are Shapley values against a typical window, in log-odds; their size says how far the forecast "
          "moves, not how likely a flare is.", ""]  # fmt: skip
    return "\n".join(L)


def plot_global(rep: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title, color in (
        (axes[0], "trees", "Tree model", "#2a9d8f"),
        (axes[1], "network", "Neural network", "#1f6feb"),
    ):
        g = rep[key]["groups"][:10][::-1]
        ax.barh(
            [r["label"][:42] for r in g],
            [r["auc_drop"] for r in g],
            xerr=[r["auc_drop_std"] for r in g],
            color=color,
        )
        ax.set_title(f"{title} (AUC {rep[key]['baseline_auc']:.3f})", fontsize=10)
        ax.set_xlabel("AUC lost when the group is shuffled")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"What the models rely on: {rep['dataset']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

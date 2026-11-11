"""ROC, precision-recall and confusion-matrix figures for a model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flarenet.compare.metrics import pr_points, roc_points  # noqa: E402

COLORS = ["#1f6feb", "#d1495b", "#2a9d8f", "#e9a100", "#7b5ea7", "#555555"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_roc_pr(
    title: str, y, probs: dict[str, np.ndarray], labels: dict[str, str], metrics: dict, path: Path
) -> Path:
    """One figure, two panels: ROC (left) and precision-recall (right) for every model on the same windows."""
    y = np.asarray(y).astype(int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (name, p) in enumerate(probs.items()):
        c = COLORS[i % len(COLORS)]
        fpr, tpr = roc_points(y, p)
        rec, prec = pr_points(y, p)
        m = metrics[name]
        a.plot(fpr, tpr, color=c, lw=1.6, label=f"{labels[name]} (AUC {m['ROC_AUC']:.3f})")
        b.plot(rec, prec, color=c, lw=1.6, label=f"{labels[name]} (AP {m['PR_AUC']:.3f})")
    a.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random")
    prev = float(y.mean()) if len(y) else 0.0
    b.axhline(prev, color="grey", ls="--", lw=1, label=f"random ({prev:.2f})")
    _style(a, "ROC curve", "false-positive rate", "true-positive rate")
    _style(b, "Precision-recall curve", "recall", "precision")
    a.legend(fontsize=7, loc="lower right")
    b.legend(fontsize=7, loc="lower left")
    b.set_ylim(0, 1.02)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confusions(title: str, metrics: dict, labels: dict[str, str], path: Path) -> Path:
    """Confusion matrices at each model's own (validation-chosen) threshold: counts, shaded by the share of each true class."""
    names = list(metrics)
    cols = min(3, len(names))
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names, strict=False):
        cm = np.asarray(metrics[name]["confusion"], dtype=float)
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{int(v):,}", ha="center", va="center", fontsize=9,
                    color="white" if share[r, c] > 0.55 else "black")  # fmt: skip
        ax.set_xticks([0, 1], ["no alert", "alert"], fontsize=8)
        ax.set_yticks([0, 1], ["quiet", "M/X follows"], fontsize=8)
        ax.set_title(f"{labels[name]}\nthreshold {metrics[name]['threshold']:.2f}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

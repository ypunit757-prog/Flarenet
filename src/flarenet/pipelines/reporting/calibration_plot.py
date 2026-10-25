"""Reliability diagram for the forecast probabilities of every evaluated dataset."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flarenet.core.constants import BG_DARK, BG_PANEL, GRID_KW  # noqa: E402

PALETTE = ["#4a90e2", "#e67e22", "#2ecc71", "#9b59b6", "#f1c40f", "#e74c3c"]


def render_reliability(results: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
    for i, (_, res) in enumerate(results.items()):
        cal = res.calibration
        if not cal:
            continue
        conf, freq = np.asarray(cal["bin_confidence"]), np.asarray(cal["bin_frequency"])
        m = np.isfinite(conf) & np.isfinite(freq)
        ax.plot(conf[m], freq[m], "o-", color=PALETTE[i % len(PALETTE)], ms=4,
                label=f"{res.name}  Brier {cal['brier']:.3f}  ECE {cal['ece']:.3f}")  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast probability", color="white")
    ax.set_ylabel("observed M/X frequency", color="white")
    ax.set_title("Reliability", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(**GRID_KW)
    leg = ax.legend(fontsize=7, facecolor="#1c2128", edgecolor="#555")
    for t in leg.get_texts():
        t.set_color("white")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

"""Dark-theme 12-panel evaluation dashboard and a single-panel forecast timeline plot."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from flarenet.core.constants import (  # noqa: E402
    BG_DARK,
    BG_PANEL,
    CLASS_COLORS,
    CLASS_NAMES,
    GRID_KW,
    M_CLASS_IDX,
    N_CLASSES,
)
from flarenet.core.utils import get_logger  # noqa: E402
from flarenet.pipelines.evaluation.scoring import EvalResult  # noqa: E402

log = get_logger(__name__)
SYN_DAYS_SHOWN = 3  # synthetic lightcurves are long; show one contiguous 3-day stretch


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    if title:
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.grid(**GRID_KW)


def _legend(ax, **kw) -> None:
    leg = ax.legend(fontsize=7, facecolor="#1c2128", edgecolor="#555", **kw)
    for t in leg.get_texts():
        t.set_color("white")


def _placeholder(ax, text: str) -> None:
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes, color="#888", ha="center", va="center", fontsize=10
    )


def _view_slice(result: EvalResult, df: pd.DataFrame, real: bool):
    """Pick the time range to plot: all of the real day, or the first 3 test days for synthetic."""
    if real:
        lo, hi = float(df.time_min.min()), float(df.time_min.max()) + 1
    else:
        lo = float(result.times.min())
        hi = lo + SYN_DAYS_SHOWN * 1440
    return lo, hi


def _alert_timeline(ax, times, probs, threshold, flare_db, lo, hi, title) -> None:
    _style(ax, title)
    m = (times >= lo) & (times < hi)
    t_h, p = times[m] / 60, probs[m]
    ax.plot(t_h, p, color="#f1c40f", lw=1.1, label="P(M/X within horizon)", zorder=3)
    ax.axhline(
        threshold, ls="--", color="#e74c3c", lw=1.4, label=f"Alert thr={threshold:.2f}", zorder=4
    )
    ax.fill_between(
        t_h, 0, p, where=(p >= threshold), color="#e74c3c", alpha=0.25, label="ALERT", zorder=2
    )
    if len(flare_db):
        for _, ev in flare_db[
            (flare_db.class_idx >= M_CLASS_IDX)
            & (flare_db.onset_min >= lo)
            & (flare_db.onset_min < hi)
        ].iterrows():
            ax.axvline(ev.onset_min / 60, color="#9b59b6", lw=1.2, ls=":", zorder=5)
            ax.text(
                ev.onset_min / 60,
                0.96,
                ev["class"],
                color="#9b59b6",
                fontsize=7,
                ha="center",
                va="top",
            )
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Forecast P")
    _legend(ax)


def render_dashboard(
    syn: EvalResult,
    real: EvalResult | None,
    df_syn: pd.DataFrame,
    flare_db_syn: pd.DataFrame,
    df_real: pd.DataFrame | None,
    flare_db_real: pd.DataFrame | None,
    history: dict,
    threshold: float,
    save_path: str,
) -> None:
    """Render the dashboard. Real-data panels fall back to synthetic or a placeholder if absent."""
    use_real = real is not None and df_real is not None
    view_res = real if use_real else syn
    view_df = df_real if use_real else df_syn
    view_db = flare_db_real if use_real else flare_db_syn
    view_name = real.name if use_real else "Synthetic test"
    lo, hi = _view_slice(view_res, view_df, use_real)

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38, top=0.94, bottom=0.04)

    # Row 0: lightcurves and alert timeline
    seg = view_df[(view_df.time_min >= lo) & (view_df.time_min < hi)]
    t_hr = seg.time_min.to_numpy() / 60
    ax = fig.add_subplot(gs[0, 0])
    _style(ax, f"Soft X-ray | {view_name}")
    sxr = seg.sxr_raw.to_numpy()
    ax.plot(t_hr, sxr, color="#4a90e2", lw=0.6, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hour")
    ax.set_ylabel("soft-channel flux (source units)")
    ylo, yhi = np.nanmin(sxr) * 0.5, np.nanmax(sxr) * 2
    for ci, cn in enumerate(CLASS_NAMES[1:], 1):
        m = seg.flare_class.to_numpy() == ci
        if m.any():
            ax.fill_between(
                t_hr, ylo, yhi, where=m, color=CLASS_COLORS[cn], alpha=0.22, label=cn, zorder=1
            )
    ax.set_ylim(ylo, yhi)
    if ax.get_legend_handles_labels()[0]:
        _legend(ax)

    ax = fig.add_subplot(gs[0, 1])
    _style(ax, f"Hard X-ray | {view_name}")
    hxr_col = "hxr_raw" if "hxr_raw" in seg.columns else "hxr"
    ax.plot(t_hr, seg[hxr_col].to_numpy(), color="#e74c3c", lw=0.6)
    ax.set_xlabel("Hour")
    ax.set_ylabel("soft-channel flux (source units)")

    ax = fig.add_subplot(gs[0, 2])
    _alert_timeline(
        ax,
        view_res.times,
        view_res.forecast_prob,
        threshold,
        view_db,
        lo,
        hi,
        f"Forecast alert timeline | {view_name}",
    )

    # Row 1: ROC curves and per-class TSS
    for col, (res, color) in enumerate([(syn, "#4a90e2"), (real, "#e67e22")]):
        ax = fig.add_subplot(gs[1, col])
        if res is None:
            _style(ax, "ROC | external data")
            _placeholder(ax, "No external dataset supplied")
            continue
        _style(ax, f"ROC | {res.name}")
        ax.plot(*res.roc, color=color, lw=1.4, label=f"AUC={res.auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#555")
        pofd = res.scores["FP"] / max(res.scores["FP"] + res.scores["TN"], 1)
        pod = res.scores["POD"] if np.isfinite(res.scores["POD"]) else 0.0
        ax.scatter(
            [pofd], [pod], color="#e74c3c", s=60, zorder=5, label=f"TSS={res.scores['TSS']:.3f}"
        )
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        _legend(ax)

    ax = fig.add_subplot(gs[1, 2])
    _style(ax, "Per-class detection TSS (C/M/X)")
    xp, w = np.arange(3), 0.35
    series = [("Synthetic", syn, "#4a90e2", -w / 2)]
    if real is not None:
        series.append(("Real", real, "#e67e22", w / 2))
    for label, res, color, off in series:
        vals = [r["TSS"] for r in res.per_class]
        bars = ax.bar(xp + off, vals, w, color=color, alpha=0.85, label=label)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.025,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
            )
    ax.set_xticks(xp)
    ax.set_xticklabels(["C", "M", "X"], color="white")
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_ylabel("TSS")
    _legend(ax)

    # Row 2: training curves and lead-time histogram
    epochs = range(1, len(history["train_loss"]) + 1)
    ax = fig.add_subplot(gs[2, 0])
    _style(ax, "Training loss")
    ax.plot(epochs, history["train_loss"], color="#4a90e2", label="Train")
    ax.plot(epochs, history["val_loss"], color="#e74c3c", label="Val")
    ax.set_xlabel("Epoch")
    _legend(ax)

    ax = fig.add_subplot(gs[2, 1])
    _style(ax, "Validation forecast TSS and AUC")
    ax.plot(epochs, history["val_tss"], color="#e74c3c", label="Val TSS")
    ax.plot(epochs, history["val_auc"], color="#9b59b6", label="Val AUC")
    ax.axhline(0.6, ls="--", color="#555", lw=0.9, label="TSS=0.6")
    ax.set_xlabel("Epoch")
    _legend(ax)

    ax = fig.add_subplot(gs[2, 2])
    _style(ax, "Lead time of M/X hits (min), by dataset")
    plotted = False
    for res, color in [(syn, "#4a90e2"), (real, "#e67e22")]:
        if res is None or len(res.lead) == 0:
            continue
        leads = res.lead[res.lead.status == "HIT"]["lead_time_min"].dropna()
        if len(leads):
            ax.hist(
                leads,
                bins=min(20, max(len(leads), 1)),
                color=color,
                edgecolor="#111",
                alpha=0.7,
                label=f"{res.name} (median {leads.median():.0f})",
            )
            plotted = True
    if plotted:
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("Count")
        _legend(ax)
    else:
        _placeholder(ax, "No M/X hits to plot")

    # Row 3: confusion matrix, flare table, metrics summary
    cm_res = real if real is not None else syn
    ax = fig.add_subplot(gs[3, 0])
    _style(ax, f"Nowcast confusion matrix | {cm_res.name}")
    cm = confusion_matrix(cm_res.nowcast_true, cm_res.nowcast_pred, labels=list(range(N_CLASSES)))
    ax.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=8)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    half = cm.max() / 2 if cm.max() else 1
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > half else "#222",
            )

    ax = fig.add_subplot(gs[3, 1])
    ax.set_facecolor(BG_PANEL)
    ax.axis("off")
    ax.set_title(f"Flare database | {view_name}", color="white", fontsize=9, fontweight="bold")
    if view_db is not None and len(view_db) > 0:
        db = view_db[(view_db.onset_min >= lo) & (view_db.onset_min < hi)].head(14)
        cols = (
            ["class", "onset_utc", "duration_min", "peak_sxr"]
            if "onset_utc" in db.columns
            else ["class", "onset_min", "duration_min", "peak_sxr"]
        )
        if len(db):
            shown = db[cols].copy()
            shown["peak_sxr"] = shown["peak_sxr"].apply(lambda v: f"{v:.2e}")
            shown["duration_min"] = shown["duration_min"].round(1)
            tbl = ax.table(
                cellText=shown.values,
                colLabels=[c.replace("_min", " (min)").replace("_utc", " (UTC)") for c in cols],
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            for (r, _c), cell in tbl.get_celld().items():
                cell.set_facecolor("#21262d" if r == 0 else "#1c2128")
                cell.set_text_props(color="white")
                cell.set_edgecolor("#444")
            tbl.auto_set_column_width(list(range(len(cols))))
        else:
            _placeholder(ax, "No events in view")
    else:
        _placeholder(ax, "No flare events detected")

    ax = fig.add_subplot(gs[3, 2])
    ax.set_facecolor(BG_PANEL)
    ax.axis("off")
    ax.set_title("Key metrics summary", color="white", fontsize=9, fontweight="bold")
    lines: list[tuple[str, str]] = []
    for res in [syn] + ([real] if real is not None else []):
        lines += [
            (f"== {res.name}", ""),
            ("  Nowcast acc", f"{res.accuracy:.4f}"),
            ("  Forecast AUC", f"{res.auc:.4f}"),
            ("  Forecast TSS", f"{res.scores['TSS']:.4f}"),
            ("  POD / FAR", f"{res.scores['POD']:.3f} / {res.scores['FAR']:.3f}"),
        ]
    lines.append(("== Alert threshold", f"{threshold:.2f}"))
    y = 0.97
    for key, val in lines:
        header = key.startswith("==")
        ax.text(
            0.02,
            y,
            key,
            transform=ax.transAxes,
            color="#f1c40f" if header else "white",
            fontsize=8.5,
            fontweight="bold" if header else "normal",
            va="top",
        )
        if val:
            ax.text(0.62, y, val, transform=ax.transAxes, color="#aaaaaa", fontsize=8.5, va="top")
        y -= 0.065

    plt.suptitle(
        "FlareNet | Solar flare nowcasting and forecasting dashboard",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    log.info("Dashboard saved to %s", save_path)


def render_forecast_plot(
    predictions: pd.DataFrame, df: pd.DataFrame, threshold: float, save_path: str, title: str = ""
) -> None:
    """Single-figure forecast timeline for `flarenet predict` output."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.patch.set_facecolor(BG_DARK)
    _style(a1, title or "Soft X-ray lightcurve")
    _style(a2, "Forecast P(M/X within horizon)")
    a1.plot(df.time_min / 60, df.sxr_raw, color="#4a90e2", lw=0.6)
    a1.set_yscale("log")
    a1.set_ylabel("cts/s")
    t, p = predictions.time_min.to_numpy() / 60, predictions.p_mx_6h.to_numpy()
    a2.plot(t, p, color="#f1c40f", lw=1.2)
    a2.axhline(threshold, ls="--", color="#e74c3c", label=f"Alert thr={threshold:.2f}")
    a2.fill_between(t, 0, p, where=p >= threshold, color="#e74c3c", alpha=0.25)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("Hour")
    a2.set_ylabel("P")
    _legend(a2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=BG_DARK)
    plt.close(fig)
    log.info("Forecast plot saved to %s", save_path)

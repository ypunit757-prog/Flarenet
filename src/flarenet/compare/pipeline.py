"""Train and score every model on the same windows, and write one comparison.

Rules that keep the comparison fair (the same ones ``flarenet.gbm6h.pipeline`` follows):

* every new model is fitted on TRAINING windows only;
* probability calibration (Platt) and every alert threshold are chosen on VALIDATION windows only;
* the ``eval`` sources (e.g. GOES-18) and the later period of a training source are only scored, never used for a choice;
* all models are scored on the identical windows, and the differences from the CNN-BiLSTM carry a paired block-bootstrap
  interval, so "model A scored higher" is not mistaken for "model A is better".

One training run per model (one seed): the intervals cover the sampling of the test windows, not the seed-to-seed
variation of the training, which for the neural networks is large (see docs/IMPROVEMENTS.md).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.compare.metrics import curves, full_metrics
from flarenet.compare.models import fit_lstm, random_forest, xgboost_model
from flarenet.compare.plots import plot_confusions, plot_roc_pr
from flarenet.core import jsonsafe
from flarenet.core.utils import get_logger
from flarenet.gbm6h.features import set_features
from flarenet.gbm6h.model import HourlyGBM
from flarenet.gbm6h.pipeline import drop_features
from flarenet.gbm6h.stack import fit_stack
from flarenet.pipelines.evaluation.calibration import apply_calibration, fit_platt
from flarenet.pipelines.evaluation.metrics import best_threshold
from flarenet.pipelines.evaluation.paired import paired_block_bootstrap, reading
from flarenet.pipelines.evaluation.pipeline import align_channels
from flarenet.pipelines.evaluation.scoring import infer

log = get_logger(__name__)

LABELS = {
    "cnn_bilstm": "CNN-BiLSTM (FlareNet network)",
    "lstm": "LSTM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient boosting (tree model)",
    "stack": "Stack: network + gradient boosting",
}
SHORT = {
    "cnn_bilstm": "CNN-BiLSTM", "lstm": "LSTM", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "gradient_boosting": "Grad. boosting", "stack": "Stack",
}  # fmt: skip
FAMILY = {
    "cnn_bilstm": "deep learning, 90-minute window of 12 channels",
    "lstm": "deep learning, same window, one LSTM layer",
    "random_forest": "bagged trees, long-memory feature table",
    "xgboost": "boosted trees, long-memory feature table",
    "gradient_boosting": "boosted trees + logistic blend, long-memory feature table",
    "stack": "logistic blend of the CNN-BiLSTM and the gradient-boosting probabilities",
}
BASE = "cnn_bilstm"
ALL_MODELS = ("cnn_bilstm", "lstm", "random_forest", "xgboost", "gradient_boosting")
# already calibrated on validation data by their own training; the others get a Platt curve fitted here
SELF_CALIBRATED = {"cnn_bilstm", "gradient_boosting", "stack"}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.split(" ")[0])


def _period(sub, bundles) -> dict | None:
    """First and last window end (UTC) of a set of windows, over all of its sources."""
    if len(sub) == 0:
        return None
    lo, hi = [], []
    for k, name in enumerate(sub.source_names):
        sel = sub.source_idx == k
        if sel.any() and name in bundles:
            t0 = bundles[name].t0_unix
            lo.append(t0 + float(sub.t_end[sel].min()) * 60.0)
            hi.append(t0 + float(sub.t_end[sel].max()) * 60.0)
    if not lo:
        return None
    iso = lambda u: pd.Timestamp(u, unit="s", tz="UTC").strftime("%Y-%m-%d")  # noqa: E731
    return {"start": iso(min(lo)), "end": iso(max(hi))}


def compare_models(
    cfg, splits: dict, bundles: dict, lm, device, out: Path, train_stride: int = 60, n_boot: int = 1000,
    seed: int = 0, drop: tuple[str, ...] = (), lstm_epochs: int = 8, skip: tuple[str, ...] = (),
) -> dict:  # fmt: skip
    """Fit the models, score them on validation and on every held-out set, write the files and return the report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    horizon = cfg.data.horizon_min
    skipped: dict[str, str] = {m: "skipped by request" for m in skip}

    t = time.time()
    tab_tr, rows_tr = set_features(splits["train"], bundles, horizon, train_stride)
    tab_tr, dropped = drop_features(tab_tr, drop)
    train = splits["train"].subset(rows_tr)
    y_tr = np.asarray(train.y_fc).astype(int)
    log.info(
        "features for %d training windows (%d flare) in %.1f min",
        len(y_tr),
        int(y_tr.sum()),
        (time.time() - t) / 60,
    )

    def with_features(ws):
        tab, rows = set_features(ws, bundles, horizon)
        return ws.subset(rows), drop_features(tab, drop)[0]

    val, tab_v = with_features(splits["val"])
    y_val = np.asarray(val.y_fc).astype(int)

    # ---- fit
    predictors: dict = {
        BASE: lambda ws, tab: infer(lm.model, align_channels(ws, lm.channels).X, lm.mu, lm.sg, device,
                                    calibration=lm.calibration)[2],
    }  # fmt: skip
    notes: dict[str, str] = {}

    def fit(name: str, make) -> None:
        if name in skip:
            return
        t0 = time.time()
        try:
            model = make()
        except ImportError as e:
            skipped[name] = f"not installed ({e.name or e}); pip install {e.name or name}"
            return
        predictors[name] = model
        log.info("%s fitted in %.1f min", LABELS[name], (time.time() - t0) / 60)

    if "lstm" not in skip:

        def make_lstm():
            m = fit_lstm(align_channels(train, lm.channels).X, y_tr, align_channels(val, lm.channels).X, y_val,
                         lm.mu, lm.sg, device, epochs=lstm_epochs, seed=seed)  # fmt: skip
            notes["lstm"] = (
                f"best epoch {m.best_epoch} of at most {lstm_epochs} (validation AUC {m.val_auc:.3f})"
            )
            return lambda ws, tab: m.predict_proba(align_channels(ws, lm.channels).X)

        fit("lstm", make_lstm)

    def tabular(name, build):
        def make():
            m = build().fit(tab_tr, y_tr)
            return lambda ws, tab: m.predict_proba(tab)

        fit(name, make)

    tabular("random_forest", lambda: random_forest(seed))
    tabular("xgboost", lambda: xgboost_model(y_tr, seed))
    tabular("gradient_boosting", HourlyGBM)

    def raw(name: str, ws, tab) -> np.ndarray:
        return np.asarray(predictors[name](ws, tab), dtype=float)

    # ---- calibration and thresholds, from validation only
    val_raw = {m: raw(m, val, tab_v) for m in list(predictors)}
    for m in [m for m, p in val_raw.items() if not np.isfinite(p).all()]:
        skipped[m] = "gave non-finite probabilities on the validation windows"
        predictors.pop(m)
        val_raw.pop(m)
    platt: dict[str, dict | None] = {}
    for m in predictors:
        if m not in SELF_CALIBRATED:
            platt[m] = fit_platt(y_val, val_raw[m])
            if platt[m] is None:
                notes[m] = (
                    notes.get(m, "") + " probabilities not calibrated (too few validation flares)"
                ).strip()

    def probs_of(ws, tab, vals: dict | None = None) -> dict[str, np.ndarray]:
        r = vals if vals is not None else {m: raw(m, ws, tab) for m in predictors}
        p = {m: (apply_calibration(platt[m], r[m]) if m in platt else r[m]) for m in predictors}
        if BASE in p and "gradient_boosting" in p and stack is not None:
            p["stack"] = stack.predict(p[BASE], p["gradient_boosting"])
        return p

    stack = None
    if BASE in predictors and "gradient_boosting" in predictors:
        stack = fit_stack(y_val, val_raw[BASE], val_raw["gradient_boosting"])
        if stack is None:
            skipped["stack"] = (
                "the stacking regression gave one of the two models no positive weight"
            )
    val_p = probs_of(val, tab_v, val_raw)
    models = list(val_p)
    thr = {m: float(best_threshold(y_val, p, smooth=2)[0]) for m, p in val_p.items()}

    report: dict = {
        "run": cfg.run_name, "horizon_min": int(horizon), "seed": seed,
        "train": {"windows": int(len(y_tr)), "flare_windows": int(y_tr.sum()), "period": _period(train, bundles)},
        "validation_set": {"windows": int(len(y_val)), "flare_windows": int(y_val.sum()), "period": _period(val, bundles)},
        "features_used": [c for c in tab_tr.columns if tab_tr[c].notna().any()], "features_dropped": dropped,
        "models": {m: {"label": LABELS[m], "family": FAMILY[m], "note": notes.get(m, ""),
                       "calibrated": m in SELF_CALIBRATED or platt.get(m) is not None} for m in models},
        "skipped": skipped, "thresholds_from_validation": thr,
        "validation": {m: full_metrics(y_val, p, thr[m]) for m, p in val_p.items()},
        "datasets": {},
    }  # fmt: skip

    held = {**{f"{k} (later period of a training source)": v for k, v in splits.get("test", {}).items()},
            **dict(splits.get("eval", {}).items())}  # fmt: skip
    for name, ws in held.items():
        sub, tab = with_features(ws)
        if len(sub) == 0:
            continue
        y = np.asarray(sub.y_fc).astype(int)
        probs = probs_of(sub, tab)
        block = {
            "windows": int(len(y)), "flare_windows": int(y.sum()), "period": _period(sub, bundles),
            "metrics": {m: full_metrics(y, p, thr[m]) for m, p in probs.items()},
            "curves": {m: curves(y, p) for m, p in probs.items()},
            "paired_vs_cnn_bilstm": {},
        }  # fmt: skip
        if BASE in probs:
            for m in models:
                if m != BASE:
                    block["paired_vs_cnn_bilstm"][m] = paired_block_bootstrap(
                        y, probs[BASE], probs[m], sub.t_end, thr[BASE], thr[m], n=n_boot, seed=seed)  # fmt: skip
        report["datasets"][name] = block
        tag = _safe(name)
        pd.DataFrame({"t_end": sub.t_end, "y": y, **{f"p_{m}": p for m, p in probs.items()}}).to_csv(
            out / f"model_comparison_predictions_{tag}.csv", index=False)  # fmt: skip
        title = f"{name}: {len(y):,} windows, {int(y.sum()):,} followed by an M/X flare within {horizon // 60} h"
        plot_roc_pr(title, y, probs, SHORT, block["metrics"], out / f"roc_pr_{tag}.png")
        plot_confusions(title, block["metrics"], SHORT, out / f"confusion_{tag}.png")

    (out / "model_comparison.json").write_text(jsonsafe.dumps(report, indent=2))
    (out / "model_comparison.md").write_text(render(report))
    return report


def _f(v, d: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def render(rep: dict) -> str:
    L = ["# Model comparison: CNN-BiLSTM, LSTM, Random Forest, XGBoost, gradient boosting", ""]
    tr, va = rep["train"], rep["validation_set"]
    p = lambda d: f", {d['period']['start']} to {d['period']['end']}" if d.get("period") else ""  # noqa: E731
    L += [
        f"Target: an M or X flare starts within {rep['horizon_min'] // 60} hours of the forecast time. "
        f"Trained on {tr['windows']:,} windows ({tr['flare_windows']:,} flare-positive{p(tr)}). Probability calibration "
        f"and alert thresholds from {va['windows']:,} validation windows ({va['flare_windows']:,} flare-positive{p(va)}). "
        "Every model is scored on the same windows below; none of the test windows was used to choose anything.",
        "",
    ]
    if rep["features_dropped"]:
        L += [f"Feature table: {len(rep['features_used'])} features used, {len(rep['features_dropped'])} left out "
              f"({', '.join(rep['features_dropped'])}).", ""]  # fmt: skip
    L += ["| model | what it reads | notes |", "|---|---|---|"]
    L += [f"| {m['label']} | {m['family']} | {m['note'] or ('calibrated' if m['calibrated'] else 'uncalibrated')} |"
          for m in rep["models"].values()]  # fmt: skip
    for k, why in rep["skipped"].items():
        L.append(f"| {LABELS.get(k, k)} | | **not run:** {why} |")
    L.append("")
    for name, d in rep["datasets"].items():
        L += [f"## {name}", "",
              f"{d['windows']:,} windows{p(d)}, {d['flare_windows']:,} followed by an M/X flare "
              f"(a random ranking scores PR-AUC = {d['flare_windows'] / max(d['windows'], 1):.3f}).", "",
              "| model | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC | TSS | Brier |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for m, s in d["metrics"].items():
            L.append(f"| {SHORT[m]} | {_f(s['accuracy'])} | {_f(s['precision'])} | {_f(s['recall'])} | {_f(s['F1'])} | "
                     f"{_f(s['ROC_AUC'])} | {_f(s['PR_AUC'])} | {_f(s['TSS'])} | {_f(s['Brier'], 4)} |")  # fmt: skip
        L += ["", "Confusion matrix at each model's validation-chosen threshold:", "",
              "| model | threshold | quiet, no alert (TN) | quiet, alert (FP) | flare, no alert (FN) | flare, alert (TP) |",
              "|---|---|---|---|---|---|"]  # fmt: skip
        for m, s in d["metrics"].items():
            (tn, fp), (fn, tp) = s["confusion"]
            L.append(f"| {SHORT[m]} | {s['threshold']:.2f} | {tn:,} | {fp:,} | {fn:,} | {tp:,} |")
        if d["paired_vs_cnn_bilstm"]:
            L += ["", "Difference from the CNN-BiLSTM on the same windows (block bootstrap, 95 % interval):", "",
                  "| model | metric | difference | 95 % interval | reading |", "|---|---|---|---|---|"]  # fmt: skip
            for m, res in d["paired_vs_cnn_bilstm"].items():
                for k in ("AUC", "TSS", "Brier"):
                    r = res[k]
                    lo, hi = r["ci95"]
                    verdict = (
                        reading(k, r)
                        .replace("B better", f"{SHORT[m]} better")
                        .replace("B worse", f"{SHORT[m]} worse")
                    )
                    L.append(
                        f"| {SHORT[m]} | {k} | {r['diff']:+.4f} | {_f(lo, 4)} to {_f(hi, 4)} | {verdict} |"
                    )
        tag = _safe(name)
        L += ["", f"Figures: `roc_pr_{tag}.png`, `confusion_{tag}.png`. Per-window probabilities: "
              f"`model_comparison_predictions_{tag}.csv`.", ""]  # fmt: skip
    L += ["## How to read this", "",
          "- **Accuracy is not the headline.** Most windows are quiet, so a model that never alerts has a high accuracy. "
          "Use TSS, ROC-AUC, PR-AUC against the random-ranking value above, and the confusion matrix.",
          "- A difference counts only when its interval excludes zero. The intervals cover the sampling of the test windows, "
          "not the seed-to-seed variation of training (one seed here), and the test period is one satellite.",
          "- Thresholds come from validation windows (TSS-optimal), so models are compared at their own best operating point, "
          "not at a common 0.5.",
          "- The tree models read days of X-ray history; the two sequence models read 90 minutes. A difference mixes the "
          "model type with the information it sees, which is why the gradient-boosting row and the stack are shown.",
          ""]  # fmt: skip
    return "\n".join(L)

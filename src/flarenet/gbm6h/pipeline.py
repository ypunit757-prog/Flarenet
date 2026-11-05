"""Fit the tree model, stack it with a trained network, and score all three on held-out windows.

Nothing here touches the network's weights. Rules that keep the comparison honest:

* the tree model is fitted on TRAINING windows only;
* the stacking weights and every alert threshold are chosen on VALIDATION windows only;
* the ``eval`` sources (e.g. GOES-18) are only scored, never used for any choice;
* the stack is compared with the network alone on identical windows with a paired block bootstrap.
"""

from __future__ import annotations

import fnmatch
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.utils import get_logger
from flarenet.gbm6h.features import set_features
from flarenet.gbm6h.live import file_sha256, package
from flarenet.gbm6h.model import HourlyGBM
from flarenet.gbm6h.stack import fit_stack
from flarenet.pipelines.evaluation.calibration import alert_levels
from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc, score_at_threshold
from flarenet.pipelines.evaluation.paired import paired_block_bootstrap, reading
from flarenet.pipelines.evaluation.pipeline import align_channels
from flarenet.pipelines.evaluation.scoring import infer
from flarenet.pipelines.evaluation.uncertainty import calibration_summary

log = get_logger(__name__)
MODELS = ("cnn", "gbm", "stack")
# What a forecast made from the live NOAA feed can compute: that feed reaches back 7 days, so no 27-day rotation
# features, and the 168-hour windows would be cut short at the very edge of the feed.
LIVE_COMPATIBLE_DROP = ("rot_*", "*168*")


def drop_features(table: pd.DataFrame, patterns) -> tuple[pd.DataFrame, list[str]]:
    """Remove the columns whose name matches any shell-style pattern (``rot_*``, ``*168*``); returns the rest and what went."""
    gone = [c for c in table.columns if any(fnmatch.fnmatch(c, p) for p in patterns)]
    return table.drop(columns=gone), gone


def scored(ws, bundles, lm, gbm: HourlyGBM, device, horizon_min: int, stride_min: int = 0):
    """The windows of ``ws`` that have features, with the network's and the tree model's probabilities."""
    tab, rows = set_features(ws, bundles, horizon_min, stride_min)
    sub = ws.subset(rows)
    _, _, p_cnn = infer(lm.model, align_channels(sub, lm.channels).X, lm.mu, lm.sg, device,
                        calibration=lm.calibration)  # fmt: skip
    p_gbm = gbm.predict_proba(tab) if gbm.cols else np.full(len(sub), np.nan)
    return sub, tab, np.asarray(p_cnn, dtype=float), np.asarray(p_gbm, dtype=float)


def _metrics(y, p, thr: float) -> dict:
    s = score_at_threshold(y, p, thr)
    cal = calibration_summary(y, p)
    return {
        "AUC": safe_auc(y, p), "Brier": cal.get("brier", float("nan")), "ECE": cal.get("ece", float("nan")),
        "TSS": float(s["TSS"]), "POD": float(s["POD"]), "FAR": float(s["FAR"]), "threshold": float(thr),
    }  # fmt: skip


def fit_and_evaluate(
    cfg, splits: dict, bundles: dict, lm, device, out: Path, train_stride: int = 60, n_boot: int = 1000,
    seed: int = 0, drop: tuple[str, ...] = (), checkpoint_path: Path | None = None,
) -> dict:  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    horizon = cfg.data.horizon_min

    t = time.time()
    tab, rows = set_features(splits["train"], bundles, horizon, train_stride)
    y_tr = np.asarray(splits["train"].y_fc)[rows].astype(int)
    tab, dropped = drop_features(tab, drop)
    log.info("features for %d training windows in %.1f min", len(rows), (time.time() - t) / 60)
    t = time.time()
    gbm = HourlyGBM().fit(tab, y_tr)
    log.info("tree model fitted on %d windows (%d flare-positive) in %.1f min",
             len(y_tr), int(y_tr.sum()), (time.time() - t) / 60)  # fmt: skip

    v_ws, v_tab, v_cnn, v_gbm = scored(splits["val"], bundles, lm, gbm, device, horizon)
    y_val = np.asarray(v_ws.y_fc).astype(int)
    stack = fit_stack(y_val, v_cnn, v_gbm)
    val_p = {"cnn": v_cnn, "gbm": v_gbm}
    if stack is not None:
        val_p["stack"] = stack.predict(v_cnn, v_gbm)
    thr = {k: float(best_threshold(y_val, p, smooth=2)[0]) for k, p in val_p.items()}
    levels = alert_levels(y_val, val_p["stack"], smooth=2) if stack is not None else {}
    report: dict = {
        "checkpoint_run": cfg.run_name, "train_windows": int(len(y_tr)), "val_windows": int(len(y_val)),
        "features_used": list(gbm.cols), "features_dropped": dropped, "alert_levels_validation": levels,
        "stack": stack.to_dict() if stack else None,
        "stack_gbm_share": stack.share_gbm() if stack else None,
        "thresholds_from_validation": thr,
        "validation": {k: _metrics(y_val, p, thr[k]) for k, p in val_p.items()},
        "datasets": {},
    }  # fmt: skip

    held = {**{f"{k} (later period of a training source)": v for k, v in splits.get("test", {}).items()},
            **{k: v for k, v in splits.get("eval", {}).items()}}  # fmt: skip
    for name, ws in held.items():
        sub, _, pc, pg = scored(ws, bundles, lm, gbm, device, horizon)
        y = np.asarray(sub.y_fc).astype(int)
        probs = {"cnn": pc, "gbm": pg}
        if stack is not None:
            probs["stack"] = stack.predict(pc, pg)
        block = {"windows": int(len(y)), "flare_windows": int(y.sum()),
                 "metrics": {k: _metrics(y, p, thr[k]) for k, p in probs.items()}}  # fmt: skip
        for other in ("stack", "gbm"):
            if other in probs:
                block[f"paired_{other}_minus_cnn"] = paired_block_bootstrap(
                    y,
                    probs["cnn"],
                    probs[other],
                    sub.t_end,
                    thr["cnn"],
                    thr[other],
                    n=n_boot,
                    seed=seed,
                )
        report["datasets"][name] = block
        pd.DataFrame(
            {"t_end": sub.t_end, "y": y, **{f"p_{k}": p for k, p in probs.items()}}
        ).to_csv(out / f"gbm6h_predictions_{name.split(' ')[0]}.csv", index=False)

    # a few validation rows and the probabilities they gave now: a machine that gives other numbers is refused later
    pick = np.linspace(0, len(v_gbm) - 1, min(8, len(v_gbm))).astype(int)
    check = {"X": v_tab.iloc[pick].reset_index(drop=True), "p_gbm": v_gbm[pick]}
    sha = file_sha256(checkpoint_path) if checkpoint_path else None
    pkg = package(gbm, stack, thr, levels, horizon, sha, check, dropped)
    with open(out / "gbm6h.pkl", "wb") as f:
        pickle.dump(pkg, f)
    (out / "gbm6h_report.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "gbm6h_report.md").write_text(render(report))
    return report


def _f(v, d=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def render(rep: dict) -> str:
    L = ["# Tree model and stack against the network alone", ""]
    if rep.get("features_dropped"):
        L += [f"Features left out ({len(rep['features_dropped'])}): {', '.join(rep['features_dropped'])}. "
              f"{len(rep['features_used'])} features used.", ""]  # fmt: skip
    if rep["stack"] is None:
        L += ["The stacking regression gave the tree model no positive weight (or there were too few flare windows in "
              "validation): **the tree model adds nothing to this network here; keep the network alone.**", ""]  # fmt: skip
    else:
        s = rep["stack"]
        L += [f"Stack = logistic regression on the logits of the two probabilities, fitted on {rep['val_windows']} "
              f"validation windows: weight on the network {s['a_cnn']:.2f}, on the tree model {s['a_gbm']:.2f} "
              f"({100 * rep['stack_gbm_share']:.0f} % of the weight on the trees).", ""]  # fmt: skip
    for name, d in rep["datasets"].items():
        L += [f"## {name}: {d['windows']} windows, {d['flare_windows']} followed by an M/X flare", "",
              "| model | AUC | Brier | ECE | TSS | POD | FAR |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for k, m in d["metrics"].items():
            L.append(f"| {k} | {_f(m['AUC'])} | {_f(m['Brier'], 4)} | {_f(m['ECE'], 4)} | {_f(m['TSS'])} | "
                     f"{_f(m['POD'])} | {_f(m['FAR'])} |")  # fmt: skip
        for other in ("stack", "gbm"):
            res = d.get(f"paired_{other}_minus_cnn")
            if not res:
                continue
            L += ["", f"{other} minus network on the same windows (block bootstrap, 95 % interval):", "",
                  "| metric | difference | 95 % interval | reading |", "|---|---|---|---|"]  # fmt: skip
            for k, r in res.items():
                lo, hi = r["ci95"]
                L.append(
                    f"| {k} | {r['diff']:+.4f} | {_f(lo, 4)} to {_f(hi, 4)} | {reading(k, r)} |"
                )
        L.append("")
    L += ["Thresholds, stacking weights and the tree model's calibration come from training and validation windows "
          "only. A gain counts only where the interval of the difference excludes zero on the dataset neither model "
          "saw (an `eval` source); numbers on the later part of a training source are a weaker check.", ""]  # fmt: skip
    return "\n".join(L)

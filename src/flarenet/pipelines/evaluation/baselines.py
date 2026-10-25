"""Reference forecasters every learned model should be compared with.

A skill number means little without them: on a rare-event problem even a "predict the long-run
rate" forecaster scores a respectable Brier value, and a one-line flux rule is often a hard
baseline for short horizons. All baselines see only information available at issue time and are
fitted on the *training* sources' windows (never the test windows).

* ``climatology``      constant probability = training base rate (TSS 0 by construction)
* ``flux_persistence`` score = highest soft X-ray level in the last 6 h; threshold tuned on the
                       validation windows (same protocol as the model's alert threshold)
* ``nowcast_flux_rule`` current class read straight from the flux with the GOES thresholds
                       (only meaningful for GOES-calibrated sources)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from flarenet.core.constants import CLASS_THRESHOLDS
from flarenet.core.schema import LightcurveBundle, WindowSet
from flarenet.pipelines.evaluation.metrics import contingency, safe_auc, score_at_threshold, tss
from flarenet.pipelines.features.channels import fill_gaps, log10_clip, rolling_max


def flux_features(
    ws: WindowSet, bundles: dict[str, LightcurveBundle]
) -> tuple[np.ndarray, np.ndarray]:
    """(current log10 soft flux, 6 h running max of it) at the last minute of every window."""
    cur = np.zeros(len(ws))
    m6 = np.zeros(len(ws))
    for k, name in enumerate(ws.source_names):
        sel = ws.source_idx == k
        if not sel.any() or name not in bundles:
            cur[sel] = m6[sel] = np.nan
            continue
        ls = log10_clip(fill_gaps(bundles[name].df["sxr"].to_numpy(dtype=float)))
        r6 = rolling_max(ls, 360)
        idx = np.clip(np.rint(ws.t_end[sel]).astype(int), 0, len(ls) - 1)
        cur[sel], m6[sel] = ls[idx], r6[idx]
    return cur, m6


def _tune_threshold(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan")
    grid = np.unique(np.quantile(score[ok], np.linspace(0.02, 0.98, 97)))
    best = max(grid, key=lambda t: tss(*contingency(y[ok], (score[ok] >= t).astype(int))))
    return float(best)


def fit_baselines(train: WindowSet, val: WindowSet, bundles: dict) -> dict:
    """Parameters of the baselines, from training / validation windows only."""
    _, m6_val = flux_features(val, bundles)
    return {
        "base_rate": float(np.mean(train.y_fc)),
        "flux_threshold": _tune_threshold(np.asarray(val.y_fc), m6_val),
    }


def nowcast_from_flux(cur_log: np.ndarray) -> np.ndarray:
    """GOES class index (0 QUIET .. 4 X) from log10 flux."""
    return np.searchsorted(np.log10(CLASS_THRESHOLDS), cur_log, side="right").astype(np.int64)


def score_baselines(
    ws: WindowSet, bundles: dict, params: dict, model_brier: float | None = None
) -> dict:
    """Baseline scores on one window set (same windows the model is scored on)."""
    y = np.asarray(ws.y_fc)
    cur, m6 = flux_features(ws, bundles)
    p0 = params["base_rate"]
    brier_clim = float(np.mean((y - p0) ** 2))
    out: dict = {
        "climatology": {"TSS": 0.0, "Brier": brier_clim, "base_rate": p0},
    }
    thr = params["flux_threshold"]
    if np.isfinite(thr) and np.isfinite(m6).all():
        s = score_at_threshold(y, (m6 >= thr).astype(float), 0.5)
        out["flux_persistence"] = {
            "TSS": s["TSS"], "HSS": s["HSS"], "POD": s["POD"], "FAR": s["FAR"],
            "AUC": safe_auc(y, m6), "threshold_log10_flux": thr,
        }  # fmt: skip
    if np.isfinite(cur).all():
        pred = nowcast_from_flux(cur)
        out["nowcast_flux_rule"] = {
            "accuracy": float((pred == ws.y_now).mean()),
            "macro_F1": float(f1_score(ws.y_now, pred, average="macro")),
        }
    if model_brier is not None and brier_clim > 0:
        out["model_brier_skill_vs_climatology"] = float(1.0 - model_brier / brier_clim)
    return out

"""The explanation block of the live forecast (``explain`` in forecast.json).

For the newest forecast it answers: how much of today's probability comes from the neural network and how much from the tree
model, and which groups of inputs raised or lowered each of them compared with a typical window. The probability it explains
is the one on the page; the contributions add up to it exactly (in log-odds).

What it does not say: that a group *caused* a flare. A contribution is how much the model's own number changes when that
group of inputs takes its real value instead of a typical one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.goes import flux_to_class_type
from flarenet.explain.groups import channel_groups, tree_groups
from flarenet.explain.local import logit, network_effects, tree_effects

METHOD = "Shapley values over groups of inputs, compared with a typical window"
BASE_NOTE = (
    "The reference is a typical window: the training-period average for the network's channels and the median of validation "
    "windows for the tree model's features. A contribution is how far the forecast moves, in log-odds, because a group has its "
    "real value instead of the typical one. Contributions describe the model's reasoning, not the physics of the Sun, and "
    "the forecast remains a probability, not a prediction that a flare will occur."
)


def _sig(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def describe(gid: str, row: pd.Series) -> str | None:
    """A plain-words value for a tree-model group, from the feature row (None when it cannot be told)."""

    def v(name):
        x = row.get(name, np.nan)
        return None if pd.isna(x) else float(x)

    if gid == "level_now" and v("f_cur") is not None:
        return f"{flux_to_class_type(10 ** v('f_cur'))} now"
    if gid == "flares_hours" and v("h_nC_6") is not None:
        return f"{int(v('h_nC_6'))} C-class or stronger and {int(v('h_nM_6') or 0)} M-class or stronger in the last 6 h"
    if gid == "flares_day" and v("h_nC_24") is not None:
        return f"{int(v('h_nC_24'))} C-class or stronger and {int(v('h_nM_24') or 0)} M-class or stronger in the last 24 h"
    if gid == "flares_3days" and v("h_nC_72") is not None:
        return f"{int(v('h_nC_72'))} C-class or stronger and {int(v('h_nM_72') or 0)} M-class or stronger in the last 72 h"
    if gid == "since_flare" and v("h_since_M") is not None:
        h = v("h_since_M")
        return (
            "no M-class flare in the history used"
            if h >= 720
            else f"last M-class flare {h:.0f} h ago"
        )
    if gid == "peak_hours" and v("f_max_6h") is not None:
        return f"highest flux in the last 6 h: {flux_to_class_type(10 ** v('f_max_6h'))}"
    if gid == "activity_days" and v("f_max_72h") is not None:
        return f"highest flux in the last 3 days: {flux_to_class_type(10 ** v('f_max_72h'))}"
    if gid == "coverage" and v("f_cov_24h") is not None:
        return f"{100 * v('f_cov_24h'):.0f}% of the last 24 h present"
    return None


def _top(
    effects: dict, labels: dict, scale: float, details: dict | None = None, k: int = 8
) -> list[dict]:
    rows = [
        {
            "id": g,
            "label": labels[g],
            "effect": round(scale * e, 3),
            **({"detail": details[g]} if details and details.get(g) else {}),
        }
        for g, e in effects.items()
    ]
    rows.sort(key=lambda r: -abs(r["effect"]))
    return rows[:k]


def explain_latest(
    network_proba, window: np.ndarray, reference_window: np.ndarray, channels: list[str], time_iso: str,
    p_final: float, live=None, feature_row: pd.Series | None = None,
) -> dict:  # fmt: skip
    """Explanation of one forecast.

    ``network_proba`` maps raw windows (n, channels, minutes) to calibrated probabilities; ``window`` is the newest
    window and ``reference_window`` the typical one. ``live`` (a ``LiveStack``) and ``feature_row`` add the tree model and
    the stack; without them the forecast is the network's alone."""
    cg = channel_groups(channels)
    net = network_effects(network_proba, window, reference_window, cg)
    block: dict = {
        "method": METHOD,
        "time": time_iso,
        "p": round(float(p_final), 4),
        "note": BASE_NOTE,
    }
    if live is None or feature_row is None or getattr(live, "reference", None) is None:
        block["p_base"] = round(net["p_ref"], 4)
        block["models"] = [
            {
                "id": "network",
                "label": "Neural network (last 90 minutes)",
                "p": round(net["p_row"], 4),
                "weight": 1.0,
            }
        ]
        block["groups"] = [
            {**g, "model": "network"}
            for g in _top(net["effects"], {k: v[0] for k, v in cg.items()}, 1.0)
        ]
        block["consistent"] = abs(net["p_row"] - p_final) < 1e-3
        return block

    tg = tree_groups(live.gbm.cols)
    tre = tree_effects(
        live.gbm.predict_proba,
        feature_row.reindex(live.gbm.cols),
        live.reference.reindex(live.gbm.cols),
        tg,
    )
    a, g, b = live.stack.a_cnn, live.stack.a_gbm, live.stack.b
    z_base = a * logit(net["p_ref"]) + g * logit(tre["p_ref"]) + b
    z_now = a * logit(net["p_row"]) + g * logit(tre["p_row"]) + b
    details = {gid: describe(gid, feature_row) for gid in tg}
    groups = [
        {**r, "model": "network"} for r in _top(net["effects"], {k: v[0] for k, v in cg.items()}, a)
    ]
    groups += [
        {**r, "model": "trees"}
        for r in _top(tre["effects"], {k: v[0] for k, v in tg.items()}, g, details)
    ]
    groups.sort(key=lambda r: -abs(r["effect"]))
    block.update(
        p_base=round(_sig(float(z_base)), 4),
        models=[
            {
                "id": "network",
                "label": "Neural network (last 90 minutes)",
                "p": round(net["p_row"], 4),
                "weight": round(a, 3),
            },
            {
                "id": "trees",
                "label": "Tree model (days of history)",
                "p": round(tre["p_row"], 4),
                "weight": round(g, 3),
            },
        ],
        groups=groups[:12],
        consistent=bool(abs(_sig(float(z_now)) - p_final) < 1e-3),
    )
    return block


def explain_for_export(lm, device, bundle, fc: pd.DataFrame, live=None) -> dict:
    """The ``explain`` block for the newest row of ``fc`` (the table ``scripts/export_web.py`` builds).

    ``live`` is a loaded ``LiveStack`` or None; the tree model is explained only when it was used for the newest row."""
    from flarenet.gbm6h.features import prepare, window_features
    from flarenet.pipelines.evaluation.scoring import infer
    from flarenet.pipelines.features.windows import build_windows
    from flarenet.pipelines.inference.predict import feature_kwargs

    d = lm.config["data"]
    ws = build_windows(
        bundle.df, d["window_min"], d["stride_min"], d["horizon_min"], for_inference=True,
        channel_names=lm.channels, max_missing_frac=d.get("max_missing_frac", 1.0), **feature_kwargs(lm.config),
    ).subset(slice(-1, None))  # fmt: skip
    last = fc.iloc[-1]
    if abs(float(ws.t_end[-1]) - float(last["time_min"])) > 1e-6:
        raise ValueError("the newest window does not match the newest forecast row")
    reference = np.repeat(np.asarray(lm.mu), ws.X.shape[2], axis=2)[0]

    def proba(w: np.ndarray) -> np.ndarray:
        return infer(lm.model, w, lm.mu, lm.sg, device, calibration=lm.calibration)[2]

    row = None
    if live is not None and bool(last.get("stack_used", False)):
        lf, ev = prepare(bundle)
        row = pd.Series(
            window_features(
                int(round(float(last["time_min"]))), lf, ev, bundle.t0_unix, live.horizon_min
            )
        )
    else:
        live = None
    return explain_latest(
        proba,
        ws.X[0],
        reference,
        list(lm.channels),
        str(last["time_utc"]) + "Z",
        float(last["p_mx_6h"]),
        live,
        row,
    )

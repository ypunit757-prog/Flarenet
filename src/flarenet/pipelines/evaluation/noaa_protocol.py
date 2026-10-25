"""Daily 24 h "will an M-class flare occur" forecasts, scored the way NOAA/SWPC forecasts are.

NOAA SWPC issues one probability per UTC day for the next 24 h. To compare a FlareNet model with
that, the model must be trained with ``data.horizon_min: 1440`` and is then asked once per day at
00:00 UTC (using only the 90 minutes before). The label is "an M/X flare starts within the next
24 h". NOAA's own verification labels are "an M-class flare happened on the day", which is the
same event up to flares that straddle midnight.

Published NOAA reference (arXiv:2508.01114, daily M-class, theta = 0.5, a different period and
catalogue than yours): TSS 0.44, HSS 0.47, AUC 0.87, Brier 0.11. Those numbers are context only;
the fair comparison is against NOAA's probabilities for *your* days (``noaa_csv``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc, score_at_threshold
from flarenet.pipelines.evaluation.uncertainty import block_bootstrap_ci  # noqa: F401

NOAA_PUBLISHED_M = {
    "TSS": 0.44,
    "HSS": 0.47,
    "AUC": 0.87,
    "Brier": 0.11,
    "source": "arXiv:2508.01114",
}


def midnight_starts(t0_unix: float, n_minutes: int, window_min: int) -> np.ndarray:
    """Start indices of the windows whose last sample is the 00:00 UTC minute of a day."""
    first = int(np.ceil(t0_unix / 86400.0) * 86400 - t0_unix) // 60  # index of the first 00:00
    mids = np.arange(first, n_minutes, 1440)
    return mids - (window_min - 1)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def daily_scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    s = score_at_threshold(y, p, threshold)
    base = float(np.mean(y)) if len(y) else float("nan")
    b = brier(y, p)
    return {
        "n_days": int(len(y)), "positive_days": int(np.sum(y)), "threshold": float(threshold),
        "TSS": s["TSS"], "HSS": s["HSS"], "POD": s["POD"], "FAR": s["FAR"],
        "AUC": safe_auc(y, p), "Brier": b,
        "Brier_skill_vs_climatology": 1.0 - b / (base * (1 - base)) if 0 < base < 1 else float("nan"),
    }  # fmt: skip


def read_noaa_csv(path: str) -> pd.DataFrame:
    """CSV with ``date`` (YYYY-MM-DD) and ``prob_m`` (0-1 or percent) -> Series indexed by date."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dc = cols.get("date") or cols.get("time")
    pc = next((cols[k] for k in ("prob_m", "m", "m_prob", "probability") if k in cols), None)
    if dc is None or pc is None:
        raise ValueError(f"{path}: need columns 'date' and 'prob_m', found {list(df.columns)}")
    p = df[pc].to_numpy(dtype=float)
    if np.nanmax(p) > 1.0:
        p = p / 100.0
    idx = pd.to_datetime(df[dc], utc=True).dt.floor("D")
    return pd.Series(p, index=idx).groupby(level=0).last()


def paired_bootstrap_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, thr_a: float, thr_b: float,
    metric: str = "TSS", block: int = 7, n: int = 1000, seed: int = 0,
) -> tuple[float, float, float]:  # fmt: skip
    """Mean and 95 % interval of ``metric(A) - metric(B)`` resampling blocks of days together."""
    rng = np.random.default_rng(seed)
    m = len(y)
    nb = max(1, int(np.ceil(m / block)))
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, max(m - block + 1, 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, m)) for s in starts])[:m]
        ya = y[idx]
        a = score_at_threshold(ya, p_a[idx], thr_a)[metric]
        b = score_at_threshold(ya, p_b[idx], thr_b)[metric]
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def compare(y, p_model, model_threshold, noaa: np.ndarray | None = None) -> dict:
    """Model (at theta=0.5 and at its validated threshold) vs NOAA probabilities, same days."""
    out = {
        "model_theta_0.5": daily_scores(y, p_model, 0.5),
        "model_validated_threshold": daily_scores(y, p_model, model_threshold),
        "model_oracle_TSS": best_threshold(y, p_model)[1],
        "noaa_published_reference": NOAA_PUBLISHED_M,
    }
    if noaa is not None:
        out["noaa_same_days_theta_0.5"] = daily_scores(y, noaa, 0.5)
        out["noaa_oracle_TSS"] = best_threshold(y, noaa)[1]
        for m in ("TSS", "HSS"):
            d, lo, hi = paired_bootstrap_diff(y, p_model, noaa, 0.5, 0.5, m)
            out[f"{m}_diff_model_minus_noaa_theta_0.5"] = {"mean": d, "ci95": [lo, hi]}
    return out


def format_markdown(res: dict, source: str, caveats: list[str]) -> str:
    rows = [("FlareNet (theta 0.5)", res["model_theta_0.5"]),
            ("FlareNet (validated threshold)", res["model_validated_threshold"])]  # fmt: skip
    if "noaa_same_days_theta_0.5" in res:
        rows.append(("NOAA probabilities, same days (theta 0.5)", res["noaa_same_days_theta_0.5"]))
    ref = res["noaa_published_reference"]
    f = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
    lines = [f"# 24 h M/X daily forecast, NOAA-style protocol: `{source}`", "",
             "| forecaster | days | M/X-days | TSS | HSS | POD | FAR | AUC | Brier | Brier skill |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in rows:
        lines.append(f"| {name} | {r['n_days']} | {r['positive_days']} | {f(r['TSS'])} | {f(r['HSS'])} | "
                     f"{f(r['POD'])} | {f(r['FAR'])} | {f(r['AUC'])} | {f(r['Brier'])} | "
                     f"{f(r['Brier_skill_vs_climatology'])} |")  # fmt: skip
    lines.append(f"| NOAA published ({ref['source']}, other period) | - | - | {ref['TSS']} | {ref['HSS']} | - | - | "
                 f"{ref['AUC']} | {ref['Brier']} | - |")  # fmt: skip
    for k in ("TSS_diff_model_minus_noaa_theta_0.5", "HSS_diff_model_minus_noaa_theta_0.5"):
        if k in res:
            d = res[k]
            lines += ["", f"{k.split('_diff')[0]} difference, FlareNet minus NOAA (same days, theta 0.5): "
                      f"{d['mean']:+.3f} (95 % interval {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}; a block "
                      "bootstrap over weeks of days. If the interval contains 0 the difference is not established.)"]  # fmt: skip
    lines += ["", "## Read before quoting", ""] + [f"- {c}" for c in caveats]
    return "\n".join(lines) + "\n"

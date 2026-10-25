"""Is a pooled win over NOAA consistent over time, or carried by one stretch?

A pooled test over several years can pass because one period is very different from the rest (here: the
two years after the activity peak). This module re-asks the question in ways that cannot be carried by one
period or by year-to-year changes in how active the Sun is:

- AUC *within* each year (weighted by flare days), so activity level differences between years drop out;
- paired differences year by year;
- the first and second half of the test period (split so each half holds half the flare days);
- NOAA recalibrated on a rolling year, a stronger baseline than one calibration for all history.

The verdict is deliberately strict: "consistent win" needs all of them to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flarenet.daily import evaluate as ev
from flarenet.daily.model import _logit

NAN = float("nan")


def _auc_fn(y, p):
    return lambda i: ev.safe_auc(y[i], p[i])


def _brier_fn(y, p):
    return lambda i: ev.brier(y[i], p[i])


def _within_year_auc(
    y: np.ndarray, p: np.ndarray, years: np.ndarray, idx: np.ndarray, min_events: int
) -> float:
    """Mean of per-year AUC weighted by flare days, over years that have at least ``min_events`` of them."""
    num = den = 0.0
    for yv in np.unique(years[idx]):
        m = idx[years[idx] == yv]
        e = int(y[m].sum())
        if e >= min_events and e < len(m):
            num += e * ev.safe_auc(y[m], p[m])
            den += e
    return num / den if den else NAN


def _pack(est, lo, hi, higher_is_better) -> dict:
    return {"diff": est, "ci95": [lo, hi], "verdict": ev.verdict(est, lo, hi, higher_is_better)}


def rolling_recalibration(
    d: pd.DataFrame, window_days: int = 365, step_days: int = 90, gap_days: int = 3,
    col: str = "noaa",
) -> pd.Series:  # fmt: skip
    """NOAA's probabilities Platt-scaled on only the previous ``window_days`` (refit every ``step_days``).
    NaN until a window holds enough flare and quiet days."""
    out = pd.Series(np.nan, index=d.index)
    y = d["y"].to_numpy(int)
    p = d[col].to_numpy(float)
    t = d.index[0]
    while t <= d.index[-1]:
        te = (d.index >= t) & (d.index < t + pd.Timedelta(days=step_days))
        tr = (d.index >= t - pd.Timedelta(days=window_days + gap_days)) & (
            d.index < t - pd.Timedelta(days=gap_days)
        )
        if te.any() and y[tr].sum() >= 15 and (1 - y[tr]).sum() >= 15:
            m = LogisticRegression(C=100.0, max_iter=1000).fit(_logit(p[tr])[:, None], y[tr])
            out[te] = m.predict_proba(_logit(p[te])[:, None])[:, 1]
        t += pd.Timedelta(days=step_days)
    return out


def consistency(
    d: pd.DataFrame, block: int = 14, n: int = 1000, min_year_events: int = 30, seed: int = 0,
) -> dict:  # fmt: skip
    """``d``: one row per day with y, noaa (as published), p_noaa_cal (NOAA recalibrated on earlier days) and
    p_model (FlareNet), indexed by UTC date. Returns paired differences (FlareNet minus NOAA) and a verdict."""
    s = d[d["noaa"].notna() & d["p_noaa_cal"].notna() & d["p_model"].notna()].sort_index()
    if len(s) < 60 or s["y"].sum() < 20:
        return {"available": False, "reason": f"only {len(s)} days with both forecasts"}
    y = s["y"].to_numpy(int)
    pm, pn, pc = (
        s["p_model"].to_numpy(float),
        s["noaa"].to_numpy(float),
        s["p_noaa_cal"].to_numpy(float),
    )
    years = s.index.year.to_numpy()
    allidx = np.arange(len(s))
    out: dict = {"available": True, "n_days": int(len(s)), "event_days": int(y.sum())}

    def boot(fa, fb, m):
        return ev.paired_block_bootstrap(m, fa, fb, block, n, seed)

    pooled_auc = boot(_auc_fn(y, pm), _auc_fn(y, pn), len(s))
    pooled_brier = boot(_brier_fn(y, pm), _brier_fn(y, pc), len(s))
    out["pooled"] = {
        "AUC": _pack(*pooled_auc, True),
        "Brier vs recalibrated NOAA": _pack(*pooled_brier, False),
    }

    # AUC within each year
    ye = [int(v) for v in np.unique(years) if y[years == v].sum() >= min_year_events]
    wy = boot(
        lambda i: _within_year_auc(y, pm, years, i, min_year_events),
        lambda i: _within_year_auc(y, pn, years, i, min_year_events), len(s),
    )  # fmt: skip
    out["within_year_auc"] = {**_pack(*wy, True), "years": ye}

    # year by year
    rows = []
    for yv in np.unique(years):
        m = np.where(years == yv)[0]
        e = int(y[m].sum())
        if e < min_year_events:
            continue
        ya = y[m]
        a = boot(_auc_fn(ya, pm[m]), _auc_fn(ya, pn[m]), len(m))
        b = boot(_brier_fn(ya, pm[m]), _brier_fn(ya, pc[m]), len(m))
        rows.append({"year": int(yv), "days": int(len(m)), "event_days": e, "AUC": _pack(*a, True),
                     "Brier vs recalibrated NOAA": _pack(*b, False)})  # fmt: skip
    out["years"] = rows

    # two halves holding equal numbers of flare days
    cut = int(np.searchsorted(np.cumsum(y), y.sum() / 2.0)) + 1
    halves = []
    for name, sl in (("first half", slice(0, cut)), ("second half", slice(cut, len(s)))):
        m = allidx[sl]
        ya = y[m]
        a = boot(_auc_fn(ya, pm[m]), _auc_fn(ya, pn[m]), len(m))
        b = boot(_brier_fn(ya, pm[m]), _brier_fn(ya, pc[m]), len(m))
        halves.append({"name": name, "start": str(s.index[m[0]].date()), "end": str(s.index[m[-1]].date()),
                       "days": int(len(m)), "event_days": int(ya.sum()), "AUC": _pack(*a, True),
                       "Brier vs recalibrated NOAA": _pack(*b, False)})  # fmt: skip
    out["halves"] = halves

    # a stronger NOAA baseline: recalibrated on the last year only
    roll = rolling_recalibration(s).to_numpy()
    ok = np.isfinite(roll)
    if ok.sum() >= 60 and y[ok].sum() >= 20:
        r = np.where(ok)[0]
        rb = boot(_brier_fn(y[r], pm[r]), _brier_fn(y[r], roll[r]), len(r))
        out["rolling_recalibrated_noaa"] = {"days": int(len(r)), "event_days": int(y[r].sum()),
                                            "Brier": _pack(*rb, False)}  # fmt: skip
    out["verdict"] = _verdict(out)
    return out


def _verdict(o: dict) -> dict:
    p = o["pooled"]
    auc_good = p["AUC"]["ci95"][0] > 0
    brier_good = p["Brier vs recalibrated NOAA"]["ci95"][1] < 0
    auc_bad = p["AUC"]["ci95"][1] < 0
    brier_bad = p["Brier vs recalibrated NOAA"]["ci95"][0] > 0
    few = o["event_days"] < 200
    prefix = (
        f"Only {o['event_days']} flare days; treat any difference as tentative. " if few else ""
    )
    if auc_bad and brier_bad:
        return {
            "code": "noaa_better",
            "text": prefix + "NOAA is better on both AUC and Brier score.",
        }
    if not (auc_good and brier_good):
        return {
            "code": "tie",
            "text": prefix + "No established difference from NOAA on the pooled test.",
        }
    problems = []
    wy = o["within_year_auc"]
    if not wy["ci95"][0] > 0:
        problems.append(
            "AUC within each year does not exclude 0 (the pooled AUC partly reflects year-to-year activity)"
        )
    for h in o["halves"]:
        ahead = h["AUC"]["ci95"][0] > 0 or h["Brier vs recalibrated NOAA"]["ci95"][1] < 0
        behind = h["AUC"]["ci95"][1] < 0 or h["Brier vs recalibrated NOAA"]["ci95"][0] > 0
        if behind:
            problems.append(f"NOAA is better in the {h['name']} ({h['start']} to {h['end']})")
        elif not ahead:
            problems.append(
                f"FlareNet is only level with NOAA in the {h['name']} ({h['start']} to {h['end']})"
            )
    rr = o.get("rolling_recalibrated_noaa")
    if rr is not None and not rr["Brier"]["ci95"][1] < 0:
        problems.append(
            "the Brier edge does not exclude 0 against NOAA recalibrated on a rolling year"
        )
    lost = [
        y["year"]
        for y in o["years"]
        if y["AUC"]["ci95"][1] < 0 or y["Brier vs recalibrated NOAA"]["ci95"][0] > 0
    ]
    if lost:
        problems.append("NOAA is significantly better in " + ", ".join(str(v) for v in lost))
    if not problems:
        return {"code": "consistent_win", "text": prefix + "FlareNet is ahead of NOAA on the pooled test, within years, in both halves "
                "of the period and against NOAA recalibrated on a rolling year."}  # fmt: skip
    return {
        "code": "partial_win",
        "text": prefix + "The pooled advantage is not consistent: " + "; ".join(problems) + ".",
    }


def _row(label: str, a: dict, b: dict) -> str:
    return (f"| {label} | {a['diff']:+.3f} | {a['ci95'][0]:+.3f} to {a['ci95'][1]:+.3f} | {a['verdict']} | "
            f"{b['diff']:+.4f} | {b['ci95'][0]:+.4f} to {b['ci95'][1]:+.4f} | {b['verdict']} |")  # fmt: skip


def format_consistency(o: dict) -> list[str]:
    if not o.get("available"):
        return ["## Is the edge consistent?", "", f"Not available: {o.get('reason', '')}.", ""]
    head = ("| period | AUC diff | 95% interval | reading | Brier diff vs recalibrated NOAA | 95% interval | reading |\n"
            "|---|---|---|---|---|---|---|")  # fmt: skip
    L = ["## Is the edge consistent?", "",
         f"**{o['verdict']['text']}**", "",
         "Differences are FlareNet minus NOAA (AUC: positive is better for FlareNet; Brier: negative is better). "
         "A pooled win that is carried by one stretch of time is not a general win.", "", head,
         _row("all days (pooled)", o["pooled"]["AUC"], o["pooled"]["Brier vs recalibrated NOAA"])]  # fmt: skip
    for h in o["halves"]:
        L.append(_row(f"{h['name']} {h['start']} to {h['end']} ({h['event_days']} flare days)", h["AUC"],
                      h["Brier vs recalibrated NOAA"]))  # fmt: skip
    for y in o["years"]:
        L.append(
            _row(
                f"{y['year']} ({y['event_days']} flare days)",
                y["AUC"],
                y["Brier vs recalibrated NOAA"],
            )
        )
    wy = o["within_year_auc"]
    L += ["", f"AUC computed inside each year, then averaged by flare days (years {', '.join(map(str, wy['years']))}): "
          f"{wy['diff']:+.3f}, 95% interval {wy['ci95'][0]:+.3f} to {wy['ci95'][1]:+.3f} ({wy['verdict']}). This removes "
          "the effect of years differing in how active the Sun was."]  # fmt: skip
    rr = o.get("rolling_recalibrated_noaa")
    if rr:
        b = rr["Brier"]
        L += ["", f"Against NOAA recalibrated on only the previous 365 days ({rr['days']} days, {rr['event_days']} flare days): "
              f"Brier difference {b['diff']:+.4f}, 95% interval {b['ci95'][0]:+.4f} to {b['ci95'][1]:+.4f} ({b['verdict']})."]  # fmt: skip
    L.append("")
    return L

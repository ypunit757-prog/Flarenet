"""Forward test: log tomorrow's forecasts from a frozen model, later score them on what actually happened.

Nothing here can be tuned after the fact:
- the model file is fixed (its fingerprint is logged with every row; scoring warns if it ever changes);
- a forecast row is appended once and never rewritten (``append_forecast`` refuses a second row for a day);
- outcomes are filled in later from the X-ray feed, only into rows whose outcome is still empty.

Columns of the log (one row per forecast day):
for_day, issued_utc, model_id, trained_until, target, p_flarenet, threshold, alert, p_noaa, p_noaa_recal,
threshold_noaa_recal, noaa_source, noaa_issued, regions_used, warnings, logged_utc, y, y_cov, resolved_utc
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.daily import evaluate as ev
from flarenet.daily.features import events_table

LOG_COLUMNS = [
    "for_day", "issued_utc", "model_id", "trained_until", "target", "p_flarenet", "threshold", "alert",
    "p_noaa", "p_noaa_recal", "threshold_noaa_recal", "noaa_source", "noaa_issued", "regions_used",
    "warnings", "logged_utc", "y", "y_cov", "resolved_utc",
]  # fmt: skip
TEXT_COLUMNS = ["for_day", "issued_utc", "model_id", "trained_until", "target", "noaa_source", "noaa_issued",
                "warnings", "logged_utc", "resolved_utc"]  # fmt: skip
NAN = float("nan")


def read_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(
        p, dtype={"for_day": str, "model_id": str, "noaa_source": str, "warnings": str}
    )
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in TEXT_COLUMNS:  # an all-empty text column is read as float; keep it able to hold text
        df[c] = df[c].astype(object)
    return df[LOG_COLUMNS]


def _write(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.sort_values("for_day").to_csv(tmp, index=False)
    tmp.replace(p)


def append_forecast(path: str | Path, row: dict) -> bool:
    """Add one forecast row. Returns False (and changes nothing) if that day is already logged."""
    df = read_log(path)
    if str(row["for_day"]) in set(df["for_day"].astype(str)):
        return False
    row = {
        **row,
        "logged_utc": row.get("logged_utc")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    new = pd.DataFrame([{c: row.get(c, np.nan) for c in LOG_COLUMNS}])
    _write(path, new if df.empty else pd.concat([df, new], ignore_index=True))
    return True


def resolve_outcomes(
    path: str | Path, bundle, target: str = "M", min_cov: float = 0.6, now_utc: datetime | None = None,
) -> int:  # fmt: skip
    """Fill ``y`` for logged days that are over and fully inside ``bundle``'s data (the last ~7 days of the
    live feed, so run this at least weekly). A day with too little X-ray data is left empty, not guessed.
    Returns how many rows were resolved."""
    df = read_log(path)
    if df.empty:
        return 0
    t0 = bundle.t0_unix
    n = len(bundle.df)
    t_last = t0 + (n - 1) * 60.0
    flux = (bundle.df["sxr_raw"] if "sxr_raw" in bundle.df.columns else bundle.df["sxr"]).to_numpy(
        float
    )
    ok = np.isfinite(flux) & (flux > 0)
    ev_ = events_table(bundle)
    cls_min = 3 if target.upper() == "M" else 4
    stamp = (now_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = 0
    for i, r in df.iterrows():
        if pd.notna(r["y"]):
            continue
        t_day = pd.Timestamp(str(r["for_day"]), tz="UTC").timestamp()
        if t_day < t0 or t_day + 86400.0 > t_last + 60.0:
            continue  # outside the data we hold, or the day is not over yet
        j0, j1 = int(round((t_day - t0) / 60.0)), int(round((t_day + 86400.0 - t0) / 60.0))
        cov = float(ok[j0:j1].mean())
        df.loc[i, "y_cov"] = round(cov, 3)
        if cov < min_cov:
            continue
        sel = (ev_["onset"] >= t_day) & (ev_["onset"] < t_day + 86400.0) & (ev_["cls"] >= cls_min)
        df.loc[i, "y"] = float(sel.any())
        df.loc[i, "resolved_utc"] = stamp
        done += 1
    if done or df["y_cov"].notna().any():
        _write(path, df)
    return done


def fill_noaa_from_archive(
    path: str | Path, text_dir: str | Path, workers: int = 2, recal=None
) -> int:
    """For logged days with no NOAA probability, take the one NOAA archived for that day (the same RSGA
    product the walk-forward test used); ``noaa_source`` becomes 'archive'. ``recal`` is the model file's
    ``noaa_recal`` entry; with it the frozen recalibration is applied to NOAA's number too.
    Only empty cells are filled. Returns how many NOAA probabilities were added."""
    from datetime import timedelta

    from flarenet.daily.model import apply_reference
    from flarenet.daily.noaa_text import fetch_range, load_rsga

    df = read_log(path)
    if df.empty:
        return 0
    target = str(df["target"].dropna().iloc[0]) if df["target"].notna().any() else "M"
    col = f"noaa_{target.lower()}1"
    added = 0
    miss = df[df["p_noaa"].isna()]
    if not miss.empty:
        days = [pd.Timestamp(d).date() for d in miss["for_day"]]
        files, _ = fetch_range("rsga", min(days) - timedelta(days=1), max(days), text_dir, workers)
        table, _bad = load_rsga(files)
        for i, r in miss.iterrows():
            ts = pd.Timestamp(str(r["for_day"]), tz="UTC")
            if len(table) and ts in table.index and np.isfinite(table.loc[ts, col]):
                df.loc[i, "p_noaa"] = float(table.loc[ts, col])
                df.loc[i, "noaa_source"] = "archive"
                added += 1
    if recal is not None:
        need = df["p_noaa"].notna() & df["p_noaa_recal"].isna()
        if need.any():
            p, thr = apply_reference(recal, df.loc[need, "p_noaa"].to_numpy(float))
            df.loc[need, "p_noaa_recal"] = p
            df.loc[need, "threshold_noaa_recal"] = thr
    if added or recal is not None:
        _write(path, df)
    return added


def compare_live_vs_archive(
    path: str | Path, text_dir: str | Path, workers: int = 2, tol: float = 0.005
) -> dict:
    """Check the assumption behind logging NOAA's live number: for days where both exist, the live product's
    probability should equal the one NOAA archived (the source the walk-forward test used)."""
    from datetime import timedelta

    from flarenet.daily.noaa_text import fetch_range, load_rsga

    df = read_log(path)
    live = df[df["p_noaa"].notna() & (df["noaa_source"].astype(str) != "archive")]
    out = {"compared": 0, "mismatched": 0, "max_abs_diff": 0.0}
    if live.empty:
        return out
    days = [pd.Timestamp(d).date() for d in live["for_day"]]
    files, _ = fetch_range("rsga", min(days) - timedelta(days=1), max(days), text_dir, workers)
    table, _bad = load_rsga(files)
    if table.empty:
        return out
    target = str(df["target"].dropna().iloc[0]) if df["target"].notna().any() else "M"
    col = f"noaa_{target.lower()}1"
    for _, r in live.iterrows():
        ts = pd.Timestamp(str(r["for_day"]), tz="UTC")
        if ts in table.index and np.isfinite(table.loc[ts, col]):
            d = abs(float(table.loc[ts, col]) - float(r["p_noaa"]))
            out["compared"] += 1
            out["mismatched"] += int(d > tol)
            out["max_abs_diff"] = max(out["max_abs_diff"], d)
    return out


def score_log(
    df: pd.DataFrame, block: int = 14, n: int = 1000, min_days: int = 120, min_events: int = 30,
) -> dict:  # fmt: skip
    """Score every resolved day. Paired differences use only days where NOAA's forecast was also logged."""
    r = df[df["y"].notna() & df["p_flarenet"].notna()].copy()
    # a forecast written after its own day had begun cannot count as a forecast of that day
    start = pd.to_datetime(r["for_day"], utc=True)
    logged = pd.to_datetime(r["logged_utc"], utc=True, errors="coerce")
    late = (logged.notna() & (logged >= start)).to_numpy()
    n_late = int(late.sum())
    r = r[~late].copy()
    r["for_day"] = pd.to_datetime(r["for_day"], utc=True)
    r = r.sort_values("for_day")
    out: dict = {
        "logged_days": int(len(df)),
        "resolved_days": int(len(r)),
        "unresolved_days": int(len(df) - len(r) - n_late),
        "late_logged_days_excluded": n_late,
        "models": sorted(set(df["model_id"].dropna().astype(str))),
        "first_day": None if r.empty else str(r["for_day"].iloc[0].date()),
        "last_day": None if r.empty else str(r["for_day"].iloc[-1].date()),
    }
    if not df.empty:
        days = pd.to_datetime(df["for_day"], utc=True)
        span = int((days.max() - days.min()).days) + 1
        out["calendar_gaps"] = span - int(
            days.nunique()
        )  # days between first and last with no forecast logged
    if r.empty:
        out["verdict"] = {"code": "no_data", "text": "No resolved days yet."}
        return out
    y = r["y"].to_numpy(int)
    pf = r["p_flarenet"].to_numpy(float)
    out["event_days"] = int(y.sum())
    out["flarenet"] = ev.scores(y, pf, float(r["threshold"].iloc[-1]))
    both = r[r["p_noaa"].notna()]
    out["same_days"] = {"n_days": int(len(both))}
    ok_size = len(r) >= min_days and int(y.sum()) >= min_events
    if len(both) >= 20:
        yb, pb, pn = (
            both["y"].to_numpy(int),
            both["p_flarenet"].to_numpy(float),
            both["p_noaa"].to_numpy(float),
        )
        out["same_days"]["flarenet"] = ev.scores(yb, pb, float(both["threshold"].iloc[-1]))
        out["same_days"]["noaa_as_published"] = ev.scores(yb, pn, 0.5)
        diffs = {
            "AUC": (lambda i: ev.safe_auc(yb[i], pb[i]), lambda i: ev.safe_auc(yb[i], pn[i]), True),
            "Brier vs NOAA as published": (lambda i: ev.brier(yb[i], pb[i]), lambda i: ev.brier(yb[i], pn[i]), False),
        }  # fmt: skip
        rec = both["p_noaa_recal"].to_numpy(float)
        if np.isfinite(rec).all() and len(rec):
            pc = rec
            out["same_days"]["noaa_recalibrated"] = ev.scores(
                yb, pc, float(both["threshold_noaa_recal"].iloc[-1])
            )
            diffs["Brier vs recalibrated NOAA"] = (
                lambda i: ev.brier(yb[i], pb[i]), lambda i: ev.brier(yb[i], pc[i]), False,
            )  # fmt: skip
        out["paired_differences"] = {}
        if len(both) >= 60:
            for k, (fa, fb, hib) in diffs.items():
                mean, lo, hi = ev.paired_block_bootstrap(len(yb), fa, fb, block, n)
                out["paired_differences"][k] = {
                    "diff": mean,
                    "ci95": [lo, hi],
                    "verdict": ev.verdict(mean, lo, hi, hib),
                }
        out["reliability"] = ev.reliability(yb, pb, bins=5)
    if len(set(df["model_id"].dropna().astype(str))) > 1:
        out["verdict"] = {"code": "model_changed", "text": "The log contains forecasts from more than one model; "
                          "a forward test is only valid for one frozen model. Score each model_id separately."}  # fmt: skip
    elif not ok_size:
        out["verdict"] = {"code": "too_early", "text": f"Too early to read: need at least {min_days} resolved days with at "
                          f"least {min_events} flare days; have {len(r)} and {int(y.sum())}."}  # fmt: skip
    else:
        pdiff = out.get("paired_differences", {})
        good = [k for k, v in pdiff.items() if v["verdict"].startswith("FlareNet better")]
        bad = [k for k, v in pdiff.items() if v["verdict"].startswith("NOAA better")]
        if good and not bad and "AUC" in good and any("Brier" in k for k in good):
            out["verdict"] = {
                "code": "flarenet_ahead",
                "text": "FlareNet is ahead of NOAA on AUC and Brier in the forward test.",
            }
        elif bad and not good:
            out["verdict"] = {
                "code": "noaa_ahead",
                "text": "NOAA is ahead of FlareNet in the forward test.",
            }
        else:
            out["verdict"] = {
                "code": "no_difference",
                "text": "No established difference from NOAA in the forward test.",
            }
    return out


def format_forward(res: dict) -> str:
    f = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
    L = [
        "# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made",
        "",
    ]
    L += [f"**{res['verdict']['text']}**", ""]
    L += [f"- Forecast days logged: {res['logged_days']}; resolved (outcome known): {res['resolved_days']}; "
          f"unresolved: {res['unresolved_days']}; logged after their day began (excluded): "
          f"{res['late_logged_days_excluded']}; calendar days with no forecast logged: {res.get('calendar_gaps', 0)}.",
          f"- Model fingerprints in the log: {', '.join(res['models']) or 'none'}."]  # fmt: skip
    if res.get("first_day"):
        L += [
            f"- Period: {res['first_day']} to {res['last_day']}; flare days: {res['event_days']}."
        ]
    sd = res.get("same_days", {})
    if "flarenet" in sd:
        L += ["", f"## Days with both forecasts ({sd['n_days']} days)", "",
              "| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |", "|---|---|---|---|---|"]  # fmt: skip
        for name, key in (("FlareNet", "flarenet"), ("NOAA as published (TSS at 0.5)", "noaa_as_published"),
                          ("NOAA recalibrated (frozen mapping)", "noaa_recalibrated")):  # fmt: skip
            if key in sd:
                s = sd[key]
                L.append(
                    f"| {name} | {f(s['AUC'])} | {f(s['Brier'])} | {f(s['BSS'])} | {f(s['TSS'])} |"
                )
        if res.get("paired_differences"):
            L += [
                "",
                "### Paired differences, FlareNet minus NOAA",
                "",
                "| metric | difference | 95% interval | reading |",
                "|---|---|---|---|",
            ]
            for k, v in res["paired_differences"].items():
                L.append(
                    f"| {k} | {v['diff']:+.3f} | {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} | {v['verdict']} |"
                )
        else:
            L += ["", "Paired intervals need at least 60 days with both forecasts."]
    else:
        L += [
            "",
            "Fewer than 20 resolved days have both a FlareNet and a NOAA forecast logged, so there is no comparison yet.",
        ]
    L += ["", "## Notes", "",
          "- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.",
          "- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.",
          "- Do not retrain or change the model during the test. If you do, start a new log.", ""]  # fmt: skip
    return "\n".join(L)

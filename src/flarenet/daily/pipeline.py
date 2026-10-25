"""``flarenet daily``: build day tables, run walk-forward forecasts, score against NOAA on the same days."""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.config import Config
from flarenet.core.utils import get_logger, save_json
from flarenet.daily import consistency as cs
from flarenet.daily import evaluate as ev
from flarenet.daily.features import build_day_table, feature_columns
from flarenet.daily.model import DailyModel, DailyParams, apply_reference, recalibrate_reference
from flarenet.daily.noaa_text import fetch_range, load_rsga, load_srs

log = get_logger(__name__)

GROUPS = {
    "flux only": ("f_",),
    "+ flare history": ("f_", "h_"),
    "+ rotation recurrence": ("f_", "h_", "rot_"),
    "+ sunspot regions": ("f_", "h_", "rot_", "r_"),
}


def _bundles(cfg: Config, force: bool):
    from flarenet.orchestration.runner import _context
    from flarenet.pipelines import ingestion, preprocessing

    ctx = _context(cfg, force, [s.name for s in cfg.sources])
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    return ctx, ctx.state["bundles"]


def build_table(cfg: Config, bundles: dict, text_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Day table over every source, with NOAA's own forecast attached where it exists."""
    dc = cfg.daily
    t_lo = min(b.t0_unix for b in bundles.values())
    t_hi = max(b.t0_unix + b.n_minutes * 60.0 for b in bundles.values())
    d0 = datetime.fromtimestamp(t_lo, tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(t_hi, tz=timezone.utc).date()
    info: dict = {}
    regions = None
    if dc.use_regions:
        files, st = fetch_range("srs", d0 - timedelta(days=3), d1, text_dir, dc.workers)
        regions, bad = load_srs(files)
        info["srs"] = {"files": len(files), "missing": st.missing, "unreadable": bad}
        log.info(
            "SRS: %d usable days (%d not published, %d unreadable)", len(regions), st.missing, bad
        )
    parts = []
    for name, b in bundles.items():
        t = build_day_table(b, regions, dc.issue_hour_utc, dc.min_coverage)
        t["role"] = cfg.source(name).role
        parts.append(t)
        log.info("[%s] %d usable days, %d with a %s-class flare", name, len(t), int(t["y_m" if dc.target == "M" else "y_x"].sum()), dc.target)  # fmt: skip
    tab = pd.concat(parts).sort_index()
    tab = (
        tab.sort_values("target_cov", kind="stable")
        .loc[lambda d: ~d.index.duplicated(keep="last")]
        .sort_index()
    )
    tab = tab[tab["target_cov"] >= dc.min_coverage].copy()
    tab["y"] = tab["y_m" if dc.target == "M" else "y_x"].astype(int)
    tab["noaa"] = np.nan
    if dc.compare_noaa:
        files, st = fetch_range("rsga", d0 - timedelta(days=1), d1, text_dir, dc.workers)
        noaa, bad = load_rsga(files)
        col = "noaa_m1" if dc.target == "M" else "noaa_x1"
        if len(noaa):
            tab["noaa"] = noaa[col].reindex(tab.index)
        info["rsga"] = {"files": len(files), "missing": st.missing, "unreadable": bad, "matched_days": int(tab["noaa"].notna().sum())}  # fmt: skip
        log.info("NOAA forecasts: %d of %d days matched", int(tab["noaa"].notna().sum()), len(tab))
    return tab, info


def _folds(tab: pd.DataFrame, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    dc = cfg.daily
    dates = tab.index
    if dc.mode == "train_eval":
        tr = np.where(tab["role"].to_numpy() == "train")[0]
        te = np.where(tab["role"].to_numpy() == "eval")[0]
        if len(tr) == 0 or len(te) == 0:
            raise ValueError(
                "daily.mode=train_eval needs sources with both role 'train' and role 'eval'"
            )
        return [(tr, te)]
    start = (
        pd.Timestamp(dc.first_test, tz="UTC")
        if dc.first_test
        else dates[0] + pd.Timedelta(days=dc.min_train_days)
    )
    out, t = [], start
    while t <= dates[-1]:
        te = np.where((dates >= t) & (dates < t + pd.Timedelta(days=dc.step_days)))[0]
        tr = np.where(dates < t - pd.Timedelta(days=dc.gap_days))[0]
        if len(te) and len(tr) >= 200 and tab["y"].to_numpy()[tr].sum() >= 20:
            out.append((tr, te))
        t += pd.Timedelta(days=dc.step_days)
    if not out:
        raise ValueError(
            "No walk-forward fold has enough training days; lower daily.min_train_days or add data."
        )
    return out


def _params(cfg: Config) -> DailyParams:
    dc = cfg.daily
    return DailyParams(blend_gbm=dc.blend_gbm, calibration=dc.calibration, oof_blocks=dc.oof_blocks,
                       gap_days=dc.gap_days, seed=cfg.seed)  # fmt: skip


def walk_forward(tab: pd.DataFrame, cols: list[str], cfg: Config) -> pd.DataFrame:
    """Out-of-sample probabilities: rows of ``tab`` that fall in a test block, each predicted by a model
    that never saw that block (or anything within gap_days of it)."""
    folds = _folds(tab, cfg)
    y = tab["y"].to_numpy()
    res = []
    for k, (tr, te) in enumerate(folds, 1):
        m = DailyModel(_params(cfg)).fit(tab.iloc[tr][cols], y[tr], tab.index[tr])
        p = m.predict_proba(tab.iloc[te][cols])
        ref = recalibrate_reference(tab["noaa"].to_numpy(float)[tr], y[tr])
        p_ref, thr_ref = apply_reference(ref, tab["noaa"].to_numpy(float)[te])
        res.append(
            pd.DataFrame(
                {
                    "p": p,
                    "thr": m.threshold,
                    "p_clim": float(y[tr].mean()),
                    "p_noaa_cal": p_ref,
                    "thr_noaa": thr_ref,
                },
                index=tab.index[te],
            )
        )
        log.info("fold %d/%d: train %d days, test %s..%s (%d days), threshold %.2f", k, len(folds), len(tr),
                 tab.index[te][0].date(), tab.index[te][-1].date(), len(te), m.threshold)  # fmt: skip
    return pd.concat(res)


def headline_columns(tab: pd.DataFrame, cfg: Config) -> list[str]:
    """Feature columns of the headline model (the last, largest feature group)."""
    groups = {k: v for k, v in GROUPS.items() if cfg.daily.use_regions or "r_" not in "".join(v)}
    return feature_columns(tab, groups[list(groups)[-1]])


def fit_final_models(tab: pd.DataFrame, cfg: Config, out_dir: Path) -> dict[str, Path]:
    """Fit the headline model on every labelled day and save it for live use.

    ``daily_model.pkl`` has every feature; ``daily_model_live.pkl`` leaves out the 27-day rotation features
    (a 7-day live feed cannot provide them). Both also carry NOAA's recalibration (fitted on the same days),
    so a forward test can score NOAA the way the walk-forward test did.
    """
    dc = cfg.daily
    cols = headline_columns(tab, cfg)
    y = tab["y"].to_numpy()
    ref = recalibrate_reference(tab["noaa"].to_numpy(float), y)
    out = {}
    for fname, c in (
        ("daily_model.pkl", cols),
        ("daily_model_live.pkl", [x for x in cols if not x.startswith("rot_")]),
    ):
        model = DailyModel(_params(cfg)).fit(tab[c], y, tab.index)
        path = out_dir / fname
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "columns": c,
                    "target": dc.target,
                    "issue_hour_utc": dc.issue_hour_utc,
                    "trained_days": int(len(tab)),
                    "trained_until": str(tab.index[-1].date()),
                    "noaa_recal": ref,
                },
                f,
            )
        out[fname] = path
    return out


def run_daily(cfg: Config, force: bool = False) -> dict:
    dc = cfg.daily
    ctx, bundles = _bundles(cfg, force)
    text_dir = Path(dc.text_dir) if dc.text_dir else cfg.data_dir / "downloads" / "noaa_text"
    tab, info = build_table(cfg, bundles, text_dir)
    info["synthetic"] = any(str(b.kind).startswith("synthetic") for b in bundles.values())
    out_dir = ctx.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "day_table.csv")

    groups = {k: v for k, v in GROUPS.items() if dc.use_regions or "r_" not in "".join(v)}
    headline_name = list(groups)[-1]
    sets = groups if dc.ablate else {headline_name: groups[headline_name]}
    oos: dict[str, pd.DataFrame] = {}
    for name, g in sets.items():
        cols = feature_columns(tab, g)
        log.info("=== feature set %r: %d features ===", name, len(cols))
        oos[name] = walk_forward(tab, cols, cfg)
    head = oos[headline_name]
    d = tab.loc[head.index, ["y", "noaa", "h_nM_24" if dc.target == "M" else "h_nX_24"]].copy()
    d["p_model"], d["thr"], d["p_clim"] = head["p"], head["thr"], head["p_clim"]
    d["p_noaa_cal"], d["thr_noaa"] = head["p_noaa_cal"], head["thr_noaa"]
    d["p_persist"] = (d["h_nM_24" if dc.target == "M" else "h_nX_24"] >= 1).astype(float)
    for name, o in oos.items():
        d[f"p_{name}"] = o["p"]
    d.to_csv(out_dir / "predictions.csv")

    res = score_all(d, oos, dc, headline_name)
    res["data"] = info
    res["config"] = {"target": dc.target, "mode": dc.mode, "issue_hour_utc": dc.issue_hour_utc,
                     "calibration": dc.calibration, "step_days": dc.step_days}  # fmt: skip
    save_json(res, out_dir / "daily_metrics.json")
    md = format_report(res, headline_name, dc)
    if info["synthetic"]:
        md = (
            "> **SYNTHETIC DATA: this run checks the code only. These scores say nothing about the real Sun or about NOAA.**\n\n"
            + md
        )
    (out_dir / "daily_report.md").write_text(md)

    fit_final_models(tab, cfg, out_dir)
    log.info(
        "Wrote %s, %s and the daily_model*.pkl files",
        out_dir / "daily_report.md",
        out_dir / "daily_metrics.json",
    )
    res["_markdown"] = md
    return res


def score_all(d: pd.DataFrame, oos: dict, dc, headline: str) -> dict:
    out: dict = {"n_oos_days": int(len(d)), "event_days": int(d["y"].sum()),
                 "period": [str(d.index[0].date()), str(d.index[-1].date())]}  # fmt: skip
    y = d["y"].to_numpy()
    table = {
        "climatology (train base rate)": ev.scores(y, d["p_clim"], 0.5),
        "persistence (a flare in the last 24 h)": ev.scores(y, d["p_persist"], 0.5),
    }
    for name, o in oos.items():
        s = ev.scores(y, d[f"p_{name}"], 0.5)
        s["TSS_validated_threshold"] = ev.scores(
            y, d[f"p_{name}"], o["thr"].reindex(d.index).to_numpy()
        )["TSS"]
        s["oracle_TSS"] = ev.oracle_tss(y, d[f"p_{name}"])
        table[f"FlareNet {name}"] = s
    out["all_days"] = table
    have_noaa = d["noaa"].notna()
    same = d[
        have_noaa & d["p_noaa_cal"].notna()
    ]  # days NOAA forecast AND a recalibrated NOAA exists for
    out["same_days"] = {
        "n_days": int(len(same)),
        "noaa_days_without_recalibration": int(have_noaa.sum() - len(same)),
    }
    if len(same) >= 30:
        ys, pm, pn = same["y"].to_numpy(), same["p_model"].to_numpy(), same["noaa"].to_numpy()
        pc = same["p_noaa_cal"].to_numpy()
        ts, tn = same["thr"].to_numpy(), same["thr_noaa"].to_numpy()
        noaa_row = ev.scores(ys, pn, 0.5)
        noaa_row["oracle_TSS"] = ev.oracle_tss(ys, pn)
        cal_row = ev.scores(ys, pc, tn)
        cal_row["TSS_validated_threshold"] = cal_row["TSS"]
        cal_row["oracle_TSS"] = ev.oracle_tss(ys, pc)
        st = {
            "NOAA (official forecast, as published)": noaa_row,
            "NOAA recalibrated on earlier days (same treatment as FlareNet)": cal_row,
            "climatology": ev.scores(ys, same["p_clim"], 0.5),
            "persistence": ev.scores(ys, same["p_persist"], 0.5),
        }
        for name in oos:
            s = ev.scores(ys, same[f"p_{name}"], 0.5)
            s["TSS_validated_threshold"] = ev.scores(
                ys, same[f"p_{name}"], oos[name]["thr"].reindex(same.index).to_numpy()
            )["TSS"]
            s["oracle_TSS"] = ev.oracle_tss(ys, same[f"p_{name}"])
            st[f"FlareNet {name}"] = s
        out["same_days"].update(table=st, base_rate=float(ys.mean()))
        blk, nb = dc.bootstrap_block_days, dc.bootstrap_n

        def tss_at(p, t):
            return lambda idx: ev.scores(ys[idx], p[idx], t[idx])["TSS"]

        half = np.full(len(ys), 0.5)
        diffs = {
            "AUC (threshold-free)": (
                lambda i: ev.safe_auc(ys[i], pm[i]),
                lambda i: ev.safe_auc(ys[i], pn[i]),
                True,
            ),
            "Brier vs NOAA as published (lower is better)": (
                lambda i: ev.brier(ys[i], pm[i]),
                lambda i: ev.brier(ys[i], pn[i]),
                False,
            ),
            "Brier vs recalibrated NOAA (lower is better)": (
                lambda i: ev.brier(ys[i], pm[i]),
                lambda i: ev.brier(ys[i], pc[i]),
                False,
            ),
            "TSS, each at its own validated threshold (fair)": (
                tss_at(pm, ts),
                tss_at(pc, tn),
                True,
            ),
            "TSS, both at 0.5 (NOAA rarely reaches 0.5 for M-class)": (
                tss_at(pm, half),
                tss_at(pn, half),
                True,
            ),
        }
        out["paired_differences"] = {}
        for k, (fa, fb, hib) in diffs.items():
            mean, lo, hi = ev.paired_block_bootstrap(len(ys), fa, fb, blk, nb)
            out["paired_differences"][k] = {
                "mean": mean,
                "ci95": [lo, hi],
                "verdict": ev.verdict(mean, lo, hi, hib),
            }
        out["by_year"] = ev.by_year(
            same.rename(columns={"noaa": "p_noaa"}), {"FlareNet": "p_model", "NOAA": "p_noaa"}
        )
        out["reliability"] = {"FlareNet": ev.reliability(ys, pm), "NOAA": ev.reliability(ys, pn)}
        out["consistency"] = cs.consistency(same, dc.bootstrap_block_days, dc.bootstrap_n)
    return out


def _f(v, nd=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _row(name, s):
    return (f"| {name} | {s['n_days']} | {s['event_days']} | {_f(s['TSS'])} | {_f(s.get('TSS_validated_threshold', np.nan))} | "
            f"{_f(s['HSS'])} | {_f(s['POD'])} | {_f(s['FAR'])} | {_f(s['AUC'])} | {_f(s['Brier'])} | {_f(s['BSS'])} |")  # fmt: skip


HEAD = ("| forecaster | days | event days | TSS (0.5) | TSS (validated thr.) | HSS | POD | FAR | AUC | Brier | Brier skill |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip


def format_report(res: dict, headline: str, dc) -> str:
    cls = dc.target
    L = [f"# Daily {cls}-class flare forecast vs NOAA (same days)", "",
         f"Target: an {cls}-class (or stronger) flare starts during the UTC day. Forecasts issued at {dc.issue_hour_utc:02d}:00 UTC "
         f"(NOAA's issue time) using only data before that moment. Mode: **{dc.mode}**: every day below was predicted by a model "
         "trained only on earlier days.", ""]  # fmt: skip
    sd = res["same_days"]
    if "table" in sd:
        L += [
            f"## Same days as NOAA's published forecast ({sd['n_days']} days, {sd['base_rate'] * 100:.0f}% event days)",
            "",
            HEAD,
        ]
        L += [_row(k, v) for k, v in sd["table"].items()]
        L += ["", "### Paired differences, FlareNet minus NOAA (block bootstrap over weeks of days)", "",
              "| metric | difference | 95% interval | reading |", "|---|---|---|---|"]  # fmt: skip
        for k, v in res["paired_differences"].items():
            L.append(
                f"| {k} | {v['mean']:+.3f} | {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} | {v['verdict']} |"
            )
        L += ["", "### By year (AUC / TSS at 0.5 / Brier)", "", "| year | days | event days | FlareNet AUC | NOAA AUC | FlareNet TSS | NOAA TSS | FlareNet Brier | NOAA Brier |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in res["by_year"]:
            L.append(f"| {r['year']} | {r['days']} | {r['event_days']} | {_f(r['FlareNet_AUC'])} | {_f(r['NOAA_AUC'])} | {_f(r['FlareNet_TSS'])} | {_f(r['NOAA_TSS'])} | {_f(r['FlareNet_Brier'])} | {_f(r['NOAA_Brier'])} |")  # fmt: skip
    else:
        L += ["## NOAA comparison unavailable", "", "Fewer than 30 days had both a FlareNet forecast and a NOAA forecast. "
              "Check `data.rsga` in daily_metrics.json (files downloaded vs readable).", ""]  # fmt: skip
    if "consistency" in res:
        L += cs.format_consistency(res["consistency"])
    L += [
        "",
        f"## All out-of-sample days ({res['n_oos_days']} days, {res['period'][0]} to {res['period'][1]}), with ablations",
        "",
        HEAD,
    ]
    L += [_row(k, v) for k, v in res["all_days"].items()]
    L += ["", "## Read before quoting", "",
          "- Pooled AUC mixes years: even a forecast that only knows each fold's flare rate scores about 0.73 in a rising solar cycle. "
          "Use the paired differences and the consistency section, not the absolute AUC.",
          "- Compare on the *same days* table only. Threshold-free scores (AUC, Brier) are the fairest; TSS depends on the threshold. "
          "'validated thr.' is the threshold chosen on training days only, never on the test days.",
          "- A model beats NOAA only where the paired interval above excludes 0. 'No established difference' means exactly that, "
          "not a win.",
          "- Labels come from this project's flare detection on the GOES flux, NOAA verifies against its own event list; they can "
          "differ for flares near midnight or in data gaps.",
          "- NOAA's forecast is issued at 22:00 UTC from an operational, partly human process. Check the period is long enough "
          "(hundreds of event days) before trusting any difference.",
          "- Ablation rows show what each feature group adds. Only the last row (all groups) is the headline model; do not pick the best "
          "row after seeing test results."]  # fmt: skip
    return "\n".join(L) + "\n"

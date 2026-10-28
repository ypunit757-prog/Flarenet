"""Case study: what did the model say before the biggest real flares?

For every flare of at least --min-class in a source's period, report whether the model raised an
M/X alert in the hours before the flare started (and how early), the highest forecast probability
in that window, and what the nowcast head said during the flare. Also reports how often the model
alerts overall, because a model that always alerts "catches" every flare.

    uv run python scripts/case_study.py -c configs/nasa_goes_ncei.yaml \
        --checkpoint runs/goes_active/checkpoint.pt --source goes18_autumn --min-class X

Use a source the checkpoint never trained on (role: eval). Only X-ray history is used, so a flare
that erupts from a quiet X-ray background cannot be foreseen by this model, and the flare class
reported here comes from the source's own detector (GOES-calibrated for GOES sources).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from flarenet.core.config import load_config
from flarenet.core.constants import CLASS_NAMES, CLASS_TO_IDX
from flarenet.core.utils import resolve_device
from flarenet.orchestration import runner as orch
from flarenet.pipelines import ingestion, preprocessing
from flarenet.pipelines.inference.predict import forecast_bundle
from flarenet.pipelines.training.checkpoint import load_checkpoint


def goes_label(flux: float) -> str:
    """GOES class text, e.g. 9.0e-4 W/m2 -> 'X9.0'."""
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7)):
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return f"A{flux / 1e-8:.1f}"


CASE_COLUMNS = [
    "peak_utc",
    "class",
    "forecast",
    "lead_min",
    "max_prob_before",
    "alert_at_last_issue",
    "nowcast_peak_class",
    "nowcast_ok",
]


def context_stats(
    t: np.ndarray,
    alert: np.ndarray,
    onsets: np.ndarray,
    horizon: int,
    n_shift: int = 2000,
    seed: int = 0,
) -> dict:
    """Is the warning rate better than alerting at the same rate without knowing the flares?

    * time-shift baseline: the model's own alert sequence (same rate, same episode lengths) is
      moved to random other times, and the warned count is recomputed each time;
    * precision / lift: of all alert times, how many were followed by a flare within the horizon,
      versus how often that happens at any time.
    """
    onsets = np.sort(onsets)
    lo = np.searchsorted(t, onsets - horizon, side="left")
    hi = np.searchsorted(t, onsets, side="left")
    ok = hi > lo
    lo, hi = lo[ok], hi[ok]
    a = alert.astype(np.int64)

    def warned(arr: np.ndarray) -> int:
        cs = np.concatenate([[0], np.cumsum(arr)])
        return int(((cs[hi] - cs[lo]) > 0).sum())

    obs = warned(a)
    rng = np.random.default_rng(seed)
    sims = np.array([warned(np.roll(a, int(k))) for k in rng.integers(1, len(a), n_shift)])
    j = np.searchsorted(onsets, t, side="right")
    nxt = onsets[np.minimum(j, len(onsets) - 1)]
    label = (j < len(onsets)) & (nxt <= t + horizon)
    base = float(label.mean())
    prec = float(label[alert.astype(bool)].mean()) if alert.any() else float("nan")
    return {
        "covered": int(ok.sum()),
        "observed": obs,
        "shifted_mean": float(sims.mean()),
        "shifted_p95": float(np.percentile(sims, 95)),
        "p_value": float((sims >= obs).mean()),
        "alert_rate": float(alert.mean()),
        "precision": prec,
        "base_rate": base,
        "lift": prec / base if base > 0 else float("nan"),
    }


def case_table(
    bundle, fc: pd.DataFrame, fc_now: pd.DataFrame, threshold: float, horizon: int, min_class: str
) -> pd.DataFrame:
    db = bundle.flare_db
    db = db[db["class_idx"] >= CLASS_TO_IDX[min_class]].sort_values("onset_min")
    t = fc["time_min"].to_numpy()
    p = fc["p_mx_6h"].to_numpy()
    alert = fc["alert"].to_numpy().astype(bool)
    # the nowcast is checked on a finer time grid (every few minutes), not on the hourly forecasts
    tn = fc_now["time_min"].to_numpy()
    nowc = fc_now["nowcast_class"].map({n: i for i, n in enumerate(CLASS_NAMES)}).to_numpy()
    rows = []
    for ev in db.itertuples():
        before = (t < ev.onset_min) & (t >= ev.onset_min - horizon)
        during = (tn >= ev.onset_min) & (tn <= ev.end_min)
        if before.sum() == 0:
            status, lead, pmax = "NO COVERAGE", np.nan, np.nan
        else:
            pmax = float(p[before].max())
            if alert[before].any():
                first = float(t[before][alert[before]].min())
                status, lead = "WARNED", float(ev.onset_min - first)
            else:
                status, lead = "MISSED", np.nan
        seen = int(nowc[during].max()) if during.any() else -1
        k = np.searchsorted(t, ev.onset_min, side="left") - 1
        last_on = bool(alert[k]) if k >= 0 else False
        unix = bundle.unix_of([ev.peak_min])[0]
        rows.append(
            {
                "peak_utc": pd.to_datetime(unix, unit="s", utc=True).strftime("%Y-%m-%d %H:%M"),
                "class": goes_label(ev.peak_sxr) if bundle.kind.startswith("nasa_goes") else ev._2,
                "forecast": status,
                "lead_min": lead,
                "max_prob_before": pmax,
                "alert_at_last_issue": last_on,
                "nowcast_peak_class": CLASS_NAMES[seen] if seen >= 0 else "n/a",
                "nowcast_ok": bool(seen >= ev.class_idx),
            }
        )
    # explicit columns: a quiet week with no flare of that class must give an empty table, not a table
    # without columns (the live web refresh crashed on exactly that)
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", default=None, help="default: first role=eval source")
    ap.add_argument("--min-class", default="X", choices=["C", "M", "X"])
    ap.add_argument(
        "--dates",
        default=None,
        help="only flares peaking on these UTC dates, e.g. 2024-10-01,2024-10-03",
    )
    ap.add_argument(
        "--nowcast-stride",
        type=int,
        default=5,
        help="minutes between nowcast checks (the forecast stays on the checkpoint's own stride)",
    )
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="write the table to this CSV")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    evals = [s.name for s in cfg.sources if s.role == "eval"]
    source = a.source or (evals[0] if evals else cfg.sources[0].name)
    if cfg.source(source).role == "train":
        print(
            f"WARNING: '{source}' has role train: the model saw these flares, results are in-sample.\n"
        )
    ctx = orch._context(cfg, False, [source])
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    bundle = ctx.state["bundles"][source]

    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(a.checkpoint, device)
    fc = forecast_bundle(lm, bundle, device)
    horizon = int(lm.config["data"]["horizon_min"])
    import copy

    lm_fine = copy.copy(lm)
    lm_fine.config = {**lm.config, "data": {**lm.config["data"], "stride_min": a.nowcast_stride}}
    fc_now = forecast_bundle(lm_fine, bundle, device)
    tab = case_table(bundle, fc, fc_now, lm.threshold, horizon, a.min_class)
    if a.dates:
        keep = {d.strip() for d in a.dates.split(",")}
        tab = tab[tab["peak_utc"].str[:10].isin(keep)]
    if tab.empty:
        print("No flares of that class in this source/period.")
        return 1

    show = tab.copy()
    show["lead_min"] = show["lead_min"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    show["max_prob_before"] = show["max_prob_before"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.2f}"
    )
    print(show.to_string(index=False))
    scored = tab[tab["forecast"] != "NO COVERAGE"]
    warned = (scored["forecast"] == "WARNED").sum()
    print(
        f"\n{len(tab)} flares >= {a.min_class} | forecast: warned {warned} of {len(scored)}"
        f" (alert threshold {lm.threshold:.2f}, horizon {horizon} min)"
        + (f", median lead {scored['lead_min'].median():.0f} min" if warned else "")
    )
    print(
        f"nowcast reached the flare's class at some point during the flare "
        f"(checked every {a.nowcast_stride} min): {int(tab['nowcast_ok'].sum())} of {len(tab)}"
    )
    db = bundle.flare_db
    onsets = db[db["class_idx"] >= CLASS_TO_IDX[a.min_class]]["onset_min"].to_numpy(dtype=float)
    cs = context_stats(
        fc["time_min"].to_numpy(), fc["alert"].to_numpy().astype(bool), onsets, horizon
    )
    print(
        f"\n--- is this better than chance? (alert rate {100 * cs['alert_rate']:.0f}% of forecast times)"
    )
    print(
        f"warned {cs['observed']} of {cs['covered']}; the same alerts moved to random times would warn "
        f"{cs['shifted_mean']:.1f} on average (95th percentile {cs['shifted_p95']:.0f}); "
        f"chance of doing this well by luck: p = {cs['p_value']:.2f}"
    )
    print(
        f"of all alert times, {100 * cs['precision']:.0f}% were followed by a >= {a.min_class} flare "
        f"within {horizon} min, against {100 * cs['base_rate']:.0f}% of all times "
        f"-> lift {cs['lift']:.2f}x (1.0x = no skill)"
    )
    print(
        f"alert still on at the last forecast before onset: "
        f"{int(tab['alert_at_last_issue'].sum())} of {len(tab)}"
    )
    if a.out:
        tab.to_csv(a.out, index=False)
        print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

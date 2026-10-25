"""Export model output for the web dashboard (the frontend and the API read the JSON this writes).

    uv run python scripts/export_web.py -c configs/nasa_goes_ncei.yaml \
        --checkpoint runs/goes_recent/checkpoint.pt --source goes18_autumn --days 14 \
        --metrics runs/goes_recent/metrics_report.json --out frontend/public/data.json

Runs the model over the source's last --days days (the same code `flarenet predict` uses) and writes the
flux series, the hourly forecasts and alerts, the flares of at least --flare-class with the outcome of
the forecast before each, and (optionally) the headline scores from a metrics_report.json.
For a live site, run it on a schedule against configs/live_swpc.yaml (kind: nasa_goes_swpc).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_study import case_table  # noqa: E402

from flarenet.core.config import load_config  # noqa: E402
from flarenet.core.constants import CLASS_NAMES  # noqa: E402
from flarenet.core.utils import resolve_device  # noqa: E402
from flarenet.orchestration import runner as orch  # noqa: E402
from flarenet.pipelines import ingestion, preprocessing  # noqa: E402
from flarenet.pipelines.inference.predict import forecast_bundle  # noqa: E402
from flarenet.pipelines.training.checkpoint import load_checkpoint  # noqa: E402


def _iso(unix: float) -> str:
    return datetime.fromtimestamp(float(unix), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(a, n: int) -> list:
    return [None if not np.isfinite(v) else round(float(v), n) for v in a]


def build_payload(
    bundle, lm, fc, fc_now, days: float, flare_class: str, cases: pd.DataFrame
) -> dict:
    df = bundle.df
    t_all = bundle.unix_of(df["time_min"].to_numpy())
    t_end = t_all[-1]
    lo = t_end - days * 86400.0
    keep = t_all >= lo
    # 5-minute bins: the maximum flux in each bin, so short flares stay visible
    step = 5
    flux = df["sxr_raw" if "sxr_raw" in df else "sxr"].to_numpy(dtype=float)[keep]
    tt = t_all[keep]
    n = len(flux) // step * step
    fb = np.nanmax(np.where(np.isfinite(flux[:n]), flux[:n], -np.inf).reshape(-1, step), axis=1)
    fb[~np.isfinite(fb)] = np.nan
    tb = tt[:n].reshape(-1, step)[:, 0]

    ft = bundle.unix_of(fc["time_min"].to_numpy())
    fk = ft >= lo
    nowc = fc["nowcast_class"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()

    cases = cases.copy()
    cases["peak_unix"] = (
        pd.to_datetime(cases["peak_utc"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.total_seconds()
    cases = cases[cases["peak_unix"] >= lo]
    flares = [
        {
            "peak": _iso(r.peak_unix),
            "class": str(r._2),
            "forecast": r.forecast,
            "lead_min": None if pd.isna(r.lead_min) else round(float(r.lead_min)),
            "max_prob": None if pd.isna(r.max_prob_before) else round(float(r.max_prob_before), 3),
        }
        for r in cases.itertuples()
    ]
    src = bundle.meta
    return {
        "generated": _iso(datetime.now(timezone.utc).timestamp()),
        "source": {
            "name": bundle.name,
            "kind": bundle.kind,
            "label_kind": src.get("label_kind", ""),
            "units": src.get("units", ""),
            "goes_calibrated": bundle.kind.startswith("nasa_goes"),
        },
        "model": {
            "name": lm.config.get("model", {}).get("name", "flarenet"),
            "threshold": round(float(lm.threshold), 3),
            "window_min": int(lm.config["data"]["window_min"]),
            "horizon_min": int(lm.config["data"]["horizon_min"]),
        },
        "period": {"start": _iso(tb[0]), "end": _iso(t_end), "days": round(days, 2)},
        "series": {"t": [int(x) for x in tb], "flux": _round(fb, 12), "step_min": step},
        "forecast": {
            "t": [int(x) for x in ft[fk]],
            "p": _round(fc["p_mx_6h"].to_numpy()[fk], 3),
            "alert": [int(x) for x in fc["alert"].to_numpy()[fk]],
            "nowcast": [int(x) for x in nowc[fk]],
        },
        "flare_class_min": flare_class,
        "flares": flares,
    }


def metrics_block(path: str, source: str | None) -> dict | None:
    d = json.loads(Path(path).read_text())
    res = d.get("results", {})
    key = next((k for k in res if source and k.startswith(source)), None) or next(iter(res), None)
    if key is None:
        return None
    r = res[key]
    keys = ("dataset", "n_windows", "AUC", "TSS", "HSS", "POD", "FAR", "accuracy", "macro_F1")
    out = {k: r.get(k) for k in keys}
    out.update(
        {
            "confidence_95": r.get("confidence_95"),
            "lead_time": r.get("lead_time"),
            "calibration": r.get("calibration"),
            "baselines": r.get("baselines"),
            "run": d.get("run_name"),
        }
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", default=None, help="default: first role=eval source")
    ap.add_argument(
        "--days", type=float, default=14.0, help="how many of the latest days to export"
    )
    ap.add_argument("--flare-class", default="M", choices=["C", "M", "X"])
    ap.add_argument(
        "--metrics", default=None, help="a metrics_report.json to show on the dashboard"
    )
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default="frontend/public/data.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    evals = [s.name for s in cfg.sources if s.role == "eval"]
    source = a.source or (evals[0] if evals else cfg.sources[0].name)
    ctx = orch._context(cfg, False, [source])
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    bundle = ctx.state["bundles"][source]

    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(a.checkpoint, device)
    fc = forecast_bundle(lm, bundle, device)
    import copy

    fine = copy.copy(lm)
    fine.config = {**lm.config, "data": {**lm.config["data"], "stride_min": 5}}
    fc_now = forecast_bundle(fine, bundle, device)
    horizon = int(lm.config["data"]["horizon_min"])
    cases = case_table(bundle, fc, fc_now, lm.threshold, horizon, a.flare_class)
    # class label text: GOES sources get e.g. X8.8 from the flux, others the plain class letter
    payload = build_payload(bundle, lm, fc, fc_now, a.days, a.flare_class, cases)
    if a.metrics:
        payload["metrics"] = metrics_block(a.metrics, source)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(payload['series']['t'])} flux points, "
        f"{len(payload['forecast']['t'])} forecasts, {len(payload['flares'])} flares >= {a.flare_class}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

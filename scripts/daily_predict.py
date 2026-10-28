"""Tomorrow's M-class (or X-class) flare probability from live NOAA data, next to NOAA's own number.

    uv run python scripts/daily_predict.py --model models/daily_model_live.pkl --log forward_log.csv

Run it once a day a little after 22:00 UTC (NOAA issues its forecast at 22:00 UTC). It reads the last 7 days
of GOES X-rays and the current sunspot-region summary from NOAA SWPC and uses the model that `flarenet daily`
(or scripts/daily_freeze.py) saved as `daily_model_live.pkl`.

With --log it also keeps a forward-test record (see docs/FORWARD_TEST.md): it first fills in the outcome of
earlier logged days from the X-ray feed, then appends tomorrow's forecast. A day already logged is never
changed, so the record cannot be edited after the fact. --check-noaa only shows what NOAA's live products
return (use it if the NOAA column stays empty). --merge adds the result to a forecast.json for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from flarenet.core.config import load_config
from flarenet.daily.forward import append_forecast, resolve_outcomes
from flarenet.daily.live import (
    check_noaa_live,
    feed_age_hours,
    fetch_live_srs,
    fetch_noaa_live,
    load_daily_model,
    model_id,
    noaa_prob_for_day,
    predict_tomorrow,
)
from flarenet.daily.model import apply_reference
from flarenet.orchestration import runner as orch
from flarenet.pipelines import ingestion, preprocessing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--model", help="daily_model_live.pkl written by `flarenet daily` or daily_freeze.py"
    )
    ap.add_argument("-c", "--config", default="configs/live_swpc.yaml")
    ap.add_argument("--out", default="daily_outlook.json")
    ap.add_argument("--log", default=None, help="forward-test log (CSV) to resolve and append to")
    ap.add_argument("--merge", default=None, help="a forecast.json to add the result to")
    ap.add_argument(
        "--check-noaa", action="store_true", help="show what NOAA's live products return, then stop"
    )
    a = ap.parse_args(argv)

    if a.check_noaa:
        print("\n".join(check_noaa_live()))
        return 0
    if not a.model:
        sys.exit("--model is required")

    cfg = load_config(a.config, [])
    src = next((s.name for s in cfg.sources if s.role == "eval"), cfg.sources[0].name)
    ctx = orch._context(cfg, True, [src])
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    bundle = ctx.state["bundles"][src]

    pkg = load_daily_model(a.model)
    out = predict_tomorrow(pkg, bundle, fetch_live_srs())
    age_h = feed_age_hours(bundle)
    if age_h > 3.0:
        out["warnings"].append(
            f"the live X-ray feed is {age_h:.1f} h old (its newest sample is that long ago); "
            "the forecast may be built from stale data"
        )
    for_day = date.fromisoformat(out["for_day"])
    noaa = fetch_noaa_live()
    p_noaa = noaa_prob_for_day(noaa, for_day, pkg["target"])
    p_rec, thr_rec = apply_reference(pkg.get("noaa_recal"), np.array([p_noaa]))
    out["noaa"] = None if not np.isfinite(p_noaa) else p_noaa
    out["noaa_note"] = (
        f"NOAA's published probability for {for_day} (issued {noaa['issued']})"
        if np.isfinite(p_noaa) else "NOAA's live forecast could not be read (try --check-noaa)"
    )  # fmt: skip
    out["note"] = (
        "Trained on past days only; how it compares with NOAA is whatever the walk-forward report and the "
        "forward test measured, not what this one number suggests."
    )
    Path(a.out).write_text(json.dumps(out, indent=2))
    if a.merge:
        p = Path(a.merge)
        doc = json.loads(p.read_text())
        doc["daily"] = out
        p.write_text(json.dumps(doc, separators=(",", ":")))

    if a.log:
        n_res = resolve_outcomes(a.log, bundle, pkg["target"])
        added = append_forecast(a.log, {
            "for_day": out["for_day"], "issued_utc": out["issued_utc"], "model_id": model_id(a.model),
            "trained_until": pkg.get("trained_until"), "target": pkg["target"], "p_flarenet": out["probability"],
            "threshold": out["alert_threshold"], "alert": int(out["alert"]),
            "p_noaa": None if not np.isfinite(p_noaa) else p_noaa,
            "p_noaa_recal": None if not np.isfinite(p_rec[0]) else float(p_rec[0]),
            "threshold_noaa_recal": None if not np.isfinite(thr_rec) else thr_rec,
            "noaa_source": noaa["source"] if noaa and np.isfinite(p_noaa) else None,
            "noaa_issued": str(noaa["issued"]) if noaa and np.isfinite(p_noaa) else None,
            "regions_used": int(out["regions_used"]), "warnings": "; ".join(out["warnings"]) or None,
        })  # fmt: skip
        print(
            f"log: resolved {n_res} earlier day(s); {'added' if added else 'already had'} {out['for_day']}"
        )

    cls = pkg["target"]
    noaa_txt = "n/a" if out["noaa"] is None else f"{out['noaa']:.0%}"
    print(
        f"{out['for_day']}: P({cls}-class flare) = {out['probability']:.0%} (alert threshold "
        f"{out['alert_threshold']:.0%}, alert={out['alert']}); NOAA: {noaa_txt}"
    )
    for w in out["warnings"]:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

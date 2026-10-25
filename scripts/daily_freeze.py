"""Freeze the model for a forward test, from the day table a `flarenet daily` run already wrote.

    uv run python scripts/daily_freeze.py -c configs/daily_noaa.yaml --table runs/daily_noaa/day_table.csv --out models

Refits the headline model on every labelled day in the table (seconds to a minute, no downloads) and writes
models/daily_model_live.pkl (+ daily_model.pkl), including NOAA's frozen recalibration. The forward test must use
ONE frozen model: freeze once, commit the file, and do not retrain while the test runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from flarenet.core.config import load_config
from flarenet.daily.live import model_id
from flarenet.daily.pipeline import fit_final_models


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--table", required=True, help="runs/<name>/day_table.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    tab = pd.read_csv(a.table, index_col=0, parse_dates=True)
    if tab.index.tz is None:
        tab.index = tab.index.tz_localize("UTC")
    if "y" not in tab.columns:
        tab["y"] = tab["y_m" if cfg.daily.target == "M" else "y_x"].astype(int)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = fit_final_models(tab, cfg, out)
    for p in paths.values():
        print(
            f"wrote {p}  (id {model_id(p)}, trained through {tab.index[-1].date()}, {len(tab)} days)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

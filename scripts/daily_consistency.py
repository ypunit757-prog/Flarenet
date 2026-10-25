"""Re-check a finished `flarenet daily` run: is FlareNet's edge over NOAA consistent over time?

    uv run python scripts/daily_consistency.py runs/daily_noaa/predictions.csv

Reads the predictions.csv a run wrote (no retraining) and prints the same "Is the edge consistent?" section
that daily_report.md now contains: halves of the period, each year, AUC within years, and NOAA recalibrated on
a rolling year. Use it on runs made before this check existed.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from flarenet.daily.consistency import consistency, format_consistency


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("predictions", help="runs/<name>/predictions.csv")
    ap.add_argument("--block-days", type=int, default=14)
    ap.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    a = ap.parse_args(argv)
    d = pd.read_csv(a.predictions, index_col=0, parse_dates=True)
    need = {"y", "noaa", "p_noaa_cal", "p_model"}
    if not need <= set(d.columns):
        sys.exit(
            f"{a.predictions} lacks columns {sorted(need - set(d.columns))}; re-run `flarenet daily` with the current code"
        )
    print("\n".join(format_consistency(consistency(d, a.block_days, a.n))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score the forward-test log: FlareNet vs NOAA on days that had not happened when the forecast was made.

    uv run python scripts/daily_forward_score.py forward_log.csv --model models/daily_model_live.pkl

Fills in NOAA's archived forecast for days where the live read failed (and NOAA's frozen recalibration, with
--model), then prints the report. Read the verdict line first: it says "too early" until there are enough days.
"""

from __future__ import annotations

import argparse
import sys

from flarenet.daily.forward import (
    compare_live_vs_archive,
    fill_noaa_from_archive,
    format_forward,
    read_log,
    score_log,
)
from flarenet.daily.live import load_daily_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("log", help="forward_log.csv")
    ap.add_argument(
        "--model", default=None, help="the frozen daily_model_live.pkl (for NOAA's recalibration)"
    )
    ap.add_argument("--text-dir", default="data/downloads/noaa_text")
    ap.add_argument(
        "--no-fill", action="store_true", help="do not fetch NOAA's archive for days missing it"
    )
    ap.add_argument("--out", default=None, help="also write the report to this .md file")
    a = ap.parse_args(argv)
    if not a.no_fill:
        recal = load_daily_model(a.model).get("noaa_recal") if a.model else None
        try:
            n = fill_noaa_from_archive(a.log, a.text_dir, recal=recal)
            print(f"filled {n} NOAA forecast(s) from the archive")
            c = compare_live_vs_archive(a.log, a.text_dir)
            if c["compared"]:
                print(
                    f"NOAA live vs archive: {c['compared']} day(s) compared, {c['mismatched']} differ "
                    f"(largest difference {c['max_abs_diff']:.3f})"
                )
        except Exception as e:  # network trouble must not stop scoring what is already logged
            print(f"could not read NOAA's archive ({e}); scoring what is logged")
    md = format_forward(score_log(read_log(a.log)))
    print(md)
    if a.out:
        open(a.out, "w").write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

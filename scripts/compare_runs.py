"""Compare two training runs on the data neither of them trained on.

    uv run python scripts/compare_runs.py runs/goes_all runs/goes_all_chrono

Reads ``metrics_report.json`` of each run and prints, for every EXTERNAL dataset (role: eval, e.g. GOES-18) the
runs share: forecast AUC and TSS with their 95% block-bootstrap intervals, POD, FAR, Brier score, calibration error,
and, for a run that stored alert levels, what each level did on that dataset. Held-out splits of a training source are
not compared: they differ between a block split and a chronological split.

With ``--paired`` it also scores both runs on the SAME windows and gives an interval for the difference, which is far tighter than
comparing two separate intervals (both models move with the same quiet weeks and storms). That needs the per-window files
``window_predictions_<dataset>.csv`` in each run folder; runs made before they existed get them by re-scoring:
``uv run flarenet evaluate -c <config> --checkpoint <run>/checkpoint.pt --run-name <run name>``.

Reading it: if the AUC and TSS intervals overlap, the two models are not shown to differ in ranking skill. A lower
Brier score and calibration error mean the probabilities are closer to the observed flare frequency. A lower FAR at
a similar TSS means fewer false alarms for the same skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from flarenet.pipelines.evaluation.paired import paired_block_bootstrap, reading


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def _iv(ci, k):
    v = (ci or {}).get(k)
    return "" if not v or any(x != x for x in v) else f" [{v[0]:.2f}, {v[1]:.2f}]"


def _load(run: str) -> dict:
    p = Path(run) / "metrics_report.json"
    if not p.exists():
        sys.exit(f"{p} not found. Run `flarenet run -c <config>` for that run first.")
    return json.loads(p.read_text())


def _paired(run_a: str, run_b: str, ra: dict, rb: dict, key: str, names, n_boot: int) -> None:
    files = [Path(r) / f"window_predictions_{key}.csv" for r in (run_a, run_b)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(
            f"\n(paired comparison on {key} skipped: missing {', '.join(missing)}; re-score with `flarenet evaluate`)"
        )
        return
    da, db = (pd.read_csv(f) for f in files)
    m = da.merge(db, on="t_end", suffixes=("_a", "_b"))
    if len(m) < 100:
        print(f"\n(paired comparison on {key} skipped: only {len(m)} windows in common)")
        return
    if (m["y_a"] != m["y_b"]).any():
        print(
            f"\n(paired comparison on {key} skipped: the two runs label the shared windows differently)"
        )
        return
    res = paired_block_bootstrap(
        m["y_a"],
        m["p_a"],
        m["p_b"],
        m["t_end"],
        ra["alert_threshold"],
        rb["alert_threshold"],
        n=n_boot,
    )
    print(
        f"\n### {key}: {names[1]} minus {names[0]} on the same {len(m)} windows (block bootstrap, 95% interval)\n"
    )
    print("| metric | " + names[0] + " | " + names[1] + " | difference | 95% interval | reading |")
    print("|---|---|---|---|---|---|")
    for k, d in res.items():
        lo, hi = d["ci95"]
        print(
            f"| {k} | {_f(d['a'], 4)} | {_f(d['b'], 4)} | {d['diff']:+.4f} | {_f(lo, 4)} to {_f(hi, 4)} | {reading(k, d)} |"
        )
    print(
        "\nTSS, POD and FAR use each run's own saved alert threshold; AUC and Brier use the probabilities."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_a", help="e.g. runs/goes_all (the current model)")
    ap.add_argument("run_b", help="e.g. runs/goes_all_chrono (the candidate)")
    ap.add_argument(
        "--paired",
        action="store_true",
        help="also compare on identical windows (needs window_predictions_*.csv)",
    )
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples for --paired")
    a = ap.parse_args(argv)
    ra, rb = _load(a.run_a), _load(a.run_b)
    ext = [
        k for k, v in ra["results"].items()
        if v.get("kind") == "external" and rb["results"].get(k, {}).get("kind") == "external"
    ]  # fmt: skip
    if not ext:
        sys.exit("The two runs share no external (role: eval) dataset to compare on.")
    names = (Path(a.run_a).name, Path(a.run_b).name)
    for key in ext:
        print(f"\n## {key} (never trained on by either run)\n")
        print(f"| | {names[0]} | {names[1]} |\n|---|---|---|")
        va, vb = ra["results"][key], rb["results"][key]
        print(f"| alert threshold | {_f(ra['alert_threshold'])} | {_f(rb['alert_threshold'])} |")
        print(f"| probabilities | {'calibrated' if ra.get('calibration') else 'raw'} | "
              f"{'calibrated' if rb.get('calibration') else 'raw'} |")  # fmt: skip
        for label, k in (("AUC", "AUC"), ("TSS", "TSS"), ("HSS", "HSS")):
            print(f"| {label} | {_f(va[k])}{_iv(va.get('confidence_95'), k)} | "
                  f"{_f(vb[k])}{_iv(vb.get('confidence_95'), k)} |")  # fmt: skip
        for label, k in (
            ("POD (flares warned)", "POD"),
            ("FAR (false alarms among alerts)", "FAR"),
        ):
            print(f"| {label} | {_f(va[k])} | {_f(vb[k])} |")
        ca, cb = va.get("calibration") or {}, vb.get("calibration") or {}
        print(
            f"| Brier score (lower is better) | {_f(ca.get('brier'), 4)} | {_f(cb.get('brier'), 4)} |"
        )
        print(
            f"| calibration error ECE (lower is better) | {_f(ca.get('ece'), 4)} | {_f(cb.get('ece'), 4)} |"
        )
        la, lb = va.get("lead_time") or {}, vb.get("lead_time") or {}
        print(f"| M/X flares warned in time | {la.get('n_hits', 'n/a')} of {la.get('n_events', 'n/a')} | "
              f"{lb.get('n_hits', 'n/a')} of {lb.get('n_events', 'n/a')} |")  # fmt: skip
        for nm, v in ((names[0], va), (names[1], vb)):
            lv = v.get("alert_levels") or {}
            if lv:
                print(
                    f"\nAlert levels of {nm} on {key} (thresholds fixed on its validation windows):\n"
                )
                print("| level | threshold | POD | FAR | TSS |\n|---|---|---|---|---|")
                for lname, x in lv.items():
                    print(
                        f"| {lname} | {_f(x['threshold'])} | {_f(x['POD'])} | {_f(x['FAR'])} | {_f(x['TSS'])} |"
                    )
        if a.paired:
            _paired(a.run_a, a.run_b, ra, rb, key, names, a.bootstrap)
    print(
        "\nIf the AUC and TSS intervals overlap, ranking skill is not shown to differ. Prefer the candidate when it is "
        "level on those and better on Brier/ECE, because its threshold and probabilities were fitted on a later period."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

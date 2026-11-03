"""Fit a tree model on long-memory features and stack it with a trained 6-hour network.

    uv run python scripts/gbm_6h.py -c configs/noaa_all_chrono.yaml

Uses the checkpoint ``runs/<run_name>/checkpoint.pt`` of the config (or ``--checkpoint``) and the windows already cached
by ``flarenet run``; it needs no new downloads. About 5 to 10 minutes on a laptop for the eight-year GOES-16 series
(most of it building the features). Add `--live-compatible` to leave out the features a live forecast could not compute (27-day
rotation, 168-hour windows). Results go to ``runs/<run_name>_gbm/``:

    gbm6h_report.md    network vs tree model vs stack on validation, the later part of GOES-16 and GOES-18, with
                       paired intervals for the differences
    gbm6h_report.json  the same numbers
    gbm6h_predictions_*.csv   per-window probabilities of the three models
    gbm6h.pkl          the fitted tree model and stacking weights (not used by the live forecast)

What decides whether it helped: the "stack minus network" table on goes18_test. If its AUC/Brier interval includes
zero, the tree model adds nothing measurable and the network alone stays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flarenet.core.config import load_config
from flarenet.core.utils import resolve_device
from flarenet.gbm6h.pipeline import LIVE_COMPATIBLE_DROP, fit_and_evaluate
from flarenet.orchestration import runner as orch
from flarenet.pipelines.training.checkpoint import load_checkpoint


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", help="default: <runs_dir>/<run_name>/checkpoint.pt")
    ap.add_argument("--out", help="default: <runs_dir>/<run_name>_gbm")
    ap.add_argument(
        "--train-stride",
        type=int,
        default=60,
        help="minutes between training windows of the tree model",
    )
    ap.add_argument("--live-compatible", action="store_true",
                    help="drop the features the live NOAA feed (7 days of history) cannot provide: rot_* and *168*")  # fmt: skip
    ap.add_argument(
        "--drop", default="", help="comma-separated column patterns to leave out, e.g. 'rot_*,h_*'"
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ckpt = Path(a.checkpoint) if a.checkpoint else cfg.run_dir / "checkpoint.pt"
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}: train first with `uv run flarenet run -c {a.config}`.")
        return 2
    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(ckpt, device)
    ctx = orch.run_features(cfg)
    drop = tuple(p for p in a.drop.split(",") if p) + (
        LIVE_COMPATIBLE_DROP if a.live_compatible else ()
    )
    suffix = "_gbm_live" if a.live_compatible else "_gbm"
    out = Path(a.out) if a.out else Path(cfg.paths.runs_dir) / f"{cfg.run_name}{suffix}"
    fit_and_evaluate(cfg, ctx.state["splits"], ctx.state["bundles"], lm, device, out,
                     train_stride=a.train_stride, n_boot=a.bootstrap, seed=a.seed, drop=drop)  # fmt: skip
    print((out / "gbm6h_report.md").read_text())
    print(f"Written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare the CNN-BiLSTM with an LSTM, Random Forest, XGBoost and gradient boosting on the same windows.

    uv run python scripts/compare_models.py -c configs/noaa_all_chrono.yaml

Uses the checkpoint ``runs/<run_name>/checkpoint.pt`` of the config (or ``--checkpoint``) and the windows already cached
by ``flarenet run``; it needs no new downloads. Every new model is trained on the training period, calibrated and given its
alert threshold on the validation period, and scored on the held-out data (GOES-18 in ``noaa_all_chrono.yaml``) together with
the CNN-BiLSTM. Results go to ``runs/<run_name>_compare/``:

    model_comparison.md      accuracy, precision, recall, F1, ROC-AUC, PR-AUC, TSS, Brier, confusion matrices and the
                             paired differences from the CNN-BiLSTM, per held-out dataset
    model_comparison.json    the same numbers
    roc_pr_*.png             ROC and precision-recall curves, all models on the same windows
    confusion_*.png          confusion matrices
    model_comparison_predictions_*.csv   per-window probabilities of every model

XGBoost is optional (``pip install xgboost`` or ``uv sync --extra ml``); without it that row is reported as not run.
About 10 to 25 minutes on a laptop for the eight-year GOES-16 series (most of it building the features and training the LSTM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flarenet.compare.pipeline import ALL_MODELS, compare_models
from flarenet.core.config import load_config
from flarenet.core.utils import resolve_device, setup_logging
from flarenet.gbm6h.pipeline import LIVE_COMPATIBLE_DROP
from flarenet.orchestration import runner as orch
from flarenet.pipelines.training.checkpoint import load_checkpoint


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", help="default: <runs_dir>/<run_name>/checkpoint.pt")
    ap.add_argument("--out", help="default: <runs_dir>/<run_name>_compare")
    ap.add_argument("--train-stride", type=int, default=60, help="minutes between training windows")
    ap.add_argument("--lstm-epochs", type=int, default=8)
    ap.add_argument(
        "--skip",
        default="",
        help=f"comma-separated models to leave out: {', '.join(ALL_MODELS[1:])}",
    )
    ap.add_argument("--live-compatible", action="store_true",
                    help="drop the features the live NOAA feed (7 days of history) cannot provide")  # fmt: skip
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    setup_logging("INFO")

    cfg = load_config(a.config)
    ckpt = Path(a.checkpoint) if a.checkpoint else cfg.run_dir / "checkpoint.pt"
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}: train first with `uv run flarenet run -c {a.config}`.")
        return 2
    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(ckpt, device)
    ctx = orch.run_features(cfg)
    out = Path(a.out) if a.out else Path(cfg.paths.runs_dir) / f"{cfg.run_name}_compare"
    skip = tuple(s.strip() for s in a.skip.split(",") if s.strip())
    compare_models(cfg, ctx.state["splits"], ctx.state["bundles"], lm, device, out,
                   train_stride=a.train_stride, n_boot=a.bootstrap, seed=a.seed,
                   drop=LIVE_COMPATIBLE_DROP if a.live_compatible else (),
                   lstm_epochs=a.lstm_epochs, skip=skip)  # fmt: skip
    print((out / "model_comparison.md").read_text())
    print(f"Written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How much of the forecast skill is the data, and how much the network?

Trains a plain gradient-boosted tree on simple per-window summary statistics (last value, mean,
max, min, std, drift, mean of the last 10 minutes of every channel) on the same train / val / test
split as FlareNet, for several seeds, and prints its forecast AUC / TSS next to FlareNet's results
from an experiment run (``results.jsonl`` written by ``flarenet experiment``).

If the tree model matches the network, forecast skill is limited by the data, and more network
tweaking will not help: add information (features, data, instruments) instead.

    python scripts/headroom_check.py --seeds 1,2,3,4,5 --results runs/stage2/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from flarenet.core.config import load_config
from flarenet.orchestration import runner as orch
from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc, score_at_threshold


def summary_features(X: np.ndarray) -> np.ndarray:
    """(n, channels, time) windows -> (n, 7 * channels) summary statistics."""
    return np.concatenate(
        [
            X[:, :, -1],
            X.mean(2),
            X.max(2),
            X.min(2),
            X.std(2),
            X[:, :, -1] - X[:, :, 0],
            X[:, :, -10:].mean(2),
        ],
        axis=1,
    )


def tree_baseline(seed: int, config: str | None, workdir: str) -> dict:
    cfg = load_config(config, [f"seed={seed}", "evaluation.bootstrap_n=0", "training.device=cpu"])
    tmp = tempfile.mkdtemp(dir=workdir)
    cfg.paths.data_dir, cfg.paths.runs_dir, cfg.run_name = tmp + "/data", tmp + "/runs", "h"
    cfg.validate()
    ctx = orch._context(cfg, False, [s.name for s in cfg.train_sources()])
    orch._prepare(ctx, "features")
    sp = ctx.state["splits"]
    train, val, test = sp["train"], sp["val"], next(iter(sp["test"].values()))
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0
    )
    clf.fit(summary_features(train.X), train.y_fc)
    p_val = clf.predict_proba(summary_features(val.X))[:, 1]
    p_test = clf.predict_proba(summary_features(test.X))[:, 1]
    thr, val_tss = best_threshold(val.y_fc, p_val)
    return {
        "seed": seed,
        "tree_val_TSS": val_tss,
        "tree_AUC": safe_auc(test.y_fc, p_test),
        "tree_TSS": score_at_threshold(test.y_fc, p_test, thr)["TSS"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--config", default=None, help="base config (default: built-in synthetic run)")
    ap.add_argument("--results", default=None, help="results.jsonl of `flarenet experiment`")
    ap.add_argument("--variant", default="baseline", help="FlareNet variant to compare with")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as work:
        rows = [tree_baseline(int(s), args.config, work) for s in args.seeds.split(",")]
    out = pd.DataFrame(rows).set_index("seed")
    if args.results and Path(args.results).exists():
        nn = pd.DataFrame([json.loads(x) for x in Path(args.results).read_text().splitlines() if x])
        nn = nn[nn.variant == args.variant].set_index("seed")[["AUC", "TSS"]]
        out = out.join(nn.rename(columns=lambda c: f"flarenet_{c}"))
    print(out.round(3).to_string())
    print("\nmean over seeds")
    print(out.mean().round(3).to_string())


if __name__ == "__main__":
    main()

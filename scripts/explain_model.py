"""Which inputs do the deployed models rely on? (global explanation on held-out data)

    uv run python scripts/explain_model.py -c configs/noaa_all_chrono.yaml \
        --checkpoint models/flarenet.pt --stack models/gbm6h_live.pkl

Scores the network and the tree model of the live forecast on a dataset they were not trained on (the first ``role: eval`` source,
GOES-18 in noaa_all_chrono.yaml, or --dataset), and writes to ``runs/<run_name>_explain/``:

    explain_global.json      importance of every input group, for both models (the website's Explainable AI page can show it:
                             pass it to scripts/export_web.py with --explain-global)
    explain_global.md        the same as tables
    explain_importance.png   bar charts

Add ``--shap`` for per-feature SHAP values of the tree model (needs ``pip install shap``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from flarenet.compare.pipeline import _period
from flarenet.core import jsonsafe
from flarenet.core.config import load_config
from flarenet.core.utils import resolve_device, setup_logging
from flarenet.explain.report import build_global_report, plot_global, render_markdown
from flarenet.gbm6h.features import set_features
from flarenet.gbm6h.live import load_live_stack
from flarenet.gbm6h.pipeline import LIVE_COMPATIBLE_DROP, drop_features
from flarenet.orchestration import runner as orch
from flarenet.pipelines.evaluation.pipeline import align_channels
from flarenet.pipelines.evaluation.scoring import infer
from flarenet.pipelines.training.checkpoint import load_checkpoint


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", default="models/flarenet.pt")
    ap.add_argument("--stack", default="models/gbm6h_live.pkl")
    ap.add_argument(
        "--dataset", help="held-out dataset to explain (default: the first eval source)"
    )
    ap.add_argument("--out", help="default: <runs_dir>/<run_name>_explain")
    ap.add_argument(
        "--max-windows", type=int, default=4000, help="windows scored for permutation importance"
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    setup_logging("INFO")

    cfg = load_config(a.config)
    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(a.checkpoint, device)
    live, why = load_live_stack(a.stack, a.checkpoint)
    if live is None:
        print(f"Cannot use {a.stack} with {a.checkpoint}: {why}.")
        return 2
    if live.reference is None:
        print(
            "The stack file has no reference row; re-create it with scripts/gbm_6h.py --live-compatible --install."
        )
        return 2
    ctx = orch.run_features(cfg)
    splits, bundles = ctx.state["splits"], ctx.state["bundles"]
    held = {
        **{
            f"{k} (later period of a training source)": v for k, v in splits.get("test", {}).items()
        },
        **dict(splits.get("eval", {})),
    }
    if not held:
        print("The config has no held-out dataset to explain.")
        return 2
    name = a.dataset or next(iter(splits.get("eval", {}) or held))
    if name not in held:
        print(f"No dataset {name!r}; choose one of {list(held)}.")
        return 2
    ws = held[name]
    tab, rows = set_features(ws, bundles, cfg.data.horizon_min)
    tab = drop_features(tab, LIVE_COMPATIBLE_DROP)[0]
    sub = ws.subset(rows)
    rng = np.random.default_rng(a.seed)
    keep = np.sort(rng.choice(len(sub), size=min(a.max_windows, len(sub)), replace=False))
    sub, tab = sub.subset(keep), tab.iloc[keep]
    X = align_channels(sub, lm.channels).X
    ref_window = np.repeat(np.asarray(lm.mu), X.shape[2], axis=2)[0]

    def proba(w):
        return infer(lm.model, w, lm.mu, lm.sg, device, calibration=lm.calibration)[2]

    rep = build_global_report(name, _period(sub, bundles), sub.y_fc, tab, X, proba, live, list(lm.channels), ref_window,
                              repeats=a.repeats, use_shap=a.shap, seed=a.seed)  # fmt: skip
    out = Path(a.out) if a.out else Path(cfg.paths.runs_dir) / f"{cfg.run_name}_explain"
    out.mkdir(parents=True, exist_ok=True)
    (out / "explain_global.json").write_text(jsonsafe.dumps(rep, indent=2))
    (out / "explain_global.md").write_text(render_markdown(rep))
    plot_global(rep, out / "explain_importance.png")
    print((out / "explain_global.md").read_text())
    print(f"Written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

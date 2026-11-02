"""Random search over the 6-hour model's hyper-parameters, scored on the VALIDATION period only.

    uv run python scripts/tune_6h.py -c configs/noaa_all_chrono.yaml --trials 12 --confirm 3

Every trial trains one network on the training part of the series and scores it on the validation part (the
chronological split of the config), after the same Platt calibration the real run uses. The held-out test windows and
the eval sources (GOES-18) are never read, so they stay an honest final check: run the best config once, then compare
it with the current model using ``scripts/compare_runs.py --paired``.

Reading the table: trials are ranked by validation AUC (it does not depend on a threshold; published flare work reports
TSS swinging by 0.1 to 0.2 between splits, so TSS is a poor thing to optimise), with the Brier score next to it. The
best of N noisy trials looks better on validation than it will be anywhere else, so the top ``--confirm`` trials are
retrained with other seeds and ranked by the mean; a gain smaller than the seed spread printed there is noise.

    trials.csv          one row per training run
    tuning_report.md    the ranked table
    best_config.yaml    the base config with the winning settings, ready for `flarenet run -c`

Each trial is a full training run; ``--trials 12 --confirm 3`` makes 21 of them (13 settings, then the baseline and the
top three re-run with two more seeds each), so allow about 21 times the time of one run (the log prints it per trial).
An interrupted search continues with ``--resume``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flarenet.core.config import apply_overrides, load_config
from flarenet.core.utils import get_logger, resolve_device
from flarenet.orchestration import runner as orch
from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc
from flarenet.pipelines.evaluation.pipeline import align_channels
from flarenet.pipelines.evaluation.scoring import infer
from flarenet.pipelines.evaluation.uncertainty import calibration_summary
from flarenet.pipelines.training.checkpoint import load_checkpoint
from flarenet.pipelines.training.trainer import train_model

log = get_logger("tune_6h")

SPACE: dict[str, list] = {
    "training.lr": [2e-4, 5e-4, 1e-3],
    "model.dropout": [0.2, 0.35, 0.5],
    "training.weight_decay": [1e-5, 1e-4, 1e-3],
    "training.label_smoothing": [0.0, 0.08, 0.15],
    "training.focal_gamma": [0.0, 1.0, 2.0, 3.0],
    "training.forecast_loss_weight": [1.0, 2.5, 4.0],
    "training.forecast_balance": ["fixed", "auto"],
    "training.ema_decay": [0.0, 0.99],
    "model.cnn_channels": [16, 32, 48],
    "model.forecast_hidden": [24, 40, 64],
    "training.select_metric": ["tss", "auc", "tss_auc"],
}


def sample(rng: random.Random, space: dict[str, list] = SPACE) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def validation_scores(cfg, sp, out_dir: Path, device) -> dict:
    """Train with ``cfg`` and score the trained checkpoint on the validation windows only."""
    t0 = time.time()
    train_model(cfg, sp["train"], sp["val"], out_dir, sp["source_weights"])
    lm = load_checkpoint(out_dir / "checkpoint.pt", device)
    ws = align_channels(sp["val"], lm.channels)
    _, _, p = infer(lm.model, ws.X, lm.mu, lm.sg, device, calibration=lm.calibration)
    y = np.asarray(ws.y_fc, dtype=float)
    return {
        "val_auc": safe_auc(y, p),
        "val_brier": float(np.mean((p - y) ** 2)),
        "val_ece": float(calibration_summary(y, p).get("ece", np.nan)),
        "val_tss": float(best_threshold(y, p, smooth=2)[1]),
        "minutes": round((time.time() - t0) / 60.0, 1),
    }


def run_trial(base, sp, overrides: dict, out_dir: Path, seed: int, device) -> dict:
    cfg = copy.deepcopy(base)
    apply_overrides(cfg, [f"{k}={v}" for k, v in overrides.items()] + [f"seed={seed}"])
    cfg.training.calibrate = (
        "platt"  # the same calibration for every trial, so the Brier scores are comparable
    )
    cfg.validate()
    return validation_scores(cfg, sp, out_dir, device)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per setting, best validation AUC first (ties within 0.001 broken by Brier)."""
    g = df.groupby("trial").agg(
        overrides=("overrides", "first"), seeds=("seed", "count"),
        val_auc=("val_auc", "mean"), val_auc_sd=("val_auc", "std"),
        val_brier=("val_brier", "mean"), val_ece=("val_ece", "mean"), val_tss=("val_tss", "mean"),
    )  # fmt: skip
    g["auc_bucket"] = (g["val_auc"] / 0.001).round()
    return g.sort_values(["auc_bucket", "val_brier"], ascending=[False, True]).drop(
        columns="auc_bucket"
    )


def write_report(out: Path, base_name: str, table: pd.DataFrame, best: int | None = None) -> None:
    lines = [f"# Tuning of `{base_name}` (validation period only)", "",
             "| rank | trial | seeds | val AUC (mean +- sd) | val Brier | val ECE | val TSS | settings |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for i, (trial, r) in enumerate(table.iterrows(), 1):
        sd = "" if not np.isfinite(r["val_auc_sd"]) else f" +- {r['val_auc_sd']:.4f}"
        star = " (best)" if best is not None and int(trial) == best else ""
        lines.append(f"| {i} | {trial}{star} | {int(r['seeds'])} | {r['val_auc']:.4f}{sd} | {r['val_brier']:.4f} | "
                     f"{r['val_ece']:.4f} | {r['val_tss']:.3f} | `{r['overrides']}` |")  # fmt: skip
    lines += ["", "Trial 0 is the config as it stands. Only settings re-run with every seed (the baseline and the top few) can win. A setting only counts as better than trial 0 if its validation AUC "
              "exceeds it by more than the seed spread (the +- column); otherwise it is noise. Validation numbers are "
              "optimistic for the winner: the unbiased check is one run on GOES-18 followed by "
              "`scripts/compare_runs.py --paired`.", ""]  # fmt: skip
    (out / "tuning_report.md").write_text("\n".join(lines))


def write_best_config(base_raw: dict, out: Path, overrides: dict, run_name: str) -> Path:
    raw = copy.deepcopy(base_raw)
    for k, v in overrides.items():
        section, key = k.split(".", 1)
        raw.setdefault(section, {})[key] = v
    raw.setdefault("training", {})["calibrate"] = "platt"
    raw["run_name"] = run_name
    path = out / "best_config.yaml"
    header = "# written by scripts/tune_6h.py: the base config with the best validation settings\n"
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False))
    return path


def search(
    cfg, base_raw: dict, out: Path, trials: int = 12, confirm: int = 3, confirm_seeds: int = 2,
    search_seed: int = 0, resume: bool = False, space: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, Path]:  # fmt: skip
    """Run the search; returns the ranked table and the path of ``best_config.yaml``."""
    ctx = orch.run_features(cfg)
    sp = ctx.state["splits"]
    device = resolve_device(cfg.training.device)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "trials.csv"
    rows = pd.read_csv(csv).to_dict("records") if resume and csv.exists() else []
    done = {(int(r["trial"]), int(r["seed"])) for r in rows}

    rng = random.Random(search_seed)
    settings = [{}] + [sample(rng, space or SPACE) for _ in range(trials)]

    def go(trial: int, ov: dict, seed: int) -> None:
        if (trial, seed) in done:
            return
        log.info("trial %d seed %d: %s", trial, seed, json.dumps(ov))
        m = run_trial(cfg, sp, ov, out / f"trial_{trial:02d}_s{seed}", seed, device)
        rows.append({"trial": trial, "seed": seed, "overrides": json.dumps(ov), **m})
        pd.DataFrame(rows).to_csv(csv, index=False)
        log.info(
            "  -> val AUC %.4f, Brier %.4f (%.1f min)", m["val_auc"], m["val_brier"], m["minutes"]
        )

    for t, ov in enumerate(settings):
        go(t, ov, cfg.seed)
    first = rank(pd.DataFrame(rows))
    # the baseline is always re-run with the extra seeds, so the winner is compared with it on equal terms
    top = [int(t) for t in first.index if int(t) != 0][: max(0, confirm)]
    for trial in [0, *top]:
        for s in range(1, confirm_seeds + 1):
            go(trial, settings[trial], cfg.seed + s)
    table = rank(pd.DataFrame(rows))
    # only settings run with every seed can win: a single lucky seed of an unconfirmed trial must not outrank a
    # confirmed mean (selecting the best of many noisy trials is optimistic)
    full = table[table["seeds"] == table["seeds"].max()]
    best = int(full.index[0])
    write_report(out, cfg.run_name, table, best)
    path = write_best_config(base_raw, out, settings[best], f"{cfg.run_name}_tuned")
    return table, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "-c",
        "--config",
        required=True,
        help="e.g. configs/noaa_all_chrono.yaml (use a chronological split)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=12,
        help="settings to try, besides trial 0 (the config as it is)",
    )
    ap.add_argument("--confirm", type=int, default=3, help="retrain the top K with other seeds")
    ap.add_argument(
        "--confirm-seeds", type=int, default=2, help="extra seeds per confirmed setting"
    )
    ap.add_argument("--out", default="runs/tune_6h")
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip runs already in <out>/trials.csv")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if cfg.data.split != "chrono":
        print("WARNING: data.split is not 'chrono': validation shares active periods with training and flatters every "
              "trial equally. Use configs/noaa_all_chrono.yaml.\n")  # fmt: skip
    base_raw = yaml.safe_load(Path(a.config).read_text()) or {}
    out = Path(a.out)
    table, path = search(
        cfg, base_raw, out, a.trials, a.confirm, a.confirm_seeds, a.search_seed, a.resume
    )
    print((out / "tuning_report.md").read_text())
    best = int(table.index[0])
    print(f"Best setting: trial {best}. Run it once with:\n  uv run flarenet run -c {path}\n"
          f"  uv run python scripts/compare_runs.py runs/{cfg.run_name} runs/{cfg.run_name}_tuned --paired")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Multi-seed A/B experiments: does a change help, or did it just get a lucky split?

A single train/val/test split of ~800 windows is noisy (the test TSS bootstrap interval of the
default run is [0.40, 0.81]). This module runs every *variant* (a list of ``--set`` overrides)
for every *seed* (a different synthetic world / split / initialisation when the source uses the
seed), then reports mean +- std per variant and, for each variant, the paired difference to the
baseline over the same seeds, so a gain has to be consistent to count.

    flarenet experiment -e configs/experiments/model_improvements.yaml --out runs/exp --jobs 2

Experiment file::

    seeds: [1, 2, 3, 4, 5]
    baseline: baseline                # variant the others are compared with
    base_overrides: ["training.epochs=35"]   # applied to every variant
    variants:
      baseline: []
      ema: ["training.ema_decay=0.95"]

Results are appended to ``<out>/results.jsonl`` as they finish, so an interrupted experiment
resumes where it stopped. Outputs: ``results.csv`` (one row per variant x seed) and
``summary.md``.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flarenet.core.utils import get_logger

log = get_logger(__name__)

# (column, label, higher_is_better)
KEY_METRICS = [
    ("AUC", "forecast AUC", True),
    ("TSS", "forecast TSS", True),
    ("HSS", "forecast HSS", True),
    ("brier", "Brier", False),
    ("ece", "ECE", False),
    ("macro_F1", "nowcast macro-F1", True),
    ("accuracy", "nowcast acc", True),
    ("recall_M", "nowcast recall M", True),
    ("val_tss", "val TSS", True),
    ("epochs", "epochs", False),
    ("seconds", "train s", False),
]


def load_experiment(path: str | Path) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    variants = spec.get("variants") or {}
    if not variants:
        raise ValueError(f"{path}: needs a non-empty 'variants' mapping")
    spec.setdefault("seeds", [1, 2, 3])
    spec.setdefault("base_overrides", [])
    spec.setdefault("baseline", next(iter(variants)))
    if spec["baseline"] not in variants:
        raise ValueError(f"baseline {spec['baseline']!r} is not one of the variants")
    spec["variants"] = {k: list(v or []) for k, v in variants.items()}
    return spec


def _recalls(res) -> dict[str, float]:
    from flarenet.core.constants import CLASS_NAMES

    yt, yp = np.asarray(res.nowcast_true), np.asarray(res.nowcast_pred)
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        m = yt == i
        out[f"recall_{name}"] = float((yp[m] == i).mean()) if m.any() else float("nan")
    return out


def run_one(
    config_path: str | None,
    overrides: list[str],
    variant: str,
    seed: int,
    root: str,
    device: str | None,
    threads: int,
) -> dict:
    """Train + evaluate one (variant, seed). Runs in a worker process."""
    import torch

    torch.set_num_threads(max(1, threads))
    from flarenet.core.config import load_config
    from flarenet.core.pipeline import PipelineContext
    from flarenet.orchestration import runner as orch
    from flarenet.pipelines import evaluation, training

    root_p = Path(root)
    ov = [*overrides, f"seed={seed}", "evaluation.bootstrap_n=0"]
    if device:
        ov.append(f"training.device={device}")
    cfg = load_config(config_path, ov)
    # One private data cache per run: variants may cut windows differently, and parallel workers
    # must never write the same cache directory.
    data_dir = root_p / f"seed_{seed}" / variant / "data"
    cfg.paths.data_dir = str(data_dir)
    cfg.paths.runs_dir = str(root_p / f"seed_{seed}" / variant)
    cfg.run_name = "run"
    cfg.validate()

    t0 = time.time()
    ctx: PipelineContext = orch._context(
        cfg, False, None
    )  # all sources: role=eval ones are scored too
    orch._prepare(ctx, "features")
    training.build_pipeline().run(ctx)
    train_s = time.time() - t0
    evaluation.build_pipeline().run(ctx)

    tr = ctx.state["training"]
    shutil.rmtree(data_dir, ignore_errors=True)  # keep the (small) run folder, not the data cache
    row: dict = {
        "variant": variant,
        "seed": seed,
        "val_tss": tr["best_val_tss"],
        "epochs": len(tr["history"]["train_loss"]),
        "seconds": train_s,
        "n_parameters": tr["n_parameters"],
        "threshold": tr["threshold"],
        "n_train": len(ctx.state["splits"]["train"]),
    }
    for name, res in ctx.state["eval_results"].items():
        s = res.summary()
        pre = "" if res.kind == "test" and not any(k.endswith(".AUC") for k in row) else f"{name}."
        for k in ("AUC", "TSS", "HSS", "POD", "FAR", "macro_F1", "accuracy", "TSS_oracle"):
            row[pre + k] = s[k]
        for k, v in (s.get("calibration") or {}).items():
            row[pre + k] = v
        lead = s.get("lead_time") or {}
        row[pre + "lead_median"] = lead.get("median_min")
        row[pre + "hit_rate"] = lead.get("hit_rate")
        for pc in s["perclass"]:
            row[pre + f"TSS_{pc['class']}"] = pc["TSS"]
        for k, v in _recalls(res).items():
            row[pre + k] = v
    return row


def _done(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    rows = []
    for line in results_path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # half-written last line of an interrupted run
                pass
    return rows


def run_experiment(
    spec: dict,
    out_dir: str | Path,
    config_path: str | None = None,
    jobs: int = 1,
    only: list[str] | None = None,
    seeds: list[int] | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    seeds = seeds or [int(s) for s in spec["seeds"]]
    names = list(spec["variants"])
    if only:
        unknown = [n for n in only if n not in spec["variants"]]
        if unknown:
            raise KeyError(f"Unknown variant(s) {unknown}; available: {names}")
        names = [n for n in names if n in only or n == spec["baseline"]]
    rows = _done(results_path)
    have = {(r["variant"], int(r["seed"])) for r in rows}
    todo = [(v, s) for s in seeds for v in names if (v, s) not in have]
    log.info("experiment: %d variants x %d seeds, %d runs to do (%d cached)", len(names),
             len(seeds), len(todo), len(have))  # fmt: skip
    threads = max(1, (os.cpu_count() or 1) // max(jobs, 1))

    def record(row: dict) -> None:
        rows.append(row)
        with open(results_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        log.info("done %-22s seed %-4s val TSS %.3f | test AUC %.3f TSS %.3f macroF1 %.3f", row["variant"],
                 row["seed"], row["val_tss"], row.get("AUC", float("nan")), row.get("TSS", float("nan")),
                 row.get("macro_F1", float("nan")))  # fmt: skip

    def args(v: str, s: int) -> tuple:
        return (
            config_path,
            [*spec["base_overrides"], *spec["variants"][v]],
            v,
            s,
            str(out),
            device,
            threads,
        )

    if jobs <= 1:
        for v, s in todo:
            record(run_one(*args(v, s)))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futs = {pool.submit(run_one, *args(v, s)): (v, s) for v, s in todo}
            for fut in as_completed(futs):
                v, s = futs[fut]
                try:
                    record(fut.result())
                except Exception:
                    log.error("run %s seed %s failed:\n%s", v, s, traceback.format_exc())
    df = pd.DataFrame([r for r in rows if r["variant"] in names and int(r["seed"]) in seeds])
    df.to_csv(out / "results.csv", index=False)
    (out / "summary.md").write_text(summarise(df, spec["baseline"]))
    return df


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def summarise(df: pd.DataFrame, baseline: str) -> str:
    """Markdown: per-variant mean +- std, then paired differences against ``baseline``."""
    if df.empty:
        return "No results.\n"
    seeds = sorted(df["seed"].unique())
    lines = [f"# Experiment summary ({len(seeds)} seeds: {', '.join(map(str, seeds))})", ""]
    cols = [(c, lab, hib) for c, lab, hib in KEY_METRICS if c in df.columns]

    lines += ["## Mean +- std over seeds", "", "| variant | n | " + " | ".join(lab for _, lab, _ in cols) + " |",
              "|---|---|" + "---|" * len(cols)]  # fmt: skip
    for v, g in df.groupby("variant", sort=False):
        cells = []
        for c, _, _ in cols:
            x = g[c].astype(float)
            cells.append(
                f"{_fmt(x.mean(), 1 if c == 'seconds' else 3)} +- {_fmt(x.std(ddof=1) if len(x) > 1 else float('nan'), 1 if c == 'seconds' else 3)}"
            )
        lines.append(f"| {v} | {len(g)} | " + " | ".join(cells) + " |")

    if baseline in set(df["variant"]):
        base = df[df["variant"] == baseline].set_index("seed")
        lines += ["", f"## Paired difference to `{baseline}` (same seeds; + is better)", "",
                  "Each cell: mean difference +- standard error, and how many seeds improved. "
                  "A change is only worth keeping if it is consistently positive.", "",
                  "| variant | " + " | ".join(lab for c, lab, _ in cols if c not in ("epochs", "seconds")) + " |",
                  "|---|" + "---|" * len([1 for c, _, _ in cols if c not in ("epochs", "seconds")])]  # fmt: skip
        for v, g in df.groupby("variant", sort=False):
            if v == baseline:
                continue
            g = g.set_index("seed")
            common = sorted(set(g.index) & set(base.index))
            if not common:
                continue
            cells = []
            for c, _, hib in cols:
                if c in ("epochs", "seconds"):
                    continue
                d = (g.loc[common, c].astype(float) - base.loc[common, c].astype(float)).to_numpy()
                d = d if hib else -d  # make "+" always mean better
                d = d[np.isfinite(d)]
                if len(d) == 0:
                    cells.append("n/a")
                    continue
                se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
                cells.append(f"{d.mean():+.3f} +- {_fmt(se)} ({int((d > 0).sum())}/{len(d)})")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

"""Try to beat the live 6-hour forecast, and recommend a replacement only if a paired test says so.

    uv run python scripts/challenger_6h.py -c configs/noaa_all_chrono.yaml

Unattended, on your machine (allow many hours: it trains many networks). It does, in order:

  1. tune     validation-only search of training settings (scripts/tune_6h.py); GOES-18 is not read
  2. tuned    train the winning settings once   -> runs/<name>_tuned          (skipped if trial 0 won)
  3. ens      train five networks and average   -> runs/<name>_ens<N>         (on the tuned settings if any)
  4. stack    fit the tree model on each candidate network (scripts/gbm_6h.py --live-compatible)
  5. verdict  compare each candidate's FINAL forecast (network + tree model) with the one running now, on the same
              GOES-18 windows, with a paired block bootstrap

A candidate is recommended only if its AUC is better with a 95 % interval (corrected for the number of candidates)
that excludes zero AND its Brier score is not worse. Otherwise the answer is "keep the current model". Nothing in
models/ is changed: when a candidate wins, the exact commands to put it live are printed.

Needs the live model's own run directory (default runs/<name>_gbm_live, made by `gbm_6h.py --live-compatible`).
`--steps` picks stages (e.g. `--steps stack,verdict`); finished runs are reused, so an interrupted job continues.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flarenet.pipelines.evaluation.paired import paired_block_bootstrap

HERE = Path(__file__).resolve().parent
CLI = "import sys; from flarenet.cli import main; sys.exit(main())"
DATASET = "goes18_test"  # default: the evaluation source of configs/noaa_all_chrono.yaml


def pred_file(dataset: str) -> str:
    return f"gbm6h_predictions_{dataset}.csv"


STEPS = ("tune", "tuned", "ens", "stack", "verdict")


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def verdict(res: dict, brier_tolerance: float = 0.0) -> str:
    """'better' / 'worse' / 'no established difference' for a candidate (B) against the live model (A).

    Better needs the AUC interval of B - A entirely above zero and the Brier interval's upper end not above
    ``brier_tolerance`` (a better ranking bought with worse probabilities is not an improvement)."""
    auc_lo, auc_hi = res["AUC"]["ci95"]
    br_hi = res["Brier"]["ci95"][1]
    if np.isfinite(auc_lo) and auc_lo > 0 and np.isfinite(br_hi) and br_hi <= brier_tolerance:
        return "better"
    if np.isfinite(auc_hi) and auc_hi < 0:
        return "worse"
    return "no established difference"


def load_final(
    run_dir: Path, dataset: str = DATASET
) -> tuple[pd.DataFrame, float] | tuple[None, str]:
    """GOES-18 predictions of the final (stacked) forecast of a gbm_6h run and its validation threshold."""
    csv, rep = run_dir / pred_file(dataset), run_dir / "gbm6h_report.json"
    if not csv.exists() or not rep.exists():
        return None, f"{run_dir} has no {csv.name} / gbm6h_report.json"
    df = pd.read_csv(csv)
    if "p_stack" not in df:
        return (
            None,
            f"{run_dir}: the stacking regression gave the tree model no positive weight, no stack to compare",
        )
    return df, float(json.loads(rep.read_text())["thresholds_from_validation"]["stack"])


def compare(
    cur: Path, cand: Path, n_boot: int, alpha: float, seed: int = 0, dataset: str = DATASET
) -> dict | str:
    a, thr_a = load_final(cur, dataset)
    if a is None:
        return f"current model: {thr_a}"
    b, thr_b = load_final(cand, dataset)
    if b is None:
        return thr_b
    m = a.merge(b, on="t_end", suffixes=("_a", "_b"))
    if len(m) == 0 or not (m["y_a"] == m["y_b"]).all():
        return "the two runs were not scored on the same windows"
    return paired_block_bootstrap(
        m["y_a"],
        m["p_stack_a"],
        m["p_stack_b"],
        m["t_end"],
        thr_a,
        thr_b,
        n=n_boot,
        seed=seed,
        alpha=alpha,
    ) | {"windows": len(m)}


def derive_config(src: dict, name: str, ensemble: int | None = None) -> dict:
    raw = copy.deepcopy(src)  # YAML data holds dates (start: 2024-03-01), which JSON cannot copy
    raw["run_name"] = name
    raw.setdefault("training", {})["calibrate"] = "platt"
    if ensemble:
        raw["training"]["ensemble_size"] = int(ensemble)
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "-c", "--config", required=True, help="the config the live model was trained with"
    )
    ap.add_argument(
        "--current", help="run dir of the live model's gbm_6h run (default runs/<name>_gbm_live)"
    )
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument(
        "--ensemble", type=int, default=5, help="networks in the averaged candidate (0 = skip it)"
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    a = ap.parse_args(argv)

    steps = [s for s in a.steps.split(",") if s]
    if bad := [s for s in steps if s not in STEPS]:
        ap.error(f"unknown steps {bad}; choose from {STEPS}")
    base_raw = yaml.safe_load(Path(a.config).read_text()) or {}
    name = base_raw.get("run_name", "run")
    evals = [x["name"] for x in base_raw.get("sources", []) if x.get("role") == "eval"]
    dataset = evals[0] if evals else DATASET  # the satellite the models never trained on
    runs = Path(a.runs)
    cur = Path(a.current) if a.current else runs / f"{name}_gbm_live"
    tune_dir = runs / "tune_6h"
    cfg_dir = runs / "challengers"
    py = sys.executable

    # 1. tune ---------------------------------------------------------------------------------------------------
    if "tune" in steps:
        cmd = [py, str(HERE / "tune_6h.py"), "-c", a.config, "--trials", str(a.trials), "--confirm", str(a.confirm),
               "--out", str(tune_dir), "--resume"]  # fmt: skip
        if a.dry_run:
            print("$ " + " ".join(cmd))
        elif run(cmd) != 0:
            return 1
    best_json = tune_dir / "best_trial.json"
    tuned = None
    if best_json.exists() and json.loads(best_json.read_text())["trial"] != 0:
        tuned = yaml.safe_load((tune_dir / "best_config.yaml").read_text())
    elif a.dry_run and "tune" in steps:
        tuned = {}  # unknown until the search has run
    source = tuned if tuned is not None else base_raw
    print(
        f"\nTuning found a better setting: {'yes' if tuned else 'no (or not run): trial 0, the live settings, stays'}"
    )

    # 2-3. candidates -------------------------------------------------------------------------------------------
    cands: dict[str, dict] = {}
    if tuned is not None:
        cands[f"{name}_tuned"] = derive_config(source, f"{name}_tuned")
    if a.ensemble and a.ensemble > 1:
        cands[f"{name}_ens{a.ensemble}"] = derive_config(
            source, f"{name}_ens{a.ensemble}", a.ensemble
        )
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cname, raw in cands.items():
        paths[cname] = cfg_dir / f"{cname}.yaml"
        if not a.dry_run:
            paths[cname].write_text(yaml.safe_dump(raw, sort_keys=False))
    stage_of = lambda c: "ens" if "_ens" in c else "tuned"  # noqa: E731
    for cname in cands:
        if stage_of(cname) not in steps:
            continue
        ckpt = runs / cname / "checkpoint.pt"
        cmd = [py, "-c", CLI, "run", "-c", str(paths[cname])]
        if a.dry_run:
            print("$ " + " ".join(cmd))
        elif ckpt.exists():
            print(f"\n{cname}: reusing {ckpt}")
        elif run(cmd) != 0:
            return 1

    # 4. stack --------------------------------------------------------------------------------------------------
    for cname in cands:
        if "stack" not in steps:
            break
        out = runs / f"{cname}_gbm_live"
        cmd = [py, str(HERE / "gbm_6h.py"), "-c", str(paths[cname]), "--checkpoint", str(runs / cname / "checkpoint.pt"),
               "--live-compatible", "--out", str(out), "--bootstrap", "200"]  # fmt: skip
        if a.dry_run:
            print("$ " + " ".join(cmd))
        elif (out / pred_file(dataset)).exists():
            print(f"\n{cname}: reusing {out}")
        elif run(cmd) != 0:
            return 1

    # 5. verdict ------------------------------------------------------------------------------------------------
    if "verdict" not in steps or a.dry_run:
        return 0
    alpha = 0.05 / max(1, len(cands))
    print(
        f"\n=== Candidates against the live model on {dataset} (paired block bootstrap, alpha {alpha:.3f}) ==="
    )
    winners, compared = [], 0
    for cname in cands:
        res = compare(cur, runs / f"{cname}_gbm_live", a.bootstrap, alpha, dataset=dataset)
        if isinstance(res, str):
            print(f"\n{cname}: no comparison: {res}")
            continue
        compared += 1
        v = verdict(res)
        print(f"\n{cname}: {v.upper()} ({res['windows']} windows)")
        print("| metric | live model | candidate | difference | interval |\n|---|---|---|---|---|")
        for k in ("AUC", "Brier", "TSS", "POD", "FAR"):
            r = res[k]
            lo, hi = r["ci95"]
            print(
                f"| {k} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} | {lo:+.4f} to {hi:+.4f} |"
            )
        if v == "better":
            winners.append((res["AUC"]["diff"], cname))
    print()
    if not compared:
        print(
            "RESULT: nothing could be compared (see the messages above). The live model is unchanged."
        )
        return 0
    if not winners:
        print(
            "RESULT: keep the live model. No candidate was shown to be better; that is a result, not a failure."
        )
        return 0
    best = max(winners)[1]
    print(f"RESULT: {best} is better than the live model. To put it live:\n")
    print(f"  cp runs/{best}/checkpoint.pt models/flarenet.pt")
    print(f"  uv run python scripts/gbm_6h.py -c {paths[best]} --checkpoint models/flarenet.pt \\")
    print("      --live-compatible --install models/gbm6h_live.pkl")
    print("  git add -A && git commit -m 'promote the challenger 6 h model' && git push\n")
    print(
        "Then send me the printed table: the dashboard numbers (frontend/src/lib/validation.js) describe the old model."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

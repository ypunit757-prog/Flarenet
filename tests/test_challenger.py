"""The challenger pipeline's decision rule: replace the live model only on paired evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from challenger_6h import DATASET, compare, derive_config, main, pred_file, verdict  # noqa: E402


def _res(auc, brier):
    return {"AUC": {"ci95": list(auc)}, "Brier": {"ci95": list(brier)}}


def test_verdict_needs_better_ranking_and_no_worse_probabilities():
    assert verdict(_res((0.004, 0.02), (-0.003, -0.001))) == "better"
    assert (
        verdict(_res((0.004, 0.02), (-0.003, 0.002))) == "no established difference"
    )  # worse Brier possible
    assert (
        verdict(_res((-0.002, 0.02), (-0.003, -0.001))) == "no established difference"
    )  # AUC gain not shown
    assert verdict(_res((-0.03, -0.004), (0.001, 0.004))) == "worse"
    assert verdict(_res((float("nan"), float("nan")), (-1, -1))) == "no established difference"


def _make_run(root: Path, name: str, y, p, t, thr=0.3, with_stack=True):
    d = root / name
    d.mkdir(parents=True)
    df = pd.DataFrame({"t_end": t, "y": y, "p_cnn": p, "p_gbm": p})
    if with_stack:
        df["p_stack"] = p
    df.to_csv(d / pred_file(DATASET), index=False)
    (d / "gbm6h_report.json").write_text(json.dumps({"thresholds_from_validation": {"stack": thr}}))
    return d


def test_compare_finds_a_real_improvement_and_refuses_mismatched_runs(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    t = np.arange(n) * 60.0
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z - 1)))).astype(int)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    cur = _make_run(tmp_path, "cur", y, sig(z + rng.normal(scale=1.2, size=n)), t)
    good = _make_run(tmp_path, "good", y, sig(z + rng.normal(scale=0.2, size=n)), t)
    same = _make_run(tmp_path, "same", y, sig(z + rng.normal(scale=1.2, size=n)), t)
    res = compare(cur, good, n_boot=80, alpha=0.05)
    assert isinstance(res, dict) and res["windows"] == n and verdict(res) == "better"
    assert (
        verdict(compare(cur, same, n_boot=80, alpha=0.05)) != "better"
    )  # equal skill must not be promoted

    other = _make_run(tmp_path, "other", 1 - y, sig(z), t)
    assert "not scored on the same windows" in compare(cur, other, 20, 0.05)
    nostack = _make_run(tmp_path, "nostack", y, sig(z), t, with_stack=False)
    assert "no positive weight" in compare(cur, nostack, 20, 0.05)
    assert "has no" in compare(cur, tmp_path / "missing", 20, 0.05)


def test_derived_configs_keep_data_settings_and_only_change_what_they_should():
    base = {
        "run_name": "r",
        "data": {"split": "chrono"},
        "training": {"lr": 0.001, "calibrate": "none"},
    }
    ens = derive_config(base, "r_ens5", 5)
    assert ens["run_name"] == "r_ens5" and ens["training"]["ensemble_size"] == 5
    assert ens["training"]["calibrate"] == "platt" and ens["data"] == base["data"]
    assert (
        base["training"]["calibrate"] == "none" and "ensemble_size" not in base["training"]
    )  # input untouched


def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"run_name": "live", "data": {"split": "chrono"}}))
    assert (
        main(["-c", str(cfg), "--runs", str(tmp_path / "runs"), "--dry-run", "--ensemble", "3"])
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "tune_6h.py" in out
        and "live_ens3" in out
        and "gbm_6h.py" in out
        and "--live-compatible" in out
    )
    assert not list((tmp_path / "runs").glob("**/checkpoint.pt"))

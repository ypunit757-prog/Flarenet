"""The challenger pipeline's decision rule: replace the live model only on paired evidence."""

from __future__ import annotations

import datetime as dt
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


def test_derived_configs_copy_the_dates_that_yaml_reads_for_start_and_end():
    """`start: 2024-03-01` arrives as a datetime.date; a JSON round trip of the config crashed on it."""
    base = yaml.safe_load("sources:\n  - {name: a, params: {start: 2024-03-01, end: 2024-06-30}}\n")
    assert isinstance(base["sources"][0]["params"]["start"], dt.date)
    out = derive_config(base, "r_tuned")
    assert out["sources"][0]["params"]["start"] == dt.date(2024, 3, 1)
    assert out["sources"] is not base["sources"]  # a real copy
    again = yaml.safe_load(yaml.safe_dump(out, sort_keys=False))
    assert again["sources"][0]["params"]["end"] == dt.date(2024, 6, 30)


def test_whole_pipeline_runs_from_a_tuned_config_with_dates_to_a_verdict(
    tmp_path, monkeypatch, capsys
):
    """Stages 2-5 end to end with the training steps faked: this is the path a finished tuning run takes."""
    import challenger_6h

    runs = tmp_path / "runs"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run_name: live\n"
        "data: {split: chrono}\n"
        "sources:\n"
        "  - {name: goes16, role: train, params: {start: 2017-03-01, end: 2025-04-07}}\n"
        f"  - {{name: {DATASET}, role: eval, params: {{start: 2025-04-08, end: 2026-09-30}}}}\n"
    )
    rng = np.random.default_rng(1)
    n = 3000
    t = np.arange(n) * 60.0
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z - 1)))).astype(int)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    _make_run(runs, "live_gbm_live", y, sig(z + rng.normal(scale=1.2, size=n)), t)

    tune = runs / "tune_6h"  # what a finished search leaves behind
    tune.mkdir(parents=True)
    (tune / "best_trial.json").write_text(json.dumps({"trial": 7, "overrides": {}}))
    best = yaml.safe_load(cfg.read_text()) | {"run_name": "live"}
    best["training"] = {"lr": 0.0005, "calibrate": "platt"}
    (tune / "best_config.yaml").write_text(yaml.safe_dump(best, sort_keys=False))

    def fake_run(cmd):
        text = " ".join(cmd)
        if "tune_6h.py" in text:
            return 0
        if "gbm_6h.py" in text:  # a stacked prediction file for the candidate
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"t_end": t, "y": y, "p_stack": sig(z + rng.normal(scale=0.2, size=n))}
            ).to_csv(out / pred_file(DATASET), index=False)
            (out / "gbm6h_report.json").write_text(
                json.dumps({"thresholds_from_validation": {"stack": 0.3}})
            )
            return 0
        loaded = yaml.safe_load(Path(cmd[-1]).read_text())  # `... run -c <config>`: it must load
        ck = runs / loaded["run_name"] / "checkpoint.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(b"x")
        return 0

    monkeypatch.setattr(challenger_6h, "run", fake_run)
    rc = main(["-c", str(cfg), "--runs", str(runs), "--ensemble", "0", "--bootstrap", "60"])
    out = capsys.readouterr().out
    assert rc == 0
    tuned_cfg = yaml.safe_load((runs / "challengers" / "live_tuned.yaml").read_text())
    assert tuned_cfg["sources"][0]["params"]["start"] == dt.date(2017, 3, 1)
    assert tuned_cfg["run_name"] == "live_tuned" and tuned_cfg["training"]["lr"] == 0.0005
    assert "Candidates against the live model" in out and "live_tuned: BETTER" in out
    assert "cp runs/live_tuned/checkpoint.pt models/flarenet.pt" in out


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


def test_long_running_scripts_switch_on_progress_logging(monkeypatch, tmp_path):
    """Run outside the CLI they showed nothing for hours, which looks exactly like a hang."""
    import gbm_6h
    import pytest
    import tune_6h

    calls = []
    monkeypatch.setattr(gbm_6h, "setup_logging", lambda level="INFO": calls.append("gbm"))
    monkeypatch.setattr(tune_6h, "setup_logging", lambda level="INFO": calls.append("tune"))
    assert gbm_6h.main(["-c", "x.yaml", "--install", str(tmp_path / "p.pkl")]) == 2
    with pytest.raises(FileNotFoundError):
        tune_6h.main(["-c", str(tmp_path / "missing.yaml")])
    assert calls == ["gbm", "tune"]

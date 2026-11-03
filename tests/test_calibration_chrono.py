"""Chronological split, Platt calibration and alert levels of the 6-hour model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import tiny_cfg

from flarenet import orchestration as orch
from flarenet.core.schema import WindowSet
from flarenet.pipelines.evaluation import calibration as cal
from flarenet.pipelines.features.split import chrono_split
from flarenet.pipelines.training.checkpoint import load_checkpoint


def _ws(n: int = 4000, step: float = 60.0) -> WindowSet:
    t = np.arange(n) * step
    return WindowSet(
        X=np.zeros((n, 2, 8), dtype=np.float32), y_now=np.zeros(n, dtype=np.int64),
        y_fc=np.zeros(n, dtype=np.float32), t_end=t.astype(float),
        missing_frac=np.zeros(n, dtype=np.float32), channels=["a", "b"],
    )  # fmt: skip


def test_chrono_split_is_ordered_and_leaves_a_gap():
    ws = _ws()
    gap = 449.0 + 360.0
    sp = chrono_split(ws, 0.70, 0.15, gap)
    tr, va, te = sp["train"].t_end, sp["val"].t_end, sp["test"].t_end
    assert tr.max() < va.min() < va.max() < te.min()  # strictly in time order
    assert va.min() - tr.max() >= gap  # no training label reaches into validation
    assert te.min() - va.max() >= gap
    assert len(te) > 0 and len(va) > 0 and len(tr) > len(va)
    assert abs(len(te) / len(ws) - 0.15) < 0.01  # test keeps its full share (never cut)


def test_chrono_split_ignores_input_order():
    ws = _ws(2000)
    rev = ws.subset(np.arange(len(ws))[::-1])
    a, b = chrono_split(ws, 0.7, 0.15, 100.0), chrono_split(rev, 0.7, 0.15, 100.0)
    for k in ("train", "val", "test"):
        assert np.array_equal(a[k].t_end, b[k].t_end)


def test_platt_fixes_probabilities_that_were_trained_on_rebalanced_batches():
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.normal(size=n)
    true_p = 1 / (1 + np.exp(-(1.5 * z - 2.5)))  # real flare frequency is low (about 10 %)
    y = (rng.random(n) < true_p).astype(int)
    raw = 1 / (1 + np.exp(-(1.5 * z + 0.0)))  # a model trained on 50/50 batches says about 50 %
    c = cal.fit_platt(y, raw)
    assert c is not None and c["a"] > 0
    out = cal.apply_calibration(c, raw)
    assert abs(raw.mean() - y.mean()) > 0.3 and abs(out.mean() - y.mean()) < 0.02
    assert np.mean((out - y) ** 2) < np.mean((raw - y) ** 2)
    order = np.argsort(raw)
    assert np.all(np.diff(out[order]) >= -1e-12)  # monotone: ranking and AUC are unchanged
    assert cal.apply_calibration(None, raw) is raw or np.array_equal(
        cal.apply_calibration(None, raw), raw
    )


def test_platt_refuses_when_it_cannot_be_fitted_or_would_reverse_the_ranking():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    assert cal.fit_platt(np.zeros(500, dtype=int), p) is None  # no flare windows
    y = (rng.random(500) < 1 - p).astype(int)  # anti-informative scores
    assert cal.fit_platt(y, p, min_positives=5) is None


def test_alert_levels_trade_catches_for_false_alarms():
    rng = np.random.default_rng(2)
    n = 30000
    y = (rng.random(n) < 0.1).astype(int)
    p = np.clip(rng.normal(0.25 + 0.35 * y, 0.15), 0.001, 0.999)
    lv = cal.alert_levels(y, p, sensitive_pod=0.9, strict_far=0.5)
    bal, sen, strict = lv["balanced"], lv["sensitive"], lv["strict"]
    assert sen["POD"] >= 0.9 and strict["FAR"] <= 0.5
    assert sen["threshold"] <= bal["threshold"] <= strict["threshold"]
    assert sen["POD"] >= bal["POD"] >= strict["POD"]
    assert strict["FAR"] <= bal["FAR"] <= sen["FAR"]
    # a model that cannot tell flare windows apart has no threshold within the false-alarm budget: None, not faked
    flat = cal.alert_levels(y, np.full(n, 0.3), strict_far=0.5)
    assert flat["strict"] is None
    # when most windows are flare windows, "alert on everything" is not offered as a level
    busy = (rng.random(n) < 0.9).astype(int)
    assert cal.alert_levels(busy, np.full(n, 0.5))["strict"] is None


@pytest.mark.slow
def test_chrono_run_stores_calibration_and_levels_and_evaluates_with_them(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=3)
    cfg.data.split = "chrono"
    cfg.data.train_frac, cfg.data.val_frac = 0.6, 0.2
    cfg.training.calibrate = "platt"
    cfg.sources[0].params = {
        "n_days": 120
    }  # the synthetic Sun is flare-heavy: needs enough quiet windows
    cfg.validate()
    orch.run_all(cfg)
    lm = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    assert lm.calibration and lm.calibration["kind"] == "platt" and lm.calibration["a"] > 0
    assert lm.alert_levels and set(lm.alert_levels) == {"balanced", "sensitive", "strict"}
    import json

    rep = json.loads((cfg.run_dir / "metrics_report.json").read_text())
    assert rep["calibration"]["kind"] == "platt"
    res = next(iter(rep["results"].values()))
    assert "balanced" in res["alert_levels"] and 0 <= res["alert_levels"]["balanced"]["POD"] <= 1
    assert "Alert levels" in (cfg.run_dir / "report.md").read_text()
    assert any(
        "latest 20%" in c and "later period" in c for c in rep["caveats"]
    )  # says what the test really is


def test_old_checkpoints_without_calibration_still_load_and_score_unchanged(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=2)  # defaults: block split, no calibration
    orch.run_all(cfg)
    lm = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    assert lm.calibration is None
    assert 0 < lm.threshold < 1


def test_compare_runs_prints_both_runs_on_the_external_dataset(tmp_path, capsys):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_runs import main

    def res(auc, tss, far, brier, levels=None):
        return {
            "kind": "external", "AUC": auc, "TSS": tss, "HSS": 0.3, "POD": 0.8, "FAR": far,
            "confidence_95": {"AUC": [auc - 0.02, auc + 0.02], "TSS": [tss - 0.03, tss + 0.03]},
            "calibration": {"brier": brier, "ece": 0.05}, "lead_time": {"n_hits": 300, "n_events": 393},
            "alert_levels": levels or {},
        }  # fmt: skip

    for name, calib, r in (
        ("a", None, res(0.81, 0.44, 0.75, 0.2)),
        ("b", {"kind": "platt"}, res(0.80, 0.43, 0.6, 0.1, {"strict": {"threshold": 0.4, "POD": 0.5, "FAR": 0.4, "TSS": 0.4}})),
    ):  # fmt: skip
        d = tmp_path / name
        d.mkdir()
        (d / "metrics_report.json").write_text(json.dumps({
            "alert_threshold": 0.48, "calibration": calib,
            "results": {"goes18_test": r, "goes16_all": {**r, "kind": "test"}},
        }))  # fmt: skip
    assert main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert (
        "goes18_test" in out and "goes16_all" not in out
    )  # only the dataset neither run trained on
    assert "calibrated" in out and "0.810 [0.79, 0.83]" in out and "strict" in out


def test_paired_bootstrap_sees_a_small_gain_that_separate_intervals_would_hide():
    from flarenet.pipelines.evaluation.paired import paired_block_bootstrap, reading

    rng = np.random.default_rng(5)
    n = 6000
    t = np.arange(n) * 60.0
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * z - 1.5)))).astype(int)
    pa = 1 / (
        1 + np.exp(-(1.2 * z + rng.normal(scale=1.0, size=n) - 1.5))
    )  # B sees the same signal, less noise
    pb = 1 / (1 + np.exp(-(1.2 * z + 0.4 * rng.normal(size=n) - 1.5)))
    res = paired_block_bootstrap(y, pa, pb, t, 0.2, 0.2, block_hours=24, n=300, seed=1)
    assert (
        res["AUC"]["diff"] > 0
        and res["AUC"]["ci95"][0] > 0
        and reading("AUC", res["AUC"]) == "B better"
    )
    assert res["Brier"]["diff"] < 0 and reading("Brier", res["Brier"]) == "B better"
    same = paired_block_bootstrap(y, pa, pa, t, 0.2, 0.2, block_hours=24, n=200, seed=1)
    assert same["AUC"]["diff"] == 0 and reading("AUC", same["AUC"]) == "no established difference"
    with pytest.raises(ValueError):
        paired_block_bootstrap(y, pa[:-1], pb, t)


@pytest.mark.slow
def test_tuner_scores_validation_only_and_writes_a_runnable_config(tmp_path):
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from tune_6h import search

    cfg = tiny_cfg(tmp_path, epochs=2)
    cfg.data.split = "chrono"
    cfg.data.train_frac, cfg.data.val_frac = 0.6, 0.2
    cfg.sources[0].params = {"n_days": 120}
    cfg.validate()
    space = {"training.lr": [5e-4, 1e-3], "model.dropout": [0.2, 0.4]}
    out = tmp_path / "tune"
    table, best_path = search(
        cfg, {"seed": 1}, out, trials=2, confirm=1, confirm_seeds=1, space=space
    )
    rows = pd.read_csv(out / "trials.csv")
    assert set(rows["trial"]) == {0, 1, 2}
    assert rows["val_auc"].between(0, 1).all() and (rows["val_brier"] > 0).all()
    # the baseline and the confirmed trial were re-run with a second seed; the other trial was not
    assert (rows.groupby("trial")["seed"].count() >= 1).all() and rows["seed"].nunique() == 2
    assert "Tuning of" in (out / "tuning_report.md").read_text()
    best = yaml.safe_load(best_path.read_text())
    assert best["run_name"] == "t_tuned" and best["training"]["calibrate"] == "platt"
    from flarenet.core.config import load_config

    load_config(best_path)  # the written file loads and validates
    # resuming does no new training
    n = len(rows)
    search(cfg, {"seed": 1}, out, trials=2, confirm=1, confirm_seeds=1, resume=True, space=space)
    assert len(pd.read_csv(out / "trials.csv")) == n


def _fake_run(tmp_path, name, p, y, t_end, thr, with_csv=True):
    import json

    d = tmp_path / name
    d.mkdir()
    res = {
        "kind": "external", "AUC": 0.8, "TSS": 0.4, "HSS": 0.3, "POD": 0.8, "FAR": 0.6,
        "confidence_95": {}, "calibration": {"brier": 0.1, "ece": 0.05}, "lead_time": {}, "alert_levels": {},
    }  # fmt: skip
    (d / "metrics_report.json").write_text(
        json.dumps({"alert_threshold": thr, "calibration": None, "results": {"goes18_test": res}})
    )
    if with_csv:
        pd.DataFrame({"t_end": t_end, "time_utc": "x", "y": y, "p": p}).to_csv(
            d / "window_predictions_goes18_test.csv", index=False
        )
    return d


def test_compare_runs_paired_flag_prints_the_paired_table_or_says_why_not(tmp_path, capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_runs import main

    rng = np.random.default_rng(3)
    n = 3000
    t = np.arange(n) * 60.0
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z - 1)))).astype(int)
    pa = 1 / (1 + np.exp(-(z + rng.normal(scale=1.0, size=n))))
    pb = 1 / (1 + np.exp(-(z + rng.normal(scale=0.3, size=n))))  # the better model
    a = _fake_run(tmp_path, "a", pa, y, t, 0.4)
    b = _fake_run(tmp_path, "b", pb, y, t, 0.4)
    assert main([str(a), str(b), "--paired", "--bootstrap", "60"]) == 0
    out = capsys.readouterr().out
    assert "b minus a on the same 3000 windows" in out and "| AUC |" in out and "B better" in out

    c = _fake_run(
        tmp_path, "c", pb, y, t, 0.4, with_csv=False
    )  # a run scored before predictions were saved
    assert main([str(a), str(c), "--paired"]) == 0
    assert "skipped: missing" in capsys.readouterr().out
    assert main([str(a), str(b)]) == 0  # without the flag there is no paired section
    assert "minus" not in capsys.readouterr().out

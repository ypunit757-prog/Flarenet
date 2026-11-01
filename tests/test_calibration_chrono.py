"""Chronological split, Platt calibration and alert levels of the 6-hour model."""

from __future__ import annotations

import numpy as np
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

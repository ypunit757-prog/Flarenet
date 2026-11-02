"""The hourly tree model and the stack: no look-ahead in the features, honest stacking, an end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import tiny_cfg

from flarenet import orchestration as orch
from flarenet.daily.features import events_table
from flarenet.gbm6h import features as F
from flarenet.gbm6h.pipeline import fit_and_evaluate
from flarenet.gbm6h.stack import Stack, fit_stack
from flarenet.pipelines.training.checkpoint import load_checkpoint


def _synthetic_bundle(tmp_path, days=60):
    cfg = tiny_cfg(tmp_path, epochs=1)
    cfg.sources[0].params = {"n_days": days}
    cfg.validate()
    ctx = orch.run_features(cfg)
    return cfg, ctx, ctx.state["bundles"]["synthetic"]


def test_features_use_nothing_after_the_window_ends(tmp_path):
    cfg, ctx, b = _synthetic_bundle(tmp_path)
    lf, ev = F.prepare(b)
    i = 40 * 1440 + 300
    base = F.window_features(i, lf, ev, b.t0_unix, 360)

    lf2 = lf.copy()
    lf2[i + 1 :] = 3.0  # an enormous flux after the window
    t_issue = b.t0_unix + i * 60.0
    fake = pd.DataFrame(
        {"onset": [t_issue + 600.0], "peak": [t_issue + 1200.0], "cls": [4], "lpk": [-3.0]}
    )
    ev2 = pd.concat([ev, fake]).sort_values(
        "peak", ignore_index=True
    )  # an X flare that has not happened yet
    again = F.window_features(i, lf2, ev2, b.t0_unix, 360)
    assert base.keys() == again.keys()
    for k in base:
        assert np.allclose(base[k], again[k], equal_nan=True), k


def test_a_flare_that_just_peaked_is_not_yet_in_the_history(tmp_path):
    cfg, ctx, b = _synthetic_bundle(tmp_path)
    lf, ev = F.prepare(b)
    i = 40 * 1440
    t_issue = b.t0_unix + i * 60.0
    ev0 = ev[ev["peak"] < t_issue - 7 * 86400.0]  # a quiet recent past
    just = pd.DataFrame(
        {
            "onset": [t_issue - 1500.0],
            "peak": [t_issue - 120.0 + F.DETECT_DELAY_S],
            "cls": [4],
            "lpk": [-3.5],
        }
    )
    just_in = pd.DataFrame(
        {
            "onset": [t_issue - 5000.0],
            "peak": [t_issue - 3000.0 + F.DETECT_DELAY_S],
            "cls": [4],
            "lpk": [-3.5],
        }
    )
    f0 = F.window_features(i, lf, ev0, b.t0_unix, 360)
    f_recent = F.window_features(i, lf, pd.concat([ev0, just], ignore_index=True), b.t0_unix, 360)
    f_older = F.window_features(i, lf, pd.concat([ev0, just_in], ignore_index=True), b.t0_unix, 360)
    assert f_recent["h_nX_24"] == f0["h_nX_24"]  # peaked 2 min ago: not yet known to be a peak
    assert f_older["h_nX_24"] == f0["h_nX_24"] + 1  # peaked 50 min ago: known
    assert f_older["h_nM_3"] == f0["h_nM_3"] + 1 and f_older["h_since_X"] < f0["h_since_X"]


def test_rotation_features_look_exactly_one_rotation_back(tmp_path):
    cfg, ctx, b = _synthetic_bundle(tmp_path)
    lf, ev = F.prepare(b)
    i = 40 * 1440
    t_issue = b.t0_unix + i * 60.0
    quiet = ev[(ev["onset"] < t_issue - 29 * 86400.0) | (ev["onset"] > t_issue - 25 * 86400.0)]
    d0 = t_issue - 27 * 86400.0
    m = pd.DataFrame({"onset": [d0 + 3600.0], "peak": [d0 + 4000.0], "cls": [3], "lpk": [-4.5]})
    f0 = F.window_features(i, lf, quiet, b.t0_unix, 360)
    f1 = F.window_features(i, lf, pd.concat([quiet, m], ignore_index=True), b.t0_unix, 360)
    assert f1["rot_nM_slot"] == f0["rot_nM_slot"] + 1 and f1["rot_nM_wide"] == f0["rot_nM_wide"] + 1
    early = F.window_features(5 * 1440, lf, ev, b.t0_unix, 360)  # less than a rotation of history
    assert np.isnan(early["rot_nM_slot"])


def test_stack_weights_both_models_when_both_carry_information():
    rng = np.random.default_rng(1)
    n = 6000
    a, g = rng.normal(size=n), rng.normal(size=n)  # two independent pieces of evidence
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * a + 1.0 * g - 1.5)))).astype(int)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    st = fit_stack(y, sig(a), sig(g))
    assert st is not None and st.a_cnn > 0.5 and st.a_gbm > 0.5 and 0.2 < st.share_gbm() < 0.8
    from flarenet.pipelines.evaluation.metrics import safe_auc

    assert (
        safe_auc(y, st.predict(sig(a), sig(g)))
        > max(safe_auc(y, sig(a)), safe_auc(y, sig(g))) + 0.02
    )


def test_stack_refuses_a_model_that_works_against_the_other():
    rng = np.random.default_rng(2)
    n = 4000
    a = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * a - 1)))).astype(int)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    assert (
        fit_stack(y, sig(a), sig(-a + 0.0 * rng.normal(size=n))) is None
    )  # negative weight: no blend
    assert fit_stack(y[:30], sig(a[:30]), sig(a[:30])) is None  # too few windows
    assert isinstance(Stack(1.0, 1.0, 0.0).predict([0.5], [0.5]), np.ndarray)


@pytest.mark.slow
def test_fit_and_evaluate_end_to_end_on_the_synthetic_sun(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=2)
    cfg.data.split = "chrono"
    cfg.data.train_frac, cfg.data.val_frac = 0.6, 0.2
    cfg.training.calibrate = "platt"
    cfg.sources[0].params = {"n_days": 120}
    cfg.validate()
    orch.run_all(cfg)
    lm = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    ctx = orch.run_features(cfg)
    out = tmp_path / "gbm"
    rep = fit_and_evaluate(
        cfg,
        ctx.state["splits"],
        ctx.state["bundles"],
        lm,
        torch.device("cpu"),
        out,
        train_stride=60,
        n_boot=30,
    )
    for f in (
        "gbm6h_report.md",
        "gbm6h_report.json",
        "gbm6h.pkl",
        "gbm6h_predictions_synthetic.csv",
    ):
        assert (out / f).exists(), f
    d = next(iter(rep["datasets"].values()))
    assert {"cnn", "gbm"} <= set(d["metrics"]) and 0 <= d["metrics"]["gbm"]["AUC"] <= 1
    assert "paired_gbm_minus_cnn" in d
    on_disk = json.loads((out / "gbm6h_report.json").read_text())
    assert on_disk["thresholds_from_validation"].keys() >= {"cnn", "gbm"}
    assert "network alone" in (out / "gbm6h_report.md").read_text()
    preds = pd.read_csv(out / "gbm6h_predictions_synthetic.csv")
    assert {"t_end", "y", "p_cnn", "p_gbm"} <= set(preds.columns) and preds["p_gbm"].between(
        0, 1
    ).all()
    assert len(events_table(ctx.state["bundles"]["synthetic"])) > 0

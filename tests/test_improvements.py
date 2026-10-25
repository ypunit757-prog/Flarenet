"""Unit tests for the model/training improvements and the experiment harness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import tiny_cfg

from flarenet.core.config import Config, load_config
from flarenet.models import EnsembleModel, build_model, count_parameters
from flarenet.models.losses import class_weights
from flarenet.pipelines.evaluation.metrics import best_threshold, contingency, tss
from flarenet.pipelines.evaluation.scoring import infer
from flarenet.pipelines.features.normalise import normalise
from flarenet.pipelines.training.checkpoint import load_checkpoint, save_checkpoint
from flarenet.pipelines.training.trainer import (
    WeightEMA,
    forecast_sampling,
    selection_score,
    tune_nowcast_bias,
)

# ---- config ------------------------------------------------------------------------------------


def test_new_options_default_to_the_original_behaviour():
    c = Config()
    t = c.training
    assert (t.ema_decay, t.select_metric, t.forecast_balance) == (0.0, "tss", "fixed")
    assert (t.nowcast_weight_power, t.nowcast_weight_cap, t.nowcast_bias_tuning) == (
        0.5,
        6.0,
        False,
    )
    assert (t.threshold_smooth, t.ensemble_size) == (0, 1)
    assert c.data.train_stride_min is None and c.data.purge is False
    assert (c.model.nowcast_skip, c.model.nowcast_pool) == (False, "last")


@pytest.mark.parametrize(
    "override",
    ["training.select_metric=f1", "training.forecast_balance=half", "training.ema_decay=1.0",
     "training.ensemble_size=0", "model.nowcast_pool=first", "data.train_stride_min=90"],
)  # fmt: skip
def test_invalid_new_options_are_rejected(override):
    with pytest.raises(ValueError):
        load_config(None, [override]).validate()


def test_new_options_can_be_set_from_the_command_line():
    c = load_config(None, ["data.train_stride_min=15", "training.ema_decay=0.9",
                           "model.nowcast_skip=true", "data.purge=true"])  # fmt: skip
    c.validate()
    assert c.data.train_stride_min == 15 and c.training.ema_decay == 0.9
    assert c.model.nowcast_skip is True and c.data.purge is True


# ---- weight averaging --------------------------------------------------------------------------


def test_ema_tracks_weights_and_buffers_and_never_trains():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    ema = WeightEMA(model, 0.9)
    assert not any(p.requires_grad for p in ema.shadow.parameters())
    start = [p.clone() for p in ema.shadow.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        model[1].running_mean.fill_(5.0)
    for _ in range(200):
        ema.update(model)
    for s, p0, m in zip(ema.shadow.parameters(), start, model.parameters(), strict=True):
        assert not torch.allclose(s, p0)  # moved away from the start ...
        assert torch.allclose(s, m, atol=1e-3)  # ... and converged to the live weights
    assert torch.allclose(ema.shadow[1].running_mean, torch.full((3,), 5.0), atol=1e-3)
    assert (
        ema.shadow[1].num_batches_tracked == model[1].num_batches_tracked
    )  # integer buffer copied


def test_ema_warmup_makes_early_steps_follow_the_model_quickly():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = WeightEMA(model, 0.999)
    with torch.no_grad():
        model.weight.fill_(10.0)
    ema.update(model)
    # With the warm-up the first step uses decay (1+1)/(10+1) ~ 0.18, so the shadow jumps to ~8.2;
    # a fixed 0.999 would leave it at ~0.01.
    assert float(ema.shadow.weight) == pytest.approx(10 * (1 - 2 / 11), rel=1e-4)


# ---- selection score, sampling, class weights, thresholds ---------------------------------------


def test_selection_score():
    assert selection_score("tss", 0.6, 0.9) == 0.6
    assert selection_score("auc", 0.6, 0.9) == 0.9
    assert selection_score("tss_auc", 0.6, 0.9) == pytest.approx(0.75)
    assert selection_score("auc", 0.6, float("nan")) == 0.6  # single-class validation set


def test_forecast_sampling_fixed_vs_auto():
    cfg = Config()
    tc = cfg.training
    y = np.array(
        [1] * 83 + [0] * 17, dtype=np.float32
    )  # the synthetic sun: positives are the majority
    w_fixed, pw_fixed = forecast_sampling(y, tc)
    assert w_fixed[0] == tc.positive_oversample and pw_fixed == pytest.approx(17 / 83)
    tc.forecast_balance = "auto"
    w, pw = forecast_sampling(y, tc)
    drawn_pos = float((w * y).sum() / w.sum())
    assert drawn_pos == pytest.approx(0.5) and pw == 1.0
    rare = np.array([1] * 5 + [0] * 95, dtype=np.float32)  # the real sun: positives are rare
    w, _ = forecast_sampling(rare, tc)
    assert float((w * rare).sum() / w.sum()) == pytest.approx(0.5) and w[0] > w[-1]
    w, pw = forecast_sampling(np.ones(10, dtype=np.float32), tc)  # degenerate: no negatives at all
    assert np.all(w == tc.positive_oversample) and pw >= 0


def test_class_weights_power_and_cap():
    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    sqrt = class_weights(y, 3).numpy()
    assert np.allclose(sqrt, np.sqrt(100 / (3 * np.array([90, 8, 2]))).clip(None, 6.0))
    full = class_weights(y, 3, cap=100.0, power=1.0).numpy()
    assert np.allclose(full, 100 / (3 * np.array([90, 8, 2])))
    assert class_weights(y, 3, cap=5.0, power=1.0).numpy().max() == 5.0
    assert class_weights(np.array([0, 0, 1]), 3).numpy()[2] <= 6.0  # absent class stays finite


def test_best_threshold_unchanged_without_smoothing_and_robust_with_it():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 400), 0, 1)
    thr, val = best_threshold(y, p)
    grid = np.linspace(0.01, 0.99, 249)
    curve = np.array([tss(*contingency(y, (p >= g).astype(int))) for g in grid])
    assert val == curve.max() and thr == grid[int(np.argmax(curve))]
    thr_s, val_s = best_threshold(y, p, smooth=8)
    assert val_s <= val  # smoothing can only lower the (unsmoothed) TSS at the picked point
    # a lone spike (high TSS only at one threshold) loses against a broad plateau once smoothed
    y2 = np.array([1] * 50 + [0] * 50)
    p2 = np.concatenate([np.linspace(0.45, 0.95, 50), np.linspace(0.05, 0.55, 50)])
    t0, _ = best_threshold(y2, p2, smooth=0)
    t1, _ = best_threshold(y2, p2, smooth=10)
    assert 0.3 < t1 < 0.7 and 0.3 < t0 < 0.7


def test_tune_nowcast_bias_helps_a_suppressed_rare_class_and_keeps_class0_fixed():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    n = 600
    y = rng.choice(3, n, p=[0.7, 0.2, 0.1])
    logits = rng.normal(0, 0.6, (n, 3))
    logits[np.arange(n), y] += 1.2
    logits[:, 2] -= 1.5  # model systematically under-predicts class 2
    base = f1_score(y, logits.argmax(1), average="macro")
    bias = tune_nowcast_bias(logits, y)
    tuned = f1_score(y, (logits + bias).argmax(1), average="macro")
    assert bias[0] == 0.0 and bias[2] > 0 and tuned > base + 0.02
    flat = tune_nowcast_bias(np.tile([2.0, 0.0, 0.0], (30, 1)), np.zeros(30, dtype=int))
    assert np.all(flat == 0)  # nothing to improve -> no change


# ---- model variants ----------------------------------------------------------------------------


def test_default_model_is_unchanged_and_new_options_add_only_what_they_say():
    cfg = Config()
    base = build_model(cfg.model, 12)
    assert count_parameters(base) == 122_087
    cfg.model.nowcast_skip = True
    skip = build_model(cfg.model, 12)
    assert count_parameters(skip) == 122_087 + 24 * 64  # 2 x 12 extra inputs into a 64-unit layer
    x = torch.randn(4, 12, 90)
    for pool in ("last", "final_states"):
        cfg.model.nowcast_pool = pool
        m = build_model(cfg.model, 12).eval()
        now, fc, attn = m(x)
        assert now.shape == (4, 5) and fc.shape == (4,) and attn.shape[0] == 4
    # old checkpoints (no new keys) still load into the default architecture
    assert set(base.state_dict()) <= set(skip.state_dict())


def test_nowcast_skip_makes_the_class_depend_on_the_newest_minutes():
    cfg = Config()
    cfg.model.nowcast_skip = True
    m = build_model(cfg.model, 12).eval()
    x = torch.randn(2, 12, 90)
    x2 = x.clone()
    x2[:, :, -1] += 3.0  # change only the newest minute
    with torch.no_grad():
        a, b = m(x)[0], m(x2)[0]
    assert not torch.allclose(a, b)


# ---- ensemble + checkpoint ----------------------------------------------------------------------


def _members(n=3, seed=0):
    cfg = Config()
    out = []
    for i in range(n):
        torch.manual_seed(seed + i)
        out.append(build_model(cfg.model, 12).eval())
    return cfg, out


def test_ensemble_averages_probabilities():
    _, ms = _members()
    ens = EnsembleModel(ms).eval()
    x = torch.randn(6, 12, 90)
    with torch.no_grad():
        now, fc, _ = ens(x)
        outs = [m(x) for m in ms]
    exp_fc = torch.stack([torch.sigmoid(o[1]) for o in outs]).mean(0)
    exp_now = torch.stack([torch.softmax(o[0], -1) for o in outs]).mean(0)
    assert torch.allclose(torch.sigmoid(fc), exp_fc, atol=1e-5)
    assert torch.allclose(torch.softmax(now, -1), exp_now, atol=1e-5)
    with pytest.raises(ValueError):
        EnsembleModel([])


def test_checkpoint_roundtrip_for_ensemble_and_bias_and_old_format(tmp_path):
    cfg, ms = _members(2)
    mu, sg = np.zeros((1, 12, 1), np.float32), np.ones((1, 12, 1), np.float32)
    chans = [f"c{i}" for i in range(12)]
    ens = EnsembleModel(ms)
    save_checkpoint(
        tmp_path / "e.pt", ens, mu, sg, 0.4, cfg, chans, ["x"], now_bias=[0, 0.5, 1, 0, 0]
    )
    lm = load_checkpoint(tmp_path / "e.pt", "cpu")
    assert isinstance(lm.model, EnsembleModel) and len(lm.model.members) == 2
    assert np.allclose(lm.now_bias, [0, 0.5, 1, 0, 0]) and lm.threshold == pytest.approx(0.4)
    x = np.random.default_rng(0).normal(size=(5, 12, 90)).astype(np.float32)
    with torch.no_grad():
        want = ens(torch.from_numpy(normalise(x, mu, sg)))[1]
    _, _, prob = infer(lm.model, x, lm.mu, lm.sg, "cpu")
    assert np.allclose(prob, torch.sigmoid(want).numpy(), atol=1e-5)

    save_checkpoint(tmp_path / "s.pt", ms[0], mu, sg, 0.5, cfg, chans, ["x"])
    single = load_checkpoint(tmp_path / "s.pt", "cpu")
    assert single.now_bias is None and not isinstance(single.model, EnsembleModel)


def test_infer_applies_nowcast_bias():
    cfg, ms = _members(1)
    x = np.random.default_rng(0).normal(size=(20, 12, 90)).astype(np.float32)
    mu, sg = np.zeros((1, 12, 1), np.float32), np.ones((1, 12, 1), np.float32)
    plain, _, p1 = infer(ms[0], x, mu, sg, "cpu")
    forced, probs, p2 = infer(ms[0], x, mu, sg, "cpu", now_bias=[0, 0, 0, 0, 50.0])
    assert np.all(forced == 4) and np.allclose(probs.sum(1), 1) and np.allclose(p1, p2)
    assert not np.all(plain == 4)


# ---- data: train stride, eval grid, purge --------------------------------------------------------


def test_finer_train_stride_adds_train_windows_but_keeps_test_set_identical(tmp_path):
    from flarenet.orchestration import runner

    def test_end_times(**data):
        cfg = tiny_cfg(tmp_path / f"d{len(data)}{sorted(data.items())}")
        for k, v in data.items():
            setattr(cfg.data, k, v)
        cfg.validate()
        ctx = runner.run_features(cfg)
        sp = ctx.state["splits"]
        return sp["train"], sp["val"], sp["test"]["synthetic"]

    tr0, va0, te0 = test_end_times()
    tr1, va1, te1 = test_end_times(train_stride_min=15)
    assert len(tr1) > 3 * len(tr0)  # 4x finer stride, about 4x the training windows
    assert np.array_equal(te0.t_end, te1.t_end) and np.array_equal(va0.t_end, va1.t_end)
    assert np.array_equal(te0.X, te1.X) and np.array_equal(te0.y_fc, te1.y_fc)
    assert np.all((te1.t_end - 89) % 60 == 0) and np.any((tr1.t_end - 89) % 60 != 0)

    tr2, va2, te2 = test_end_times(purge=True)
    assert np.array_equal(te0.t_end, te2.t_end)  # purging never touches the test set
    assert len(tr2) < len(tr0) and len(va2) <= len(va0)
    gap = 89 + 360
    others = np.concatenate([va2.t_end, te2.t_end])
    assert np.abs(tr2.t_end[:, None] - others[None, :]).min() >= gap
    assert np.abs(va2.t_end[:, None] - te2.t_end[None, :]).min() >= gap


# ---- experiment harness -------------------------------------------------------------------------


def _fake_rows(variants, seeds, effect):
    rows = []
    for s in seeds:
        for v in variants:
            base = 0.5 + 0.05 * s  # seeds differ a lot, variants differ a little
            gain = effect.get(v, 0.0)
            rows.append({"variant": v, "seed": s, "AUC": base + gain, "TSS": base + gain, "HSS": base,
                         "brier": 0.2 - gain, "ece": 0.1, "macro_F1": 0.5, "accuracy": 0.8,
                         "recall_M": 0.4, "val_tss": 0.6, "epochs": 10, "seconds": 5.0})  # fmt: skip
    return rows


def test_summary_pairs_variants_with_the_baseline_over_the_same_seeds():
    from flarenet.experiment import summarise

    df = pd.DataFrame(
        _fake_rows(["baseline", "good", "bad"], [1, 2, 3, 4], {"good": 0.03, "bad": -0.02})
    )
    md = summarise(df, "baseline")
    assert "4 seeds" in md and "Paired difference to `baseline`" in md
    good = next(
        line for line in md.splitlines() if line.startswith("| good |") and "+0.030" in line
    )
    assert "(4/4)" in good  # all four seeds improved
    bad = next(line for line in md.splitlines() if line.startswith("| bad |") and "-0.020" in line)
    assert "(0/4)" in bad
    brier = [line for line in md.splitlines() if line.startswith("| good |") and "+0.030" in line][
        0
    ]
    assert brier.count("+0.030") >= 3  # AUC, TSS and Brier all count lower-is-better correctly
    assert summarise(pd.DataFrame(), "baseline").startswith("No results")


def test_experiment_file_validation(tmp_path):
    from flarenet.experiment import load_experiment

    good = tmp_path / "e.yaml"
    good.write_text("seeds: [1, 2]\nvariants:\n  a: []\n  b: ['training.epochs=3']\n")
    spec = load_experiment(good)
    assert spec["baseline"] == "a" and spec["variants"]["b"] == ["training.epochs=3"]
    (tmp_path / "bad.yaml").write_text("variants: {}\n")
    with pytest.raises(ValueError, match="variants"):
        load_experiment(tmp_path / "bad.yaml")
    (tmp_path / "bad2.yaml").write_text("baseline: zzz\nvariants:\n  a: []\n")
    with pytest.raises(ValueError, match="baseline"):
        load_experiment(tmp_path / "bad2.yaml")


def test_experiment_runner_resumes_and_selects_variants(tmp_path, monkeypatch):
    import flarenet.experiment as ex

    calls = []

    def fake_run_one(config_path, overrides, variant, seed, root, device, threads):
        calls.append((variant, seed))
        return _fake_rows([variant], [seed], {})[0]

    monkeypatch.setattr(ex, "run_one", fake_run_one)
    spec = {"seeds": [1, 2], "baseline": "a", "base_overrides": [],
            "variants": {"a": [], "b": ["x=1"], "c": ["y=2"]}}  # fmt: skip
    out = tmp_path / "exp"
    df = ex.run_experiment(spec, out, jobs=1, only=["b"])
    assert sorted(calls) == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]  # baseline always runs
    assert (out / "summary.md").exists() and (out / "results.csv").exists() and len(df) == 4
    calls.clear()
    df = ex.run_experiment(spec, out, jobs=1)  # resume: only the missing variant is run
    assert sorted(calls) == [("c", 1), ("c", 2)] and len(df) == 6
    with (out / "results.jsonl").open("a") as f:
        f.write('{"variant": "a", "seed": 9, "AUC":')  # half-written line from an interrupted run
    calls.clear()
    ex.run_experiment(spec, out, jobs=1)
    assert calls == []
    with pytest.raises(KeyError, match="Unknown variant"):
        ex.run_experiment(spec, out, jobs=1, only=["nope"])
    assert json.loads((out / "results.jsonl").read_text().splitlines()[0])["variant"]

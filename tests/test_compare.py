"""Model comparison: the scores are computed correctly, every model sees the same windows, nothing leaks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import tiny_cfg

from flarenet import orchestration as orch
from flarenet.compare.metrics import full_metrics, pr_points, roc_points
from flarenet.compare.models import LSTMClassifier, TableModel, fit_lstm, random_forest
from flarenet.compare.pipeline import compare_models, render
from flarenet.pipelines.training.checkpoint import load_checkpoint


def test_full_metrics_match_hand_counts():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.2, 0.1, 0.1, 0.1])
    m = full_metrics(y, p, 0.5)  # alerts: windows 0, 1 (flares) and 4 (quiet)
    assert m["confusion"] == [[5, 1], [2, 2]]  # [[TN, FP], [FN, TP]]
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(0.5)
    assert m["F1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m["TSS"] == pytest.approx(0.5 - 1 / 6)  # POD - POFD
    assert m["prevalence"] == pytest.approx(0.4) and 0 <= m["ROC_AUC"] <= 1
    assert m["Brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_full_metrics_are_nan_not_errors_for_degenerate_inputs():
    m = full_metrics([0, 0, 0, 0], [0.1, 0.2, 0.1, 0.3], 0.5)  # no flare at all, no alert
    assert np.isnan(m["recall"]) and np.isnan(m["ROC_AUC"]) and np.isnan(m["PR_AUC"])
    assert m["accuracy"] == 1.0 and m["confusion"] == [[4, 0], [0, 0]]
    assert len(roc_points([0, 0], [0.1, 0.2])[0]) == 2  # the diagonal
    assert len(pr_points([0, 0], [0.1, 0.2])[0]) == 0


def test_a_perfect_ranking_scores_perfectly_and_a_random_one_does_not():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    perfect = full_metrics(y, y * 0.8 + 0.1, 0.5)
    assert perfect["ROC_AUC"] == 1.0 and perfect["TSS"] == 1.0 and perfect["F1"] == 1.0
    rand = full_metrics(y, rng.random(4000), 0.5)
    assert abs(rand["ROC_AUC"] - 0.5) < 0.05 and abs(rand["TSS"]) < 0.08
    assert abs(rand["PR_AUC"] - 0.2) < 0.05  # a random ranking scores the prevalence


def test_lstm_learns_a_signal_and_keeps_the_best_epoch():
    rng = np.random.default_rng(0)
    n, c, t = 1200, 3, 30
    y = (rng.random(n) < 0.25).astype(int)
    X = rng.normal(size=(n, c, t)).astype(np.float32)
    X[y == 1, 0, -10:] += 1.5  # the flare windows end in a brighter channel 0
    mu, sg = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True) + 1e-6
    fitted = fit_lstm(X[:800], y[:800], X[800:], y[800:], mu, sg, "cpu", epochs=6, seed=1)
    p = fitted.predict_proba(X[800:])
    assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
    assert fitted.val_auc > 0.85 and 1 <= fitted.best_epoch <= 6
    assert isinstance(fitted.model, LSTMClassifier)
    assert np.allclose(p, fitted.predict_proba(X[800:]))  # evaluation is deterministic


def test_table_model_survives_missing_values_and_empty_columns():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(size=n)
    tab = pd.DataFrame(
        {"a": a, "b": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)), "empty": np.nan}
    )
    y = (a + 0.3 * rng.normal(size=n) > 0.8).astype(int)
    m = random_forest(seed=0, n_jobs=1).fit(tab, y)
    assert m.cols == ["a", "b"]  # the all-empty column carries nothing
    new = pd.DataFrame({"a": [3.0, -3.0], "b": [np.nan, np.nan], "empty": [np.nan, np.nan]})
    p = m.predict_proba(new)
    assert p[0] > p[1] and ((p >= 0) & (p <= 1)).all()
    assert isinstance(m, TableModel) and set(m.importances()) == {"a", "b"}
    assert abs(sum(m.importances().values()) - 1.0) < 1e-6


@pytest.mark.slow
def test_comparison_end_to_end_on_the_synthetic_sun(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=2)
    cfg.data.split = "chrono"
    cfg.data.train_frac, cfg.data.val_frac = 0.6, 0.2
    cfg.training.calibrate = "platt"
    cfg.sources[0].params = {"n_days": 120}
    cfg.validate()
    orch.run_all(cfg)
    lm = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    ctx = orch.run_features(cfg)
    out = tmp_path / "cmp"
    rep = compare_models(cfg, ctx.state["splits"], ctx.state["bundles"], lm, torch.device("cpu"), out,
                         train_stride=60, n_boot=20, lstm_epochs=2)  # fmt: skip
    for f in ("model_comparison.md", "model_comparison.json"):
        assert (out / f).exists(), f
    assert list(out.glob("roc_pr_*.png")) and list(out.glob("confusion_*.png"))
    d = next(iter(rep["datasets"].values()))
    # the three models that need no optional package ran, on the same windows
    assert {"cnn_bilstm", "lstm", "random_forest", "gradient_boosting"} <= set(d["metrics"])
    for m in d["metrics"].values():
        assert {
            "accuracy",
            "precision",
            "recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "TSS",
            "Brier",
        } <= set(m)
        (tn, fp), (fn, tp) = m["confusion"]
        assert tn + fp + fn + tp == d["windows"]  # one confusion matrix over the identical windows
    assert "lstm" in d["paired_vs_cnn_bilstm"] and "cnn_bilstm" not in d["paired_vs_cnn_bilstm"]
    preds = pd.read_csv(next(out.glob("model_comparison_predictions_*.csv")))
    assert (
        len(preds) == d["windows"]
        and preds.filter(like="p_").apply(lambda c: c.between(0, 1).all()).all()
    )
    on_disk = json.loads((out / "model_comparison.json").read_text())
    assert on_disk["thresholds_from_validation"].keys() >= {"cnn_bilstm", "lstm", "random_forest"}
    text = (out / "model_comparison.md").read_text()
    assert "Accuracy is not the headline" in text and text == render(on_disk)


def test_a_missing_optional_package_is_reported_not_fatal(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_xgboost(name, *a, **k):
        if name == "xgboost":
            raise ImportError("No module named 'xgboost'", name="xgboost")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    from flarenet.compare.models import xgboost_model

    with pytest.raises(ImportError):
        xgboost_model(np.array([0, 1, 0, 1]))


def test_thin_curve_keeps_ends_and_limits_points():
    from flarenet.compare.metrics import thin_curve

    x = np.linspace(0, 1, 5000)
    pts = thin_curve(x, x**2, n=60)
    assert 2 <= len(pts) <= 63
    assert pts[0] == [0.0, 0.0] and pts[-1] == [1.0, 1.0]
    assert all(a[0] <= b[0] for a, b in zip(pts, pts[1:], strict=False))
    assert thin_curve([], []) == []
    assert thin_curve([0.3, 0.3], [0.2, 0.2]) == [[0.3, 0.2], [0.3, 0.2]]


def test_curves_empty_for_one_class_and_valid_otherwise():
    from flarenet.compare.metrics import curves

    p = np.linspace(0, 1, 50)
    assert curves(np.zeros(50), p) == {"roc": [], "pr": []}
    c = curves((p > 0.7).astype(int), p)
    assert c["roc"][0] == [0.0, 0.0] and c["roc"][-1] == [1.0, 1.0] and len(c["pr"]) > 2


def test_json_writer_never_emits_nan():
    import json

    from flarenet.core import jsonsafe

    doc = {
        "a": float("nan"),
        "b": [np.float32(1.5), float("inf"), np.int64(3)],
        "c": np.array([1.0, np.nan]),
    }
    text = jsonsafe.dumps(doc)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [1.5, None, 3], "c": [1.0, None]}

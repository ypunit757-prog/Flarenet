"""Explainability: exact Shapley values, honest groups, contributions that add up to the forecast on the page."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import tiny_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_web import explain_block  # noqa: E402

from flarenet import orchestration as orch  # noqa: E402
from flarenet.explain.groups import CHANNEL_LABELS, TREE_GROUPS, channel_groups, tree_groups  # noqa: E402
from flarenet.explain.importance import (  # noqa: E402
    mean_abs_shapley_table,
    permutation_importance_table,
    permutation_importance_windows,
)
from flarenet.explain.live import describe, explain_for_export, explain_latest  # noqa: E402
from flarenet.explain.local import logit, network_effects, tree_effects  # noqa: E402
from flarenet.explain.shapley import masks, shapley  # noqa: E402
from flarenet.gbm6h import features as F  # noqa: E402
from flarenet.gbm6h.features import set_features  # noqa: E402
from flarenet.gbm6h.live import file_sha256, load_live_stack, package  # noqa: E402
from flarenet.gbm6h.model import HourlyGBM  # noqa: E402
from flarenet.gbm6h.pipeline import LIVE_COMPATIBLE_DROP, drop_features  # noqa: E402
from flarenet.gbm6h.stack import Stack  # noqa: E402
from flarenet.pipelines.features.channels import CHANNEL_NAMES  # noqa: E402
from flarenet.pipelines.inference.predict import forecast_bundle  # noqa: E402
from flarenet.pipelines.training.checkpoint import load_checkpoint  # noqa: E402

sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))  # noqa: E731


# ---------------------------------------------------------------- Shapley values


def test_shapley_of_an_additive_game_is_each_players_own_term():
    w = np.array([1.0, -2.0, 0.5, 0.0])
    phi = shapley(masks(4) @ w, 4)
    assert np.allclose(phi, w)


def test_shapley_splits_an_interaction_equally_and_always_adds_up():
    assert np.allclose(
        shapley(np.array([0, 0, 0, 1.0]), 2), [0.5, 0.5]
    )  # value only when both are present
    rng = np.random.default_rng(0)
    v = rng.normal(size=2**5)
    assert shapley(v, 5).sum() == pytest.approx(v[-1] - v[0])  # efficiency
    with pytest.raises(ValueError):
        shapley(np.zeros(5), 3)
    with pytest.raises(ValueError):
        masks(30)


# ---------------------------------------------------------------- groups


def test_every_feature_the_tree_model_can_see_is_in_exactly_one_group(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=1)
    cfg.sources[0].params = {"n_days": 60}
    cfg.validate()
    b = orch.run_features(cfg).state["bundles"]["synthetic"]
    lf, ev = F.prepare(b)
    cols = list(F.window_features(40 * 1440 + 300, lf, ev, b.t0_unix, 360))
    groups = tree_groups(cols)
    assert "other" not in groups, f"features without a group: {groups.get('other')}"
    placed = [c for _, feats in groups.values() for c in feats]
    assert sorted(placed) == sorted(cols)  # nothing lost, nothing counted twice
    allf = [f for _, feats in TREE_GROUPS.values() for f in feats]
    assert len(allf) == len(set(allf))  # no feature in two groups


def test_unknown_feature_goes_to_other_and_channels_are_all_labelled():
    assert tree_groups(["f_cur", "brand_new"])["other"][1] == ["brand_new"]
    assert set(CHANNEL_NAMES) <= set(CHANNEL_LABELS)
    assert list(channel_groups(CHANNEL_NAMES)) == CHANNEL_NAMES


# ---------------------------------------------------------------- local effects


def _toy_tree(coef):
    def predict(tab: pd.DataFrame) -> np.ndarray:
        return sig(tab.to_numpy(float) @ coef - 1.0)

    return predict


def test_tree_effects_recover_a_logistic_models_terms_and_add_up():
    coef = np.array([2.0, 0.5, 0.0])
    row = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    ref = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0})
    groups = {"ga": ("A", ["a"]), "gb": ("B", ["b"]), "gc": ("C", ["c"])}
    r = tree_effects(_toy_tree(coef), row, ref, groups)
    assert r["effects"]["ga"] == pytest.approx(2.0) and r["effects"]["gb"] == pytest.approx(1.0)
    assert r["effects"]["gc"] == pytest.approx(0.0)  # the model ignores c, so c explains nothing
    assert sum(r["effects"].values()) == pytest.approx(logit(r["p_row"]) - logit(r["p_ref"]))
    assert r["p_ref"] == pytest.approx(float(sig(-1.0)))


def test_network_effects_credit_the_channel_the_network_reads():
    def proba(w: np.ndarray) -> np.ndarray:
        return sig(1.5 * w[:, 0].mean(axis=1) - 0.5)

    window = np.zeros((3, 20), dtype=np.float32)
    window[0] += 2.0
    window[1] += 9.0  # large, but not read
    r = network_effects(proba, window, np.zeros_like(window), channel_groups(["x", "y", "z"]))
    assert r["effects"]["x"] == pytest.approx(logit(sig(2.5)) - logit(sig(-0.5)))
    assert r["effects"]["y"] == pytest.approx(0.0) and r["effects"]["z"] == pytest.approx(0.0)


# ---------------------------------------------------------------- global importance


def test_permutation_importance_flags_the_used_group_and_not_the_ignored_one():
    rng = np.random.default_rng(0)
    n = 3000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < sig(2 * a - 0.5)).astype(int)
    X = pd.DataFrame({"a": a, "b": b})
    pred = lambda t: sig(2 * t["a"].to_numpy())  # noqa: E731 - a model that reads only a
    out = permutation_importance_table(
        pred, X, y, {"ga": ("A", ["a"]), "gb": ("B", ["b"])}, repeats=4, seed=1
    )
    assert out["groups"]["ga"]["auc_drop"] > 0.15 and abs(out["groups"]["gb"]["auc_drop"]) < 0.01
    sh = mean_abs_shapley_table(
        pred, X, X.median(), {"ga": ("A", ["a"]), "gb": ("B", ["b"])}, n=40, seed=0
    )
    assert sh["ga"]["mean_abs"] > 0.3 and sh["gb"]["mean_abs"] < 1e-9

    Xw = rng.normal(size=(600, 2, 10)).astype(np.float32)
    yw = (rng.random(600) < sig(3 * Xw[:, 0].mean(axis=1))).astype(int)
    w = permutation_importance_windows(
        lambda W: sig(3 * W[:, 0].mean(axis=1)), Xw, yw, channel_groups(["x", "y"]), repeats=3
    )
    assert w["groups"]["x"]["auc_drop"] > 0.1 and abs(w["groups"]["y"]["auc_drop"]) < 0.01


def test_shap_library_importance_when_installed():
    shap = pytest.importorskip("shap")
    assert shap
    from flarenet.explain.importance import shap_feature_importance

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    imp = shap_feature_importance(lambda t: sig(2 * t["a"].to_numpy()), X, X, max_rows=30)
    assert imp["a"] > 10 * max(imp["b"], 1e-9)


# ---------------------------------------------------------------- the block shown on the website


class _Trees:
    cols = ["f_cur", "h_nM_6", "h_nC_6", "h_since_M"]

    def predict_proba(self, tab: pd.DataFrame) -> np.ndarray:
        t = tab.reindex(columns=self.cols)
        return sig(
            1.2 * t["h_nM_6"].to_numpy()
            + 0.8 * (t["f_cur"].to_numpy() + 6)
            - 0.01 * t["h_since_M"].to_numpy()
            - 2.0
        )


def _fake_live():
    ref = pd.Series({"f_cur": -6.5, "h_nM_6": 0.0, "h_nC_6": 1.0, "h_since_M": 80.0})
    return SimpleNamespace(
        gbm=_Trees(), stack=Stack(0.4, 0.6, -0.3), reference=ref, horizon_min=360
    )


def test_explain_latest_adds_up_to_the_probability_on_the_page():
    live = _fake_live()
    channels = ["x", "y"]
    window = np.zeros((2, 10), dtype=np.float32)
    window[0] += 1.0
    ref_w = np.zeros_like(window)
    net = lambda w: sig(1.0 * w[:, 0].mean(axis=1) - 1.0)  # noqa: E731
    row = pd.Series({"f_cur": -5.0, "h_nM_6": 2.0, "h_nC_6": 5.0, "h_since_M": 3.0})
    p_net, p_gbm = (
        float(net(window[None])[0]),
        float(live.gbm.predict_proba(pd.DataFrame([row]))[0]),
    )
    p_final = float(live.stack.predict([p_net], [p_gbm])[0])
    b = explain_latest(net, window, ref_w, channels, "2026-11-10T03:30:00Z", p_final, live, row)
    assert b["consistent"] and b["p"] == pytest.approx(p_final, abs=1e-4)
    assert [m["id"] for m in b["models"]] == ["network", "trees"]
    assert [round(m["weight"], 3) for m in b["models"]] == [0.4, 0.6]
    total = sum(g["effect"] for g in b["groups"])  # every group is listed here, rounded to 3 places
    want = logit(p_final) - logit(b["p_base"])
    assert total == pytest.approx(want, abs=0.02)
    assert [g["effect"] for g in b["groups"]] == sorted(
        (g["effect"] for g in b["groups"]), key=abs, reverse=True
    )
    top = b["groups"][0]
    assert (
        top["model"] == "trees"
        and top["effect"] > 0
        and "note" in b
        and "not a prediction" in b["note"]
    )
    details = {g["id"]: g["detail"] for g in b["groups"] if "detail" in g}
    assert (
        details["level_now"].startswith("M1.0") and "2 M-class" in details["flares_hours"]
    )  # readable values


def test_explain_latest_without_the_tree_model_explains_the_network_alone():
    window = np.zeros((2, 10), dtype=np.float32)
    window[0] += 1.0
    net = lambda w: sig(1.0 * w[:, 0].mean(axis=1) - 1.0)  # noqa: E731
    b = explain_latest(
        net, window, np.zeros_like(window), ["x", "y"], "t", float(net(window[None])[0])
    )
    assert [m["id"] for m in b["models"]] == ["network"] and b["consistent"]
    assert {g["model"] for g in b["groups"]} == {"network"}


def test_a_forecast_that_does_not_match_is_flagged_not_hidden():
    window = np.zeros((1, 5), dtype=np.float32)
    net = lambda w: np.full(len(w), 0.2)  # noqa: E731
    b = explain_latest(net, window, window, ["x"], "t", 0.9)
    assert b["consistent"] is False


def test_describe_gives_plain_words():
    row = pd.Series(
        {"f_cur": -5.0, "h_nC_6": 4.0, "h_nM_6": 1.0, "h_since_M": 3.2, "f_cov_24h": 0.97}
    )
    assert describe("level_now", row).startswith("M1.0") and describe("level_now", row).endswith(
        "now"
    )
    assert "4 C-class or stronger and 1 M-class" in describe("flares_hours", row)
    assert describe("since_flare", row) == "last M-class flare 3 h ago"
    assert (
        describe("since_flare", pd.Series({"h_since_M": 720.0}))
        == "no M-class flare in the history used"
    )
    assert describe("coverage", row) == "97% of the last 24 h present"
    assert (
        describe("trend", row) is None
        and describe("level_now", pd.Series({"f_cur": np.nan})) is None
    )


def test_export_never_breaks_on_a_failed_explanation():
    out = explain_block(SimpleNamespace(config={}), "cpu", None, pd.DataFrame(), None)
    assert out["available"] is False and out["reason"]


# ---------------------------------------------------------------- the real thing, end to end on the synthetic Sun


@pytest.mark.slow
def test_explanation_of_a_real_forecast_matches_the_page(tmp_path):
    cfg = tiny_cfg(tmp_path, epochs=2)
    cfg.data.split = "chrono"
    cfg.data.train_frac, cfg.data.val_frac = 0.6, 0.2
    cfg.training.calibrate = "platt"
    cfg.sources[0].params = {"n_days": 120}
    cfg.validate()
    orch.run_all(cfg)
    ckpt = cfg.run_dir / "checkpoint.pt"
    lm = load_checkpoint(ckpt, "cpu")
    ctx = orch.run_features(cfg)
    bundle = ctx.state["bundles"]["synthetic"]
    tab, rows = set_features(ctx.state["splits"]["train"], ctx.state["bundles"], 360, 60)
    tab, dropped = drop_features(tab, LIVE_COMPATIBLE_DROP)
    gbm = HourlyGBM().fit(tab, np.asarray(ctx.state["splits"]["train"].y_fc)[rows].astype(int))
    X = tab.iloc[:8].reset_index(drop=True)
    pkg = package(gbm, Stack(0.5, 0.7, -0.4), {"stack": 0.4}, {}, 360, file_sha256(ckpt),
                  {"X": X, "p_gbm": gbm.predict_proba(X)}, dropped, reference=tab[gbm.cols].median().to_dict())  # fmt: skip
    path = tmp_path / "stack.pkl"
    path.write_bytes(pickle.dumps(pkg))
    live, why = load_live_stack(path, ckpt)
    assert live is not None, why
    assert live.reference is not None and set(live.reference.index) == set(gbm.cols)

    fc = live.apply(forecast_bundle(lm, bundle, torch.device("cpu")), bundle)
    assert bool(fc["stack_used"].iloc[-1])
    b = explain_for_export(lm, torch.device("cpu"), bundle, fc, live)
    assert b["consistent"], "the explanation must reproduce the forecast shown on the page"
    assert b["p"] == pytest.approx(float(fc["p_mx_6h"].iloc[-1]), abs=1e-4) and b["time"].endswith(
        "Z"
    )
    assert {g["model"] for g in b["groups"]} <= {"network", "trees"} and len(b["groups"]) >= 4
    blk = explain_block(lm, torch.device("cpu"), bundle, fc, live)
    assert blk["available"] is True and "p_base" in blk
    # a stack file without a stored reference (older files) falls back to the self-check rows
    old = dict(pkg, reference=None)
    path.write_bytes(pickle.dumps(old))
    live2, _ = load_live_stack(path, ckpt)
    assert live2.reference is not None


def test_global_report_ranks_the_used_inputs_first_and_writes_files(tmp_path):
    from flarenet.explain.report import build_global_report, plot_global, render_markdown

    rng = np.random.default_rng(0)
    n = 800
    live = _fake_live()
    tab = pd.DataFrame({
        "f_cur": rng.normal(-6, 0.7, n), "h_nM_6": rng.poisson(0.3, n).astype(float),
        "h_nC_6": rng.poisson(1.0, n).astype(float), "h_since_M": rng.uniform(0, 300, n),
    })  # fmt: skip
    X = rng.normal(size=(n, 2, 10)).astype(np.float32)
    y = (rng.random(n) < sig(logit(live.gbm.predict_proba(tab)) + 2 * X[:, 0].mean(axis=1))).astype(
        int
    )
    rep = build_global_report(
        "toy", {"start": "2026-01-01", "end": "2026-02-01"}, y, tab, X,
        lambda w: sig(2 * w[:, 0].mean(axis=1)), live, ["x", "y"], np.zeros((2, 10), dtype=np.float32),
        repeats=3, n_tree=30, n_net=20,
    )  # fmt: skip
    drops = [g["auc_drop"] for g in rep["trees"]["groups"]]
    assert rep["windows"] == n and drops == sorted(drops, reverse=True)  # ranked, largest first
    assert (
        rep["trees"]["groups"][0]["auc_drop"] > 0.05
    )  # the toy model reads every group, so the top one matters
    assert (
        rep["network"]["groups"][0]["id"] == "x" and rep["network"]["groups"][0]["auc_drop"] > 0.02
    )
    assert abs(rep["network"]["groups"][1]["auc_drop"]) < 0.01  # channel y is never read
    md = render_markdown(rep)
    assert "What the models rely on: toy" in md and "not about the Sun" in md
    assert plot_global(rep, tmp_path / "imp.png").stat().st_size > 5000

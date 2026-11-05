"""The live use of the tree model: it must only ever be used with the checkpoint it was fitted with, only where it has
enough history behind it, and must never break the page."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import tiny_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_web import apply_stack, build_payload  # noqa: E402

from flarenet import orchestration as orch  # noqa: E402
from flarenet.core.constants import CLASS_NAMES  # noqa: E402
from flarenet.gbm6h.features import prepare, set_features, window_features  # noqa: E402
from flarenet.gbm6h.live import file_sha256, load_live_stack, package  # noqa: E402
from flarenet.gbm6h.model import HourlyGBM  # noqa: E402
from flarenet.gbm6h.pipeline import LIVE_COMPATIBLE_DROP, drop_features  # noqa: E402
from flarenet.gbm6h.stack import Stack  # noqa: E402

_STACK = Stack(1.0, 1.0, -0.5)


class _Fake:
    """A stand-in tree model that uses a rotation feature (a module-level class so that it can be pickled)."""

    cols = ("f_cur", "rot_nM_slot")

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live")
    cfg = tiny_cfg(tmp, epochs=1)
    cfg.sources[0].params = {"n_days": 90}
    cfg.validate()
    ctx = orch.run_features(cfg)
    bundle = ctx.state["bundles"]["synthetic"]
    tab, rows = set_features(ctx.state["splits"]["train"], ctx.state["bundles"], 360, 60)
    tab, dropped = drop_features(tab, LIVE_COMPATIBLE_DROP)
    y = np.asarray(ctx.state["splits"]["train"].y_fc)[rows].astype(int)
    gbm = HourlyGBM().fit(tab, y)
    return SimpleNamespace(tmp=tmp, bundle=bundle, tab=tab, gbm=gbm, dropped=dropped)


def _write(
    parts,
    name="stack.pkl",
    ckpt_bytes=b"checkpoint-A",
    stack=_STACK,
    gbm=None,
    tamper=False,
):
    ckpt = parts.tmp / "ckpt.pt"
    ckpt.write_bytes(ckpt_bytes)
    g = gbm or parts.gbm
    X = parts.tab.iloc[:8].reset_index(drop=True)
    p = g.predict_proba(X) + (0.2 if tamper else 0.0)
    pkg = package(g, stack, {"cnn": 0.4, "gbm": 0.4, "stack": 0.35}, {"balanced": {"threshold": 0.35}},
                  360, file_sha256(ckpt), {"X": X, "p_gbm": p}, parts.dropped)  # fmt: skip
    path = parts.tmp / name
    path.write_bytes(pickle.dumps(pkg))
    return path, ckpt


def test_loader_accepts_the_matching_checkpoint_and_refuses_everything_else(parts):
    path, ckpt = _write(parts)
    live, why = load_live_stack(path, ckpt)
    assert live is not None and why == "" and live.threshold == 0.35

    other = parts.tmp / "other.pt"
    other.write_bytes(b"a different network")
    assert "different network checkpoint" in load_live_stack(path, other)[1]
    assert "does not exist" in load_live_stack(parts.tmp / "nope.pkl", ckpt)[1]

    bad, ckpt2 = _write(parts, "tampered.pkl", tamper=True)  # a library that gives other numbers
    assert "differ on this machine" in load_live_stack(bad, ckpt2)[1]

    none, ckpt3 = _write(parts, "nostack.pkl", stack=None)
    assert "no positive weight" in load_live_stack(none, ckpt3)[1]

    junk = parts.tmp / "junk.pkl"
    junk.write_bytes(b"not a pickle")
    assert "could not be read" in load_live_stack(junk, ckpt)[1]


def test_loader_refuses_features_the_live_feed_cannot_provide(parts):
    path, ckpt = _write(parts, "rot.pkl", gbm=_Fake())
    assert "7-day live feed" in load_live_stack(path, ckpt)[1]


def test_stack_is_applied_only_where_three_days_of_history_exist(parts):
    path, ckpt = _write(parts)
    live, _ = load_live_stack(path, ckpt)
    b = parts.bundle
    t = np.arange(120, 6 * 1440, 60, dtype=float)
    fc = pd.DataFrame({"time_min": t, "p_mx_6h": np.full(len(t), 0.3), "alert": np.zeros(len(t), dtype=int),
                       "nowcast_class": [CLASS_NAMES[0]] * len(t)})  # fmt: skip
    out = live.apply(fc, b)
    early = t < 72 * 60
    assert (~out["stack_used"][early]).all() and out["stack_used"][~early].all()
    assert np.allclose(out["p_mx_6h"][early], 0.3) and (out["p_network"] == 0.3).all()
    i = int(np.flatnonzero(~early)[5])
    lf, ev = prepare(b)
    row = pd.DataFrame([window_features(int(t[i]), lf, ev, b.t0_unix, 360)])
    want = live.stack.predict([0.3], live.gbm.predict_proba(row))[0]
    assert out["p_mx_6h"].iloc[i] == pytest.approx(want)
    assert (out["alert"][~early] == (out["p_mx_6h"][~early] >= 0.35)).all()
    info = live.info(out)
    assert (
        info["used"]
        and info["network_only_rows"] == int(early.sum())
        and info["alert_levels"] == {"balanced": 0.35}
    )


def test_apply_stack_never_raises_and_says_why(parts):
    fc = pd.DataFrame(
        {"time_min": [5000.0], "p_mx_6h": [0.2], "alert": [0], "nowcast_class": [CLASS_NAMES[0]]}
    )
    out, info = apply_stack(
        fc, parts.bundle, str(parts.tmp / "missing.pkl"), str(parts.tmp / "ckpt.pt")
    )
    assert out is fc and info["used"] is False and "does not exist" in info["reason"]

    path, ckpt = _write(parts)
    broken = parts.bundle  # a bundle whose flux column is gone makes the feature step fail
    bad = SimpleNamespace(df=broken.df.drop(columns=[c for c in ("sxr", "sxr_raw") if c in broken.df]),
                          flare_db=broken.flare_db, t0_unix=broken.t0_unix, kind="x", name="x", meta={})  # fmt: skip
    out, info = apply_stack(fc, bad, str(path), str(ckpt))
    assert out is fc and info["used"] is False and info["reason"]


def test_payload_shows_the_stack_threshold_and_which_hours_used_it(parts):
    b = parts.bundle
    t = np.arange(120, 6 * 1440, 60, dtype=float)
    fc = pd.DataFrame({"time_min": t, "p_mx_6h": np.full(len(t), 0.3), "alert": np.zeros(len(t), dtype=int),
                       "nowcast_class": [CLASS_NAMES[0]] * len(t)})  # fmt: skip
    path, ckpt = _write(parts)
    stacked, info = apply_stack(fc, b, str(path), str(ckpt))
    lm = SimpleNamespace(config={"model": {"name": "flarenet"}, "data": {"window_min": 90, "horizon_min": 360}},
                         threshold=0.19, calibration={"kind": "platt"}, alert_levels={"balanced": {"threshold": 0.19}})  # fmt: skip
    bundle = SimpleNamespace(df=b.df, flare_db=b.flare_db, kind="nasa_goes_swpc", name="swpc_live", meta={},
                             unix_of=lambda m: 1_760_000_000.0 + 60.0 * np.asarray(m, dtype=float))  # fmt: skip
    cases = pd.DataFrame(columns=["peak_utc", "class", "forecast", "lead_min", "max_prob_before"])
    pay = build_payload(bundle, lm, stacked, stacked, 6.0, "M", cases, info)
    assert pay["model"]["threshold"] == 0.35 and pay["model"]["stack"]["used"] is True
    assert pay["model"]["alert_levels"] == {"balanced": 0.35}
    assert len(pay["forecast"]["stack"]) == len(pay["forecast"]["t"]) and set(
        pay["forecast"]["stack"]
    ) <= {0, 1}
    off = build_payload(bundle, lm, fc, fc, 6.0, "M", cases, {"used": False, "reason": "why"})
    assert (
        off["model"]["threshold"] == 0.19
        and off["model"]["stack"]["reason"] == "why"
        and "stack" not in off["forecast"]
    )
    assert "stack" not in build_payload(bundle, lm, fc, fc, 6.0, "M", cases)["model"]

"""Relative (instrument-independent) features and instrument-shift augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from flarenet.core.config import Config
from flarenet.pipelines.features.channels import CHANNEL_NAMES, compute_channels
from flarenet.pipelines.features.windows import build_windows
from flarenet.pipelines.training.dataset import WindowDataset, instrument_augment
from flarenet.sources.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(n_days=6, seed=3)[0]


def _scaled(df, gain):
    out = df.copy()
    out["sxr"] = out["sxr"] * gain
    out["hxr"] = out["hxr"] * gain
    return out


def test_relative_channels_ignore_detector_gain(df):
    a = compute_channels(df, relative=True)
    b = compute_channels(_scaled(df, 7.0), relative=True)
    for name, x, y in zip(CHANNEL_NAMES, a, b, strict=True):
        assert np.allclose(x, y, atol=1e-6), name


def test_absolute_channels_do_depend_on_gain(df):
    a = compute_channels(df)
    b = compute_channels(_scaled(df, 7.0))
    assert not np.allclose(a[0], b[0])  # log_sxr moves by log10(7)


def test_relative_baseline_is_causal(df):
    full = compute_channels(df, relative=True)
    cut = 4000
    part = compute_channels(df.iloc[:cut].reset_index(drop=True), relative=True)
    # values after the first 60 minutes never see the future (leading values are back-filled)
    for x, y in zip(full, part, strict=True):
        assert np.allclose(x[100:cut], y[100:cut], atol=1e-6)


def test_build_windows_relative_flag(df):
    w_abs = build_windows(df, stride_min=120)
    w_rel = build_windows(df, stride_min=120, relative=True)
    assert w_abs.X.shape == w_rel.X.shape
    assert not np.allclose(w_abs.X[:, 0], w_rel.X[:, 0])
    assert np.array_equal(w_abs.y_fc, w_rel.y_fc)


def test_instrument_augment_shifts_levels_and_drops_hxr():
    names = CHANNEL_NAMES
    x = np.random.default_rng(0).normal(size=(len(names), 20)).astype(np.float32)
    rng = np.random.default_rng(1)
    out = instrument_augment(x.copy(), names, rng, level_jitter=0.5)
    assert not np.allclose(out[0], x[0])
    assert np.allclose(np.diff(out[0]), np.diff(x[0]), atol=1e-5)  # constant shift only
    assert np.allclose(out[2], x[2])  # derivatives untouched
    dropped = instrument_augment(x.copy(), names, np.random.default_rng(2), hxr_dropout=1.0)
    assert np.allclose(dropped[1], x[0])
    assert np.allclose(dropped[6], 0.0)


def test_dataset_plain_eval_unchanged(df):
    ws = build_windows(df, stride_min=240)
    ds = WindowDataset(ws.X, ws.y_now, ws.y_fc)
    x, _, _ = ds[0]
    assert x.shape == ws.X[0].shape
    assert np.allclose(x.numpy(), (ws.X[0] - ds.mu[0]) / ds.sg[0], atol=1e-5)


def test_dataset_instrument_mode_runs(df):
    ws = build_windows(df, stride_min=240)
    ds = WindowDataset(ws.X, ws.y_now, ws.y_fc, augment=True, channels=ws.channels,
                       level_jitter=0.4, hxr_dropout=0.5, noise_max=0.2)  # fmt: skip
    assert ds[0][0].shape == ws.X[0].shape
    assert ds.X is None


def test_config_validation():
    cfg = Config()
    cfg.features.relative = True
    cfg.training.aug_level_jitter = 0.3
    with pytest.raises(ValueError, match="aug_level_jitter"):
        cfg.validate()
    cfg2 = Config()
    cfg2.training.aug_hxr_dropout = 1.5
    with pytest.raises(ValueError):
        cfg2.validate()


# ---- simulated instrument source and cross-calibration ------------------------------------
def test_synthetic_instrument_source_and_kind():
    import flarenet.sources  # noqa: F401
    from flarenet.core.config import SourceSpec
    from flarenet.sources.base import make_source

    spec = SourceSpec(
        name="x",
        kind="synthetic_instrument",
        params={"n_days": 3, "gain": 40.0, "noise_frac": 0.1, "hxr_mode": "none"},
    )
    src = make_source(spec, 5)
    assert src.kind == "synthetic_instrument"
    raw = src.fetch()
    assert "hxr" not in raw.channels
    base = make_source(SourceSpec(name="b", kind="synthetic", params={"n_days": 3}), 5).fetch()
    ratio = np.median(raw.channels["sxr"].values / base.channels["sxr"].values)
    assert 30 < ratio < 50
    assert np.array_equal(raw.aligned["flare_class"], base.aligned["flare_class"])  # same truth


def test_goes_calibrated_scalers(tmp_path):
    import pandas as pd

    from flarenet.pipelines.preprocessing.scaling import apply_scaling

    rng = np.random.default_rng(0)
    n = 3000
    true = 10 ** rng.uniform(-8, -5, n)  # GOES-like flux
    raw = (true / 2e-9) ** 1.1 * 100  # instrument: nonlinear counts
    df = pd.DataFrame({"time_min": np.arange(n, dtype=float), "sxr_raw": raw, "hxr_raw": raw})
    t0 = 1_600_000_000.0
    ref = pd.DataFrame({"time": t0 + np.arange(n) * 60.0, "flux": true})
    f = tmp_path / "ref.csv"
    ref.to_csv(f, index=False)
    out = apply_scaling("goes_calibrated", df, {"reference_csv": str(f)}, t0)
    assert out.attrs["calibration"]["r2"] > 0.999
    assert np.allclose(np.log10(out.sxr), np.log10(true), atol=0.01)
    q = apply_scaling("goes_calibrated", df, {"quiet_flux": 2e-9}, t0)
    assert q.attrs["calibration"]["mode"] == "quiet_anchor"
    g = apply_scaling("goes_calibrated", df, {"gain": 2.0, "power": 1.0}, t0)
    assert np.allclose(g.sxr, 2.0 * df.sxr_raw.clip(1e-12))
    with pytest.raises(ValueError, match="200 overlapping"):
        apply_scaling("goes_calibrated", df.iloc[:50], {"reference_csv": str(f)}, t0)

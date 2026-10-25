import numpy as np
import pandas as pd
import pytest
from conftest import make_lightcurve

from flarenet.core.artifacts import DataStore
from flarenet.core.goes import class_type_to_flux, flux_to_class_idx, flux_to_class_type
from flarenet.core.schema import (
    Channel,
    LightcurveBundle,
    RawSeries,
    SchemaError,
    WindowSet,
    validate_lightcurve,
)
from flarenet.core.utils import fingerprint, to_unix_seconds


def test_class_type_to_flux():
    assert class_type_to_flux("M5.2") == pytest.approx(5.2e-5)
    assert class_type_to_flux("X") == pytest.approx(1e-4)
    assert class_type_to_flux("c1.0") == pytest.approx(1e-6)
    assert class_type_to_flux("X28") == pytest.approx(2.8e-3)
    for bad in ("", None, "Z9", "hello", 5):
        assert np.isnan(class_type_to_flux(bad))


def test_flux_to_class_idx_boundaries():
    f = [5e-9, 9.99e-8, 1e-7, 9.9e-7, 1e-6, 1e-5, 9e-5, 1e-4, 2e-3, np.nan]
    assert flux_to_class_idx(f).tolist() == [0, 0, 1, 1, 2, 3, 3, 4, 4, 0]


def test_flux_class_roundtrip():
    for s in ("B3.4", "C1.0", "M5.2", "X1.1"):
        assert flux_to_class_type(class_type_to_flux(s)) == s
    assert flux_to_class_type(3e-8) == "A3.0" and flux_to_class_type(float("nan")) == "?"


def test_to_unix_seconds_formats():
    out = to_unix_seconds(
        ["2024-05-10T06:27Z", None, "2024-05-10T06:27:00Z", "2024-05-10 06:27:30", "junk"]
    )
    assert out[0] == out[2] and out[3] - out[0] == 30 and np.isnan(out[1]) and np.isnan(out[4])


def test_fingerprint_stable_and_sensitive():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_validate_lightcurve_accepts_good_and_rejects_bad():
    validate_lightcurve(make_lightcurve(10))
    with pytest.raises(SchemaError, match="missing columns"):
        validate_lightcurve(make_lightcurve(10).drop(columns=["hxr"]))
    with pytest.raises(SchemaError, match="regular"):
        validate_lightcurve(make_lightcurve(10).assign(time_min=np.arange(10) * 2.0))
    with pytest.raises(SchemaError, match="0..4"):
        validate_lightcurve(make_lightcurve(10).assign(flare_class=9))
    with pytest.raises(SchemaError, match="strictly positive"):
        validate_lightcurve(make_lightcurve(10).assign(sxr=0.0))
    bad = make_lightcurve(10)
    bad.loc[3, "flare_class"] = 2  # class > 0 with flare_id == -1
    with pytest.raises(SchemaError, match="flare_id"):
        validate_lightcurve(bad)


def test_rawseries_requires_sxr_and_aligned_length():
    with pytest.raises(SchemaError):
        RawSeries("x", "k", {"hxr": Channel([0, 1], [1, 2])})
    with pytest.raises(SchemaError):
        RawSeries("x", "k", {"sxr": Channel([0, 1], [1, 2])}, aligned={"a": np.zeros(3)})
    with pytest.raises(SchemaError):
        Channel([0, 1, 2], [1, 2])


def test_datastore_roundtrips_and_fingerprint_freshness(tmp_path):
    store = DataStore(tmp_path)
    t = np.arange(5, dtype=float) * 60
    raw = RawSeries("src", "synthetic", {"sxr": Channel(t, np.arange(5.0)), "hxr": Channel(t, np.ones(5))},
                    {"m": 1}, aligned={"flare_id": -np.ones(5, dtype=int)})  # fmt: skip
    assert not store.is_fresh(store.raw_dir("src"), "fp1")
    store.save_raw(raw, "fp1")
    assert store.is_fresh(store.raw_dir("src"), "fp1") and not store.is_fresh(
        store.raw_dir("src"), "fp2"
    )
    back = store.load_raw("src")
    assert np.array_equal(back.channels["sxr"].values, raw.channels["sxr"].values)
    assert back.aligned["flare_id"].tolist() == [-1] * 5 and back.meta == {"m": 1}

    df = make_lightcurve(20)
    b = LightcurveBundle("src", "synthetic", df, 1000.0, {"k": "v"}, pd.DataFrame())
    store.save_bundle(b, "fpb")
    b2 = store.load_bundle("src")
    pd.testing.assert_frame_equal(b2.df, df)
    assert b2.t0_unix == 1000.0 and len(b2.flare_db) == 0

    ws = WindowSet(np.zeros((3, 2, 8), np.float32), np.zeros(3, np.int64), np.array([0, 1, 1], np.float32),
                   np.arange(3.0), np.zeros(3, np.float32), ["a", "b"])  # fmt: skip
    store.save_windows("src", ws.tagged("src"), "fpw")
    w2 = store.load_windows("src")
    assert len(w2) == 3 and w2.channels == ["a", "b"] and w2.source_names == ["src"]


def test_corrupt_meta_counts_as_stale(tmp_path):
    d = tmp_path / "raw" / "x"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{not json")
    assert not DataStore.is_fresh(d, "anything")


def test_windowset_concat_remaps_sources_and_rejects_mismatched_channels():
    def mk(ch):
        return WindowSet(np.zeros((2, len(ch), 8), np.float32), np.zeros(2, np.int64), np.zeros(2, np.float32),
                         np.arange(2.0), np.zeros(2, np.float32), ch)  # fmt: skip

    both = WindowSet.concat(
        [mk(["a"]).tagged("s1"), mk(["a"]).tagged("s2"), mk(["a"]).tagged("s1")]
    )
    assert both.source_names == ["s1", "s2"] and both.source_idx.tolist() == [0, 0, 1, 1, 0, 0]
    with pytest.raises(SchemaError):
        WindowSet.concat([mk(["a"]), mk(["b"])])
    with pytest.raises(SchemaError):
        WindowSet.concat([])

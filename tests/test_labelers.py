import numpy as np
import pandas as pd
import pytest
from conftest import json_dump

from flarenet.core.constants import CLASS_TO_IDX
from flarenet.core.registry import LABELERS
from flarenet.core.schema import SchemaError
from flarenet.pipelines.preprocessing import labelers  # noqa: F401
from flarenet.pipelines.preprocessing.crosscheck import compare_event_tables
from flarenet.pipelines.preprocessing.labelers.base import LabelInput, events_to_arrays
from flarenet.pipelines.preprocessing.labelers.goes_events import detect_goes_events


def flare(n, centre, peak, rise=4.0, decay=12.0, base=1e-8):
    t = np.arange(n, dtype=float)
    d = t - centre
    return base + peak * np.where(d < 0, np.exp(-(d**2) / (2 * rise**2)), np.exp(-d / decay))


def lightcurve(flux):
    n = len(flux)
    return pd.DataFrame({"time_min": np.arange(n, dtype=float), "sxr": flux, "hxr": flux,
                         "sxr_raw": flux, "hxr_raw": flux})  # fmt: skip


def test_goes_detector_finds_known_flares_and_classes():
    n = 2400
    flux = flare(n, 500, 3e-5) + flare(n, 1500, 4e-6) - 1e-8
    flux = flux + flare(n, 1000, 3e-8) - 1e-8  # A-class bump must be ignored
    events = detect_goes_events(flux)
    assert len(events) == 2
    m, c = events
    assert 475 <= m["start"] <= 500 and m["peak"] == 500 and 505 < m["end"] < 700
    assert m["peak_flux"] == pytest.approx(3e-5, rel=0.05)
    assert 1475 <= c["start"] <= 1500 and c["peak"] == 1500


def test_goes_detector_ends_event_at_half_decay_and_handles_gaps():
    flux = flare(800, 300, 1e-5)
    flux[330:345] = np.nan  # NaN gap inside the decay must not crash or split the event
    ev = detect_goes_events(flux)
    assert len(ev) == 1
    half = 1e-8 + 0.5 * (ev[0]["peak_flux"] - 1e-8)
    assert flux[ev[0]["end"]] <= half or np.isnan(flux[ev[0]["end"]])


def test_goes_detector_noise_only_gives_no_events():
    rng = np.random.default_rng(0)
    flux = 3e-8 * rng.lognormal(0, 0.15, 5000)
    assert detect_goes_events(flux) == []
    assert detect_goes_events(np.array([1.0, 2.0])) == []


def test_goes_event_labeler_assigns_class_over_whole_event():
    flux = flare(2400, 500, 3e-5)
    out = LABELERS.get("goes_events")({}).label(LabelInput(lightcurve(flux), 0.0))
    cls, fid = out["flare_class"], out["flare_id"]
    assert cls.max() == CLASS_TO_IDX["M"] and set(np.unique(fid)) == {-1, 0}
    assert (cls[fid == 0] == 3).all() and (cls[fid == -1] == 0).all()
    assert (out["active_regime"] == (cls >= 2)).all()


def test_relative_percentile_matches_the_v5_1_loop():
    rng = np.random.default_rng(1)
    s = rng.poisson(100, 900).astype(float)
    s[200:230], s[500:520], s[700:705] = 400, 2500, 160
    s[10] = np.nan
    out = LABELERS.get("relative_percentile")({}).label(LabelInput(lightcurve(s), 0.0))

    p10 = np.nanpercentile(s, 10)  # the original implementation, verbatim
    cls = np.zeros(len(s), dtype=int)
    for i, m in enumerate((1.5, 3.0, 8.0, 20.0), start=1):
        cls[s > p10 * m] = i
    fid = -np.ones(len(s), dtype=int)
    eid, inside = 0, False
    for i in range(len(s)):
        if cls[i] > 0 and not inside:
            inside, eid = True, eid + 1
        if cls[i] > 0:
            fid[i] = eid
        if cls[i] == 0:
            inside = False
    assert np.array_equal(out["flare_class"], cls) and np.array_equal(out["flare_id"], fid)


def test_precomputed_requires_labels():
    df = lightcurve(np.full(10, 1e-8))
    with pytest.raises(SchemaError, match="precomputed"):
        LABELERS.get("precomputed")({}).label(LabelInput(df, 0.0))
    al = {"flare_class": np.zeros(10, int), "flare_id": -np.ones(10, int)}
    assert (
        LABELERS.get("precomputed")({}).label(LabelInput(df, 0.0, al))["active_regime"].sum() == 0
    )


def test_unknown_labeler_param_and_name():
    with pytest.raises(KeyError):
        LABELERS.get("goes_events")({"bogus": 1})
    with pytest.raises(KeyError, match="Available"):
        LABELERS.get("nope")


def test_events_to_arrays_overlap_keeps_stronger_and_renumbers():
    out = events_to_arrays(20, [(2, 8, 2), (6, 12, 3), (15, 17, 0), (18, 25, 1)])
    cls, fid = out["flare_class"], out["flare_id"]
    assert cls[2] == 2 and cls[7] == 3 and cls[12] == 3 and cls[15] == 0 and cls[19] == 1
    assert sorted(set(fid[fid >= 0].tolist())) == [0, 1, 2]  # dense ids, class-0 event dropped


def test_catalog_labeler_marks_catalog_spans(tmp_path):
    cat = tmp_path / "cat.csv"
    t0 = 1_714_521_600.0
    iso = lambda m: pd.Timestamp(t0 + m * 60, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    pd.DataFrame({"begin_time": [iso(100), iso(400)], "peak_time": [iso(110), iso(405)],
                  "end_time": [iso(130), None], "class_type": ["M2.0", "C5.0"]}).to_csv(cat, index=False)  # fmt: skip
    lab = LABELERS.get("catalog")({"catalog": "csv", "catalog_params": {"path": str(cat)}})
    out = lab.label(LabelInput(lightcurve(np.full(600, 1e-8)), t0))
    cls = out["flare_class"]
    assert (cls[100:131] == 3).all() and cls[99] == 0 and cls[131] == 0
    assert cls[400] == 2 and cls[410] == 2  # missing end -> peak + (peak - begin)
    assert cls[412] == 0


def test_donki_catalog_labeler_from_saved_json(tmp_path):
    t0 = 1_714_521_600.0
    recs = [{"flrID": "a", "beginTime": "2024-05-01T02:00Z", "peakTime": "2024-05-01T02:10Z",
             "endTime": "2024-05-01T02:30Z", "classType": "X1.1"}]  # fmt: skip
    p = json_dump(recs, tmp_path / "d.json")
    lab = LABELERS.get("catalog")({"catalog": "donki", "catalog_params": {"json_path": str(p)}})
    out = lab.label(LabelInput(lightcurve(np.full(400, 1e-8)), t0))
    assert out["flare_class"].max() == 4 and (out["flare_class"][120:151] == 4).all()


def test_compare_event_tables_matches_by_peak_time():
    t0 = 1_000_000.0
    det = pd.DataFrame({"peak_min": [100.0, 300.0, 500.0], "class_idx": [3, 2, 2]})
    cat = pd.DataFrame({"peak_unix": [t0 + 105 * 60, t0 + 302 * 60, t0 + 900 * 60],
                        "class_type": ["M1.0", "M1.0", "C1.0"]})  # fmt: skip
    r = compare_event_tables(det, cat, t0, tol_min=15)
    assert r["matched"] == 2 and r["recall_vs_catalog"] == pytest.approx(2 / 3)
    assert r["precision_C_plus"] == pytest.approx(2 / 3) and r["class_agreement"] == pytest.approx(
        0.5
    )
    empty = compare_event_tables(det.iloc[:0], cat.iloc[:0], t0)
    assert empty["recall_vs_catalog"] is None and empty["matched"] == 0

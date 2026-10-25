import numpy as np
import pandas as pd

from flarenet.pipelines.preprocessing.flare_db import build_flare_db, format_flare_db


def _df():
    n = 100
    sxr = np.full(n, 1.0)
    sxr[20:30] = [2, 3, 5, 9, 7, 5, 4, 3, 2, 2]
    fc = np.zeros(n, dtype=int)
    fc[20:30] = [1, 2, 2, 3, 3, 2, 2, 1, 1, 1]
    fid = -np.ones(n, dtype=int)
    fid[20:30] = 0
    return pd.DataFrame(
        {"time_min": np.arange(n, dtype=float), "sxr_raw": sxr, "flare_class": fc, "flare_id": fid}
    )


def test_event_summary_values():
    db = build_flare_db(_df(), label="t")
    assert len(db) == 1
    ev = db.iloc[0]
    assert ev["class"] == "M" and ev["class_idx"] == 3
    assert ev["onset_min"] == 20 and ev["peak_min"] == 23 and ev["end_min"] == 29
    assert ev["rise_min"] == 3 and ev["decay_min"] == 6 and ev["duration_min"] == 9
    assert ev["peak_sxr"] == 9.0 and ev["dataset"] == "t"


def test_utc_columns_when_t0_given():
    db = build_flare_db(_df(), t0_unix=1_750_000_000.0)
    assert {"onset_utc", "peak_utc"} <= set(db.columns)


def test_empty_when_no_events():
    df = _df()
    df["flare_id"] = -1
    db = build_flare_db(df)
    assert len(db) == 0
    assert "No flare events" in format_flare_db(db, "x")


def test_synthetic_db_is_consistent(synthetic_small):
    _, _, db = synthetic_small
    assert len(db) > 0
    assert (db.end_min >= db.peak_min).all() and (db.peak_min >= db.onset_min).all()

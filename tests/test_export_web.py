"""scripts/export_web.py on a quiet week: the live refresh crashed when the 7-day window held no M-class
flare (KeyError: 'peak_utc'), which is exactly what happens a week after the last M flare."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from case_study import CASE_COLUMNS, case_table  # noqa: E402
from conftest import make_lightcurve  # noqa: E402
from export_web import build_payload  # noqa: E402

from flarenet.core.constants import CLASS_NAMES  # noqa: E402
from flarenet.pipelines.preprocessing.flare_db import build_flare_db  # noqa: E402

T0 = 1_760_000_000.0


def _bundle(flare_db: pd.DataFrame, n: int = 3000):
    df = make_lightcurve(n)
    return SimpleNamespace(
        df=df, flare_db=flare_db, kind="nasa_goes_swpc", name="swpc_live", meta={},
        unix_of=lambda m: T0 + 60.0 * np.asarray(m, dtype=float),
    )  # fmt: skip


def _forecasts(n: int = 3000, stride: int = 60) -> pd.DataFrame:
    t = np.arange(120, n, stride, dtype=float)
    return pd.DataFrame(
        {
            "time_min": t,
            "p_mx_6h": np.full(len(t), 0.1),
            "alert": np.zeros(len(t), dtype=int),
            "nowcast_class": [CLASS_NAMES[0]] * len(t),
        }
    )


def test_case_table_without_flares_is_empty_but_keeps_its_columns():
    b = _bundle(build_flare_db(make_lightcurve(3000), label="quiet"))
    tab = case_table(b, _forecasts(), _forecasts(stride=5), 0.5, 360, "M")
    assert tab.empty and list(tab.columns) == CASE_COLUMNS


def test_payload_for_a_week_with_no_flare_has_an_empty_flare_list():
    b = _bundle(build_flare_db(make_lightcurve(3000), label="quiet"))
    fc = _forecasts()
    cases = case_table(b, fc, _forecasts(stride=5), 0.5, 360, "M")
    lm = SimpleNamespace(
        config={"model": {"name": "flarenet"}, "data": {"window_min": 90, "horizon_min": 360}},
        threshold=0.48,
    )
    payload = build_payload(b, lm, fc, fc, 1.0, "M", cases)
    assert payload["flares"] == []
    assert len(payload["forecast"]["t"]) > 0 and len(payload["series"]["t"]) > 0


def test_payload_lists_a_flare_inside_the_window():
    b = _bundle(build_flare_db(make_lightcurve(3000), label="quiet"))
    fc = _forecasts()
    peak = pd.to_datetime(T0 + 60.0 * 2000, unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    cases = pd.DataFrame(
        [[peak, "M1.7", "MISSED", np.nan, 0.27, False, "M", True]], columns=CASE_COLUMNS
    )
    lm = SimpleNamespace(
        config={"model": {"name": "flarenet"}, "data": {"window_min": 90, "horizon_min": 360}},
        threshold=0.48,
    )
    payload = build_payload(b, lm, fc, fc, 1.0, "M", cases)
    assert [f["class"] for f in payload["flares"]] == ["M1.7"]
    assert payload["flares"][0]["forecast"] == "MISSED" and payload["flares"][0]["lead_min"] is None

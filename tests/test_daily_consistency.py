"""The consistency check must tell a uniform win from a win carried by one period, and a tie from both."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flarenet.daily.consistency import consistency, format_consistency, rolling_recalibration


def _frame(kind: str, n: int = 2400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.9 * z[i - 1] + rng.normal(0, np.sqrt(1 - 0.81))
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    y = (rng.random(n) < sig(-0.6 + 2.5 * z)).astype(int)
    noaa_noise = rng.normal(0, 1.0, n)
    own = rng.normal(0, 1.0, n)
    late = np.arange(n) >= int(0.6 * n)
    sd = {"uniform": np.full(n, 0.2), "late": np.where(late, 0.05, 1.2), "equal": np.full(n, 1.2)}[
        kind
    ]
    noaa = sig(-0.6 + 2.5 * (z + 1.2 * noaa_noise))
    model = sig(-0.6 + 2.5 * (z + sd * own))
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    d = pd.DataFrame(
        {"y": y, "noaa": np.clip(noaa, 0.01, 0.99), "p_model": np.clip(model, 0.01, 0.99)},
        index=idx,
    )
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    cal = LogisticRegression(C=100).fit(lg(d["noaa"].to_numpy())[:, None], y)
    d["p_noaa_cal"] = cal.predict_proba(lg(d["noaa"].to_numpy())[:, None])[:, 1]
    # FlareNet's probabilities are calibrated too (in the real pipeline they are, on earlier days)
    cm = LogisticRegression(C=100).fit(lg(d["p_model"].to_numpy())[:, None], y)
    d["p_model"] = cm.predict_proba(lg(d["p_model"].to_numpy())[:, None])[:, 1]
    return d


def test_uniform_better_model_is_a_consistent_win():
    o = consistency(_frame("uniform"), n=200)
    assert o["available"] and o["verdict"]["code"] == "consistent_win", o["verdict"]
    assert all(h["AUC"]["diff"] > 0 for h in o["halves"])


def test_win_in_late_period_only_is_called_partial():
    o = consistency(_frame("late"), n=200)
    assert o["verdict"]["code"] == "partial_win", o["verdict"]
    assert "first half" in o["verdict"]["text"]


def test_equal_forecasters_are_a_tie():
    o = consistency(_frame("equal", seed=3), n=200)
    assert (
        o["verdict"]["code"] in ("tie", "partial_win") and o["verdict"]["code"] != "consistent_win"
    )
    assert o["pooled"]["AUC"]["ci95"][0] < 0.02  # no real advantage to find


def test_report_lines_and_small_input():
    o = consistency(_frame("late"), n=100)
    text = "\n".join(format_consistency(o))
    assert (
        "Is the edge consistent?" in text and "second half" in text and "inside each year" in text
    )
    tiny = consistency(_frame("equal").iloc[:30], n=50)
    assert tiny["available"] is False and "Not available" in "\n".join(format_consistency(tiny))


def test_rolling_recalibration_uses_only_the_past():
    d = _frame("equal")
    r = rolling_recalibration(d)
    assert r.iloc[:30].isna().all() and r.notna().sum() > 800
    d2 = d.copy()
    d2.loc[d2.index[900:], "y"] = 1 - d2.loc[d2.index[900:], "y"]  # flip the future labels
    r2 = rolling_recalibration(d2)
    first = r.index[899]
    pd.testing.assert_series_equal(
        r.loc[:first].dropna(), r2.loc[:first].dropna()
    )  # earlier days unchanged

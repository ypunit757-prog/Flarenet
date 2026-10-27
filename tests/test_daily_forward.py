"""Forward test: an append-only log, outcomes filled from the X-ray feed, and a scorer that refuses to over-read."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flarenet.core.config import Config, SourceSpec
from flarenet.daily import forward as fw
from flarenet.daily import live
from flarenet.daily import noaa_text as nt
from flarenet.daily.features import build_day_table


def _row(day: str, p=0.4, noaa=0.3, model="abc", **kw) -> dict:
    return {"for_day": day, "issued_utc": f"{day}T22:00:00Z", "model_id": model, "trained_until": "2026-09-30",
            "target": "M", "p_flarenet": p, "threshold": 0.45, "alert": int(p >= 0.45), "p_noaa": noaa,
            "p_noaa_recal": noaa, "threshold_noaa_recal": 0.4, "noaa_source": "live", **kw}  # fmt: skip


def test_log_is_append_only(tmp_path):
    p = tmp_path / "log.csv"
    assert fw.append_forecast(p, _row("2026-10-25", p=0.31))
    assert fw.append_forecast(p, _row("2026-10-26"))
    assert not fw.append_forecast(
        p, _row("2026-10-25", p=0.99)
    )  # second forecast for a day is refused
    df = fw.read_log(p)
    assert list(df["for_day"]) == ["2026-10-25", "2026-10-26"]
    assert df.loc[0, "p_flarenet"] == pytest.approx(0.31)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from flarenet.daily.pipeline import _bundles

    cfg = Config()
    d = tmp_path_factory.mktemp("fw")
    cfg.paths.data_dir, cfg.paths.runs_dir = str(d / "data"), str(d / "runs")
    cfg.sources = [
        SourceSpec(name="syn", kind="synthetic", role="train", params={"n_days": 40, "seed": 5})
    ]
    _, b = _bundles(cfg, False)
    return b["syn"]


def test_resolve_matches_the_training_labels(bundle, tmp_path):
    tab = build_day_table(bundle, None, 22)
    days = [str(d.date()) for d in tab.index[5:15]]
    p = tmp_path / "log.csv"
    for d in days:
        fw.append_forecast(p, _row(d))
    n = fw.resolve_outcomes(p, bundle, "M")
    df = fw.read_log(p)
    assert n == len(days) and df["y"].notna().all()
    expect = tab.loc[[pd.Timestamp(d, tz="UTC") for d in days], "y_m"].to_numpy()
    assert np.array_equal(df["y"].to_numpy(), expect)  # same definition as training
    assert fw.resolve_outcomes(p, bundle, "M") == 0  # already resolved: untouched


def test_unfinished_and_uncovered_days_are_not_resolved(bundle, tmp_path):
    t_end = bundle.t0_unix + len(bundle.df) * 60.0
    last = pd.Timestamp(t_end, unit="s", tz="UTC").normalize()
    p = tmp_path / "log.csv"
    fw.append_forecast(p, _row(str((last + pd.Timedelta(days=3)).date())))  # in the future
    fw.append_forecast(p, _row(str((last - pd.Timedelta(days=3)).date())))
    gap = bundle.df.copy()
    j0 = int(((last - pd.Timedelta(days=3)).timestamp() - bundle.t0_unix) / 60)
    col = "sxr_raw" if "sxr_raw" in gap.columns else "sxr"
    gap.loc[gap.index[j0 : j0 + 1440], col] = np.nan  # that whole day has no data
    b2 = type(bundle)(
        bundle.name, bundle.kind, gap, bundle.t0_unix, dict(bundle.meta), bundle.flare_db.copy()
    )
    assert fw.resolve_outcomes(p, b2, "M") == 0
    df = fw.read_log(p)
    assert (
        df["y"].isna().all() and df["y_cov"].notna().sum() == 1
    )  # coverage recorded, no outcome invented


def test_fill_noaa_from_archive_and_recalibration(tmp_path, monkeypatch):
    from flarenet.sources.base import SourceUnavailable

    monkeypatch.setattr(
        nt, "http_get", lambda url, **kw: (_ for _ in ()).throw(SourceUnavailable("HTTP 404"))
    )
    q = tmp_path / "rsga" / "2026"
    q.mkdir(parents=True)
    (q / "20261024.txt").write_text(
        ":Issued: 2026 Oct 24 2200 UTC\nIII.  Event probabilities 25 Oct-27 Oct\nClass M    40/40/40\nClass X    05/05/05\n"
    )
    p = tmp_path / "log.csv"
    fw.append_forecast(
        p,
        _row(
            "2026-10-25", noaa=None, p_noaa_recal=None, threshold_noaa_recal=None, noaa_source=None
        ),
    )
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.uniform(0.02, 0.9, 400)
    y = (rng.random(400) < x * 0.5).astype(int)
    recal = (LogisticRegression(C=100).fit(np.log(x / (1 - x))[:, None], y), 0.2)
    assert fw.fill_noaa_from_archive(p, tmp_path, recal=recal) == 1
    r = fw.read_log(p).iloc[0]
    assert r["p_noaa"] == pytest.approx(0.40) and r["noaa_source"] == "archive"
    assert 0 < r["p_noaa_recal"] < 0.4 and r["threshold_noaa_recal"] == pytest.approx(
        0.2
    )  # recalibrated down


def test_live_noaa_parse_and_day_alignment(monkeypatch):
    from test_daily import DAYPRE_2026_10_24

    # the real live file, with the three days' numbers changed so a wrong day would show
    text = DAYPRE_2026_10_24.replace(
        "10            10            10", "40            45            50"
    ).replace("1             1             1\nProton", "5             6             7\nProton")

    class R:
        pass

    def fake(url, **kw):
        r = R()
        r.text = text if "predictions" in url else "<html>nope</html>"
        return r

    import flarenet.sources.nasa.http as http

    monkeypatch.setattr(http, "http_get", fake)
    noaa = live.fetch_noaa_live()
    assert noaa["first_day"] == date(2026, 10, 25) and noaa["source"].endswith("predictions.txt")
    assert live.noaa_prob_for_day(noaa, date(2026, 10, 25), "M") == pytest.approx(
        0.40
    )  # its day-ahead number
    assert live.noaa_prob_for_day(noaa, date(2026, 10, 25), "X") == pytest.approx(0.05)
    # a file issued the day before still covers the 26th, but only as its second day: never paired
    assert np.isnan(live.noaa_prob_for_day(noaa, date(2026, 10, 26), "M"))
    assert np.isnan(live.noaa_prob_for_day(noaa, date(2026, 10, 24), "M"))
    assert np.isnan(live.noaa_prob_for_day(None, date(2026, 10, 26), "M"))
    assert any("OK" in line for line in live.check_noaa_live())


def _log(n: int, edge: float, seed: int = 0, model="m1") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.85 * z[i - 1] + rng.normal(0, 0.5)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    y = (rng.random(n) < sig(-0.5 + 2.0 * z)).astype(int)
    p_noaa = sig(-0.5 + 2.0 * (z + rng.normal(0, 1.0, n)))
    p_fn = sig(-0.5 + 2.0 * (z + rng.normal(0, 1.0 - edge, n)))
    days = pd.date_range("2026-11-01", periods=n, freq="D")
    rows = [_row(str(d.date()), p=float(a), noaa=float(b), model=model, y=int(c), y_cov=1.0)
            for d, a, b, c in zip(days, p_fn, p_noaa, y, strict=True)]  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=fw.LOG_COLUMNS)


def test_scorer_refuses_to_read_a_short_log():
    res = fw.score_log(_log(40, 0.9), n=100)
    assert res["verdict"]["code"] == "too_early"
    assert "Too early" in fw.format_forward(res)


def test_scorer_finds_a_real_edge_and_a_non_edge():
    ahead = fw.score_log(_log(700, 0.95), n=200)
    assert ahead["verdict"]["code"] == "flarenet_ahead", ahead["verdict"]
    level = fw.score_log(_log(700, 0.0, seed=4), n=200)
    assert level["verdict"]["code"] in (
        "no_difference",
        "noaa_ahead",
        "flarenet_ahead",
    )  # a draw can go either way
    assert level["paired_differences"]["AUC"]["diff"] < ahead["paired_differences"]["AUC"]["diff"]


def test_scorer_flags_a_changed_model_and_gaps():
    df = pd.concat(
        [
            _log(100, 0.5, model="m1"),
            _log(100, 0.5, seed=1, model="m2").assign(for_day=lambda d: d["for_day"].shift(-3)),
        ]
    )
    df = df.dropna(subset=["for_day"]).reset_index(drop=True)
    res = fw.score_log(df, n=50)
    assert res["verdict"]["code"] == "model_changed" and res["models"] == ["m1", "m2"]
    gappy = _log(100, 0.5).drop(index=range(10, 20)).reset_index(drop=True)
    assert fw.score_log(gappy, n=50)["calendar_gaps"] == 10


def test_empty_log_scores_without_error(tmp_path):
    res = fw.score_log(fw.read_log(tmp_path / "none.csv"))
    assert res["verdict"]["code"] == "no_data" and "No scored days" in fw.format_forward(res)


def test_freeze_then_predict_and_log_end_to_end(tmp_path, monkeypatch, capsys):
    """scripts/daily_freeze.py and scripts/daily_predict.py, wired together, with no network."""
    import sys

    from flarenet.daily.pipeline import _bundles

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import daily_freeze
    import daily_predict

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"run_name: live\npaths: {{data_dir: {tmp_path / 'data'}, runs_dir: {tmp_path / 'runs'}}}\n"
        "sources:\n  - {name: syn, kind: synthetic, role: eval, params: {n_days: 300, seed: 5}}\n"
    )
    from flarenet.core.config import load_config

    cfg = load_config(str(cfg_path), [])
    _, b = _bundles(cfg, False)
    tab = build_day_table(b["syn"], None, 22)
    rng = np.random.default_rng(0)
    tab["noaa"] = np.clip(0.15 + 0.5 * tab["y_m"] + rng.normal(0, 0.15, len(tab)), 0.01, 0.99)
    table_csv = tmp_path / "day_table.csv"
    tab.to_csv(table_csv)

    assert (
        daily_freeze.main(
            ["-c", str(cfg_path), "--table", str(table_csv), "--out", str(tmp_path / "models")]
        )
        == 0
    )
    pkl = tmp_path / "models" / "daily_model_live.pkl"
    pkg = live.load_daily_model(pkl)
    assert pkg["noaa_recal"] is not None and not any(c.startswith("rot_") for c in pkg["columns"])
    assert pkg["trained_until"] == str(tab.index[-1].date())

    monkeypatch.setattr(daily_predict, "fetch_live_srs", lambda: None)
    monkeypatch.setattr(
        daily_predict, "fetch_noaa_live", lambda: {"issued": "2026-10-24", "source": "test"}
    )
    monkeypatch.setattr(daily_predict, "noaa_prob_for_day", lambda noaa, d, t: 0.37)
    log = tmp_path / "forward_log.csv"
    args = [
        "--model",
        str(pkl),
        "-c",
        str(cfg_path),
        "--log",
        str(log),
        "--out",
        str(tmp_path / "o.json"),
    ]
    assert daily_predict.main(args) == 0
    assert daily_predict.main(args) == 0  # a second run on the same day changes nothing
    out = capsys.readouterr().out
    assert "added" in out and "already had" in out
    df = fw.read_log(log)
    assert len(df) == 1 and df.loc[0, "model_id"] == live.model_id(pkl)
    assert df.loc[0, "p_noaa"] == pytest.approx(0.37) and 0 < df.loc[0, "p_noaa_recal"] < 1
    assert df.loc[0, "noaa_source"] == "test"


def test_forecast_logged_after_its_day_began_is_not_scored():
    df = _log(200, 0.5)
    df["logged_utc"] = [
        f"{d}T22:30:00Z" for d in pd.to_datetime(df["for_day"]).dt.date - pd.Timedelta(days=1)
    ]
    ok = fw.score_log(df, n=50)
    assert ok["late_logged_days_excluded"] == 0 and ok["resolved_days"] == 200
    df.loc[:49, "logged_utc"] = [
        f"{d}T00:10:00Z" for d in df.loc[:49, "for_day"]
    ]  # written 10 minutes into the day
    bad = fw.score_log(df, n=50)
    assert bad["late_logged_days_excluded"] == 50 and bad["resolved_days"] == 150


def test_live_vs_archive_check_counts_disagreements(tmp_path, monkeypatch):
    from flarenet.sources.base import SourceUnavailable

    monkeypatch.setattr(
        nt, "http_get", lambda url, **kw: (_ for _ in ()).throw(SourceUnavailable("HTTP 404"))
    )
    for day, m in (("20261024", "40"), ("20261025", "30")):
        q = tmp_path / "rsga" / "2026"
        q.mkdir(parents=True, exist_ok=True)
        (q / f"{day}.txt").write_text(
            f":Issued: 2026 Oct {day[-2:]} 2200 UTC\nIII.  Event probabilities 25 Oct-27 Oct\nClass M    {m}/{m}/{m}\nClass X    05/05/05\n"
        )
    p = tmp_path / "log.csv"
    fw.append_forecast(p, _row("2026-10-25", noaa=0.40))  # live number equals the archive's
    q2 = tmp_path / "rsga" / "2026"
    (q2 / "20261025.txt").write_text(
        ":Issued: 2026 Oct 25 2200 UTC\nIII.  Event probabilities 26 Oct-28 Oct\nClass M    30/30/30\nClass X    05/05/05\n"
    )
    fw.append_forecast(p, _row("2026-10-26", noaa=0.50))  # archive says 0.30: a disagreement
    c = fw.compare_live_vs_archive(p, tmp_path)
    assert c["compared"] == 2 and c["mismatched"] == 1 and c["max_abs_diff"] == pytest.approx(0.20)


def test_late_row_is_counted_even_before_its_outcome_is_known():
    df = pd.DataFrame(
        [
            _row("2026-10-25", logged_utc="2026-10-25T00:34:36Z"),
            _row("2026-10-26", logged_utc="2026-10-25T22:21:00Z"),
        ]
    )
    res = fw.score_log(df.reindex(columns=fw.LOG_COLUMNS))
    assert (
        res["late_logged_days_excluded"] == 1
        and res["unresolved_days"] == 1
        and res["resolved_days"] == 0
    )

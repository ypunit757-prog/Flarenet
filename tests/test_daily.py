"""Day-level forecaster: parsers (on real NOAA file text), features without look-ahead, and an end-to-end run.

The end-to-end test uses synthetic flux and *planted* sunspot/forecast files, so it proves the plumbing and
the absence of leakage, not skill on the real Sun."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flarenet.core.config import Config, SourceSpec
from flarenet.daily import noaa_text as nt
from flarenet.daily.features import build_day_table, feature_columns, region_features
from flarenet.daily.model import DailyModel, DailyParams

SRS_2024_05_11 = """:Product: 0511SRS.txt
:Issued: 2024 May 11 0030 UTC
# Prepared jointly by the U.S. Dept. of Commerce, NOAA,
# Space Weather Prediction Center and the U.S. Air Force.
#
Joint USAF/NOAA Solar Region Summary
SRS Number 132 Issued at 0030Z on 11 May 2024
Report compiled from data received at SWO on 10 May
I.  Regions with Sunspots.  Locations Valid at 10/2400Z 
Nmbr Location  Lo  Area  Z   LL   NN Mag Type
3663 N27W92   037  0300 Fkc  17   10 Beta-Gamma
3664 S17W48   347  2400 Fkc  20   58 Beta-Gamma-Delta
3666 N08W80   019  0050 Hsx  02   01 Alpha
3667 N28W05   304  0140 Hsx  03   01 Alpha
3670 N17E14   285  0050 Hax  03   06 Alpha
3671 N19E17   281  0030 Hsx  40   01 Alpha
3672 N18E45   254  0140 Cai  08   09 Beta
IA. H-alpha Plages without Spots.  Locations Valid at 10/2400Z May
Nmbr  Location  Lo
3661  N22W84   024
II. Regions Due to Return 11 May to 13 May
Nmbr Lat    Lo
3644 N13    200
"""
SRS_EMPTY = """:Product: 1116SRS.txt
:Issued: 2020 Nov 16 0030 UTC
I.  Regions with Sunspots.  Locations Valid at 15/2400Z 
Nmbr Location  Lo  Area  Z   LL   NN Mag Type
None
IA. H-alpha Plages without Spots.  Locations Valid at 15/2400Z Nov
Nmbr  Location  Lo
2782  S31E11   155
"""
RSGA_2024_05_11 = """:Product: 0511RSGA.txt
:Issued: 2024 May 11 2200 UTC
SDF Number 132 Issued at 2200Z on 11 May 2024
III.  Event probabilities 12 May-14 May
Class M    95/95/95
Class X    75/75/75
"""


def test_parse_srs_real_file():
    issued, reg, ok = nt.parse_srs(SRS_2024_05_11)
    assert ok and issued == pd.Timestamp("2024-05-11 00:30", tz="UTC")
    assert len(reg) == 7 and reg["area"].sum() == 300 + 2400 + 50 + 140 + 50 + 30 + 140
    big = reg[reg["number"] == 3664].iloc[0]
    assert (big.lat, big.lon, big.z, big.p, big.c, big.nn, big.mag) == (
        -17,
        48,
        "F",
        "k",
        "c",
        58,
        "Beta-Gamma-Delta",
    )
    assert reg[reg["number"] == 3670].iloc[0].lon == -14  # east of the central meridian is negative


def test_parse_srs_empty_and_garbage():
    _, reg, ok = nt.parse_srs(SRS_EMPTY)
    assert ok and len(reg) == 0
    assert nt.parse_srs("<html>not found</html>")[2] is False


def _write(text: str) -> Path:
    import tempfile

    p = Path(tempfile.mkdtemp()) / "x.txt"
    p.write_text(text)
    return p


def test_parse_rsga_real_text_and_edge_cases():
    r = nt.parse_rsga(RSGA_2024_05_11)
    assert r["first_day"] == date(2024, 5, 12) and r["m"] == [0.95] * 3 and r["x"] == [0.75] * 3
    dec = RSGA_2024_05_11.replace("2024 May 11", "2024 Dec 30").replace(
        "12 May-14 May", "31 Dec-02 Jan"
    )
    assert nt.parse_rsga(dec)["first_day"] == date(2024, 12, 31)
    assert nt.parse_rsga("nothing useful") is None
    tab, bad = nt.load_rsga({date(2024, 5, 11): _write(RSGA_2024_05_11)})
    assert (
        bad == 0
        and tab.index[0] == pd.Timestamp("2024-05-12", tz="UTC")
        and tab["noaa_m1"].iloc[0] == 0.95
    )
    assert nt.parse_rsga(RSGA_2024_05_11.replace("Class M", "Class Q")) is None


def test_region_features_values_and_growth():
    _, reg, _ = nt.parse_srs(SRS_2024_05_11)
    f = region_features(reg, reg.iloc[:5])
    assert f["r_n"] == 7 and f["r_n_delta"] == 1 and f["r_n_gamma"] == 2 and f["r_area_max"] == 2400
    assert f["r_n_new"] == 2 and f["r_area_growth"] == 30 + 140  # regions 3671 and 3672 are new
    empty = region_features(pd.DataFrame(columns=nt.REGION_COLUMNS), None)
    assert empty["r_n"] == 0 and empty["r_comp_max"] == -1
    assert np.isnan(region_features(None, None)["r_n"])


def test_fetch_range_caches_and_remembers_404(tmp_path, monkeypatch):
    calls = []

    class R:
        content = b":Issued: 2024 May 11 0030 UTC\n"

    def fake(url, **kw):
        calls.append(url)
        if "20240512" in url:
            from flarenet.sources.base import SourceUnavailable

            raise SourceUnavailable("HTTP 404 for x")
        return R()

    monkeypatch.setattr(nt, "http_get", fake)
    files, st = nt.fetch_range("srs", date(2024, 5, 10), date(2024, 5, 13), tmp_path, workers=2)
    assert st.downloaded == 3 and st.missing == 1 and len(files) == 3
    n = len(calls)
    files2, st2 = nt.fetch_range("srs", date(2024, 5, 10), date(2024, 5, 13), tmp_path, workers=2)
    assert len(calls) == n and st2.cached == 3 and st2.missing == 1  # nothing fetched twice


# ----- features and the end-to-end run ---------------------------------------------------------


def _cfg(tmp_path: Path, n_days: int) -> Config:
    cfg = Config()
    cfg.paths.data_dir, cfg.paths.runs_dir = str(tmp_path / "data"), str(tmp_path / "runs")
    cfg.sources = [
        SourceSpec(name="syn", kind="synthetic", role="train", params={"n_days": n_days, "seed": 3})
    ]
    cfg.daily.min_train_days, cfg.daily.step_days, cfg.daily.bootstrap_n = 120, 60, 100
    cfg.daily.oof_blocks = 3
    return cfg


@pytest.fixture(scope="module")
def synth_bundle(tmp_path_factory):
    from flarenet.daily.pipeline import _bundles

    cfg = _cfg(tmp_path_factory.mktemp("d"), 420)
    ctx, bundles = _bundles(cfg, False)
    return cfg, bundles["syn"]


def test_day_features_use_no_future_data(synth_bundle):
    cfg, b = synth_bundle
    t1 = build_day_table(b, None, 22)
    day = t1.index[200]
    issue = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=22)).timestamp()
    cut = int((issue - b.t0_unix) / 60) + 1
    b2 = type(b)(b.name, b.kind, b.df.copy(), b.t0_unix, dict(b.meta), b.flare_db.copy())
    b2.df.loc[b2.df.index[cut:], "sxr_raw"] = 9e-4  # a huge flare right after the issue time
    b2.df.loc[b2.df.index[cut:], "sxr"] = 9e-4
    t2 = build_day_table(b2, None, 22)
    cols = [c for c in t1.columns if c.startswith(("f_", "h_", "rot_"))]
    pd.testing.assert_series_equal(t1.loc[day, cols], t2.loc[day, cols], check_names=False)
    assert t1["y_m"].isin([0.0, 1.0]).all() and 0 < t1["y_m"].mean() < 1


def test_model_calibration_and_blocks_run():
    rng = np.random.default_rng(1)
    n = 700
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f_{i}" for i in range(6)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * X["f_0"] - 0.8)))).astype(int)
    d = pd.date_range("2021-01-01", periods=n, tz="UTC")
    m = DailyModel(DailyParams(oof_blocks=4)).fit(X.iloc[:500], y[:500], d[:500])
    p = m.predict_proba(X.iloc[500:])
    assert p.min() >= 0 and p.max() <= 1 and 0.05 < m.threshold < 0.95
    assert abs(p.mean() - y[500:].mean()) < 0.12  # calibrated on average


def _plant_files(cfg: Config, bundle, text_dir: Path) -> None:
    """Fake NOAA files in the real text format. The regions carry the signal that tomorrow is flare-active."""
    rng = np.random.default_rng(5)
    reg = bundle.df["active_regime"].to_numpy()
    t0 = pd.Timestamp(bundle.t0_unix, unit="s", tz="UTC")
    n_days = len(reg) // 1440
    for k in range(n_days):
        d = (t0 + pd.Timedelta(days=k)).date()
        tomorrow = (
            reg[min(len(reg) - 1440, (k + 1) * 1440) : (k + 2) * 1440].mean()
            if (k + 2) * 1440 <= len(reg)
            else 0.0
        )
        active = tomorrow > 0.3
        rows = []
        for j in range(int(rng.integers(2, 6)) + (3 if active else 0)):
            area = int(rng.integers(20, 200) + (900 if active and j == 0 else 0))
            mag = "Beta-Gamma-Delta" if active and j == 0 else rng.choice(["Alpha", "Beta"])
            z = "F" if active and j == 0 else rng.choice(["H", "C"])
            rows.append(
                f"{3000 + 10 * k + j:4d} N10E{10 + j:02d}   100 {area:04d} {z}kc  10   {12 if active and j == 0 else 2:02d} {mag}"
            )
        p = text_dir / "srs" / f"{d.year}" / f"{d:%Y%m%d}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f":Issued: {d:%Y} {d:%b} {d.day} 0030 UTC\nI.  Regions with Sunspots.  Locations Valid at 10/2400Z\n"
                     "Nmbr Location  Lo  Area  Z   LL   NN Mag Type\n" + "\n".join(rows) + "\nIA. H-alpha Plages\n")  # fmt: skip
        nxt = d + timedelta(days=1)
        pm = int(np.clip(30 + (35 if active else 0) + rng.normal(0, 12), 1, 99))
        q = text_dir / "rsga" / f"{d.year}" / f"{d:%Y%m%d}.txt"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(f":Issued: {d:%Y} {d:%b} {d.day} 2200 UTC\nIII.  Event probabilities {nxt.day:02d} {nxt:%b}-{nxt.day:02d} {nxt:%b}\n"
                     f"Class M    {pm:02d}/{pm:02d}/{pm:02d}\nClass X    05/05/05\n")  # fmt: skip


@pytest.mark.slow
def test_end_to_end_walk_forward_vs_noaa(tmp_path, monkeypatch):
    from flarenet.daily.pipeline import _bundles, run_daily
    from flarenet.sources.base import SourceUnavailable

    def no_network(url, **kw):  # days outside the planted files count as "NOAA did not publish"
        raise SourceUnavailable(f"HTTP 404 for {url}")

    monkeypatch.setattr(nt, "http_get", no_network)
    cfg = _cfg(tmp_path, 420)
    cfg.daily.text_dir = str(tmp_path / "noaa")
    _, bundles = _bundles(cfg, False)
    _plant_files(cfg, bundles["syn"], Path(cfg.daily.text_dir))
    res = run_daily(cfg)
    out = cfg.run_dir
    for f in (
        "daily_report.md",
        "daily_metrics.json",
        "predictions.csv",
        "day_table.csv",
        "daily_model.pkl",
    ):
        assert (out / f).exists(), f
    assert res["same_days"]["n_days"] > 100
    t = res["all_days"]
    # the planted region signal is visible only through the SRS features
    assert t["FlareNet + sunspot regions"]["AUC"] > t["FlareNet flux only"]["AUC"] + 0.02
    assert t["FlareNet + sunspot regions"]["AUC"] > 0.6
    assert set(res["paired_differences"]) >= {
        "AUC (threshold-free)",
        "Brier vs recalibrated NOAA (lower is better)",
        "TSS, each at its own validated threshold (fair)",
    }
    assert (
        "NOAA recalibrated on earlier days (same treatment as FlareNet)"
        in res["same_days"]["table"]
    )
    assert res["_markdown"].startswith("> **SYNTHETIC DATA")
    assert (out / "daily_model_live.pkl").exists()
    assert "Read before quoting" in res["_markdown"]
    # every out-of-sample day is later than the first training block
    pred = pd.read_csv(out / "predictions.csv", index_col=0, parse_dates=True)
    assert pred.index.min() >= pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(
        days=cfg.daily.min_train_days
    )


def test_reference_recalibration_fixes_a_biased_forecaster():
    from flarenet.daily.model import apply_reference, recalibrate_reference

    rng = np.random.default_rng(2)
    n = 1500
    true_p = np.clip(rng.beta(1.2, 6, n), 0.01, 0.95)
    y = (rng.random(n) < true_p).astype(int)
    biased = np.clip(true_p * 0.4, 0.001, 0.99)  # a forecaster that is too low by a factor of 2.5
    ref = recalibrate_reference(biased[:1000], y[:1000])
    assert ref is not None
    cal, thr = apply_reference(ref, biased[1000:])
    assert 0.0 < thr < 1.0
    brier = lambda p: float(np.mean((p - y[1000:]) ** 2))  # noqa: E731
    assert brier(cal) < brier(biased[1000:])
    assert abs(cal.mean() - y[1000:].mean()) < abs(biased[1000:].mean() - y[1000:].mean())
    assert recalibrate_reference(biased[:50], y[:50]) is None  # too little to fit
    none_p, none_thr = apply_reference(None, biased[:5])
    assert np.isnan(none_p).all() and np.isnan(none_thr)


def test_feature_columns_drop_all_nan(synth_bundle):
    _, b = synth_bundle
    t = build_day_table(b, None, 22)
    cols = feature_columns(t)
    assert cols and not any(c.startswith("r_") for c in cols)


def test_predict_tomorrow_runs_and_warns(synth_bundle, tmp_path):
    from flarenet.daily.live import load_daily_model, predict_tomorrow

    cfg, b = synth_bundle
    tab = build_day_table(b, None, 22)
    cols = [c for c in feature_columns(tab, ("f_", "h_")) if not c.startswith("rot_")]
    m = DailyModel(DailyParams(oof_blocks=3)).fit(tab[cols], tab["y_m"].to_numpy(), tab.index)
    pkg = {
        "model": m,
        "columns": cols,
        "target": "M",
        "issue_hour_utc": 22,
        "trained_days": len(tab),
        "trained_until": "x",
    }
    p = tmp_path / "m.pkl"
    import pickle

    p.write_bytes(pickle.dumps(pkg))
    got = predict_tomorrow(load_daily_model(p), b, srs_text=None)
    assert 0.0 <= got["probability"] <= 1.0 and got["regions_used"] is False
    assert got["for_day"] > got["issued_utc"][:10]
    # a model that uses sunspot regions says plainly that it got none
    pkg2 = {**pkg, "columns": cols + ["r_n"]}
    got2 = predict_tomorrow(pkg2, b, srs_text=None)
    assert any("sunspot" in w for w in got2["warnings"])


# the live "3-day Space Weather Predictions" file, as NOAA served it on 2026-10-24 (flare section kept)
DAYPRE_2026_10_24 = """:Product: 3-day Space Weather Predictions daypre.txt
:Issued: 2026 Oct 24 2200 UTC
# Prepared by the US Dept. of Commerce, NOAA, Space Weather Prediction Center
:Prediction_dates:   2026 Oct 25   2026 Oct 26   2026 Oct 27
:Geomagnetic_A_indices:
A_Fredericksburg         14             8             6
# Solar
:10cm_flux:
                         92            92            88
#
:Whole_Disk_Flare_Prob:
Class_M                  10            10            10
Class_X                   1             1             1
Proton                    1             1             1
#
# Region Flare Probabilities for 2026 Oct 25
# Region     Class C      M             X             P
:Reg_Prob: 2026 Oct 24
  4557          0         0             0             0
  4560         60        10             1             1
"""


def test_parse_daypre_real_live_file():
    r = nt.parse_daypre(DAYPRE_2026_10_24)
    assert r["first_day"] == date(2026, 10, 25) and r["m"] == [0.10] * 3 and r["x"] == [0.01] * 3
    assert r["issued"] == pd.Timestamp("2026-10-24 22:00", tz="UTC")
    assert (
        nt.parse_flare_forecast(DAYPRE_2026_10_24)["m"] == [0.10] * 3
    )  # one entry point for both products
    assert nt.parse_flare_forecast(RSGA_2024_05_11)["m"] == [0.95] * 3
    assert nt.parse_daypre("<html>nothing</html>") is None
    assert nt.parse_daypre(DAYPRE_2026_10_24.replace("Class_M", "Class_Z")) is None

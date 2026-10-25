"""NASA / NOAA data paths, tested without the internet.

Files are written locally in the real GOES-R NetCDF / SWPC JSON / DONKI JSON layouts; the network
code (directory listing, download, retries, query parameters) talks to a throw-away local HTTP
server that mimics the NCEI directory index, the SWPC feed and the DONKI endpoint.
"""

from __future__ import annotations

import functools
import http.server
import threading

import numpy as np
import pandas as pd
import pytest
from conftest import (
    T0_GOES,
    donki_records,
    goes_like,
    json_dump,
    swpc_records,
    write_goes_netcdf,
)

from flarenet.core.config import SourceSpec
from flarenet.pipelines.preprocessing.flare_db import build_flare_db
from flarenet.pipelines.preprocessing.pipeline import process_raw_series
from flarenet.sources import make_source
from flarenet.sources.base import SourceUnavailable
from flarenet.sources.catalogs import make_catalog
from flarenet.sources.nasa.donki import parse_donki_flares
from flarenet.sources.nasa.goes_file import times_to_unix
from flarenet.sources.nasa.goes_swpc import parse_swpc_records
from flarenet.sources.nasa.http import http_get
from flarenet.sources.nasa.netcdf import cf_time_to_unix, read_goes_xrs_netcdf


@pytest.fixture(autouse=True)
def _no_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep pytest output clean
        pass


@pytest.fixture
def serve(tmp_path):
    """Serve ``tmp_path/www`` over HTTP; returns (document root, base url)."""
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(_Quiet, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ---- NetCDF reader ---------------------------------------------------------------------------


def test_cf_time_units():
    assert (
        cf_time_to_unix(np.array([0.0, 60.0]), "seconds since 2000-01-01 12:00:00")[0]
        == 946_728_000.0
    )
    assert cf_time_to_unix(np.array([1.0]), "days since 1970-01-01")[0] == 86400.0
    assert cf_time_to_unix(np.array([2.0]), "minutes since 1970-01-01T00:00:00Z")[0] == 120.0
    with pytest.raises(SourceUnavailable):
        cf_time_to_unix(np.array([1.0]), "fortnights since 1970-01-01")


def test_netcdf_reader_roundtrip_fill_values_and_flags(tmp_path):
    t = T0_GOES + np.arange(10) * 60.0
    soft = np.full(10, 2e-6)
    soft[3] = np.nan  # written as the fill value -9999
    hard = np.full(10, 1e-7)
    flag = np.zeros(10, dtype=int)
    flag[5] = 2
    p = write_goes_netcdf(tmp_path / "x.nc", t, soft, hard, flag=flag)
    tt, s, h = read_goes_xrs_netcdf(p, use_quality_flags=True)
    assert np.allclose(tt, t) and np.isnan(s[3]) and np.isnan(s[5]) and np.isfinite(h).all()
    _, s_noflag, _ = read_goes_xrs_netcdf(p, use_quality_flags=False)
    assert np.isfinite(s_noflag[5]) and np.isnan(s_noflag[3])  # fill value is never real flux


def test_netcdf_reader_rejects_unknown_layout(tmp_path):
    import netCDF4

    p = tmp_path / "bad.nc"
    with netCDF4.Dataset(str(p), "w") as ds:
        ds.createDimension("n", 2)
        ds.createVariable("something", "f8", ("n",))
    with pytest.raises(SourceUnavailable, match="expected a time variable"):
        read_goes_xrs_netcdf(p)


# ---- local-file source (bring your own NASA/NOAA data) ------------------------------------------


def test_file_source_netcdf_glob_merges_days_and_dedupes(goes_files):
    d, t, soft, _hard, _ = goes_files
    src = make_source(
        SourceSpec(name="g", kind="nasa_goes_file", params={"paths": [str(d / "*.nc")]})
    )
    raw = src.fetch()
    ch = raw.channels["sxr"]
    assert len(ch.t_unix) == np.isfinite(soft).sum() or len(ch.t_unix) == len(t)
    assert np.all(np.diff(ch.t_unix) > 0) and "hxr" in raw.channels
    # fingerprint follows the files: touching one changes it
    fp = src.fingerprint()
    next(d.glob("*.nc")).touch()
    assert (
        make_source(
            SourceSpec(name="g", kind="nasa_goes_file", params={"paths": [str(d / "*.nc")]})
        ).fingerprint()
        != fp
    )


def test_file_source_csv_with_custom_columns_and_mjd(tmp_path):
    n = 200
    t = T0_GOES + np.arange(n) * 60.0
    df = pd.DataFrame(
        {"mjd": t / 86400 + 40587, "B_AVG": np.full(n, 3e-6), "A_AVG": np.full(n, 2e-7)}
    )
    df.loc[10, "B_AVG"] = -99999.0  # sentinel -> NaN
    p = tmp_path / "legacy.csv"
    df.to_csv(p, index=False)
    spec = SourceSpec(name="c", kind="nasa_goes_file", params={
        "paths": [str(p)], "time_column": "mjd", "sxr_column": "B_AVG", "hxr_column": "A_AVG"})  # fmt: skip
    raw = make_source(spec).fetch()
    assert np.isnan(raw.channels["sxr"].values[10]) and raw.channels["sxr"].t_unix[
        0
    ] == pytest.approx(T0_GOES, abs=1)


def test_file_source_errors_are_helpful(tmp_path):
    with pytest.raises(SourceUnavailable, match="no files match"):
        make_source(
            SourceSpec(
                name="x", kind="nasa_goes_file", params={"paths": [str(tmp_path / "none*.nc")]}
            )
        ).fetch()
    p = tmp_path / "a.csv"
    pd.DataFrame({"time": [1, 2], "other": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(SourceUnavailable, match="'xrsb' not found"):
        make_source(SourceSpec(name="x", kind="nasa_goes_file", params={"paths": [str(p)]})).fetch()


def test_times_to_unix_auto_detects_formats():
    t = 1.7e9
    assert times_to_unix(pd.Series([t, t + 60]), "auto")[0] == t
    assert times_to_unix(pd.Series([t * 1000]), "auto")[0] == pytest.approx(t)
    assert times_to_unix(pd.Series([60000.5]), "auto")[0] == pytest.approx(
        (60000.5 - 40587) * 86400
    )
    assert times_to_unix(pd.Series(["2024-05-01T00:00:00Z"]), "auto")[0] == T0_GOES


# ---- SWPC feed ---------------------------------------------------------------------------------


def test_swpc_parse_both_channels_and_duplicates():
    t, soft, hard, _ = goes_like(1)
    recs = swpc_records(t[:300], soft[:300], hard[:300])
    recs += recs[:10]  # duplicated timestamps from a re-published feed
    ch = parse_swpc_records(recs)
    assert set(ch) == {"sxr", "hxr"} and np.all(np.diff(ch["sxr"].t_unix) > 0)
    assert set(parse_swpc_records(recs, satellite_number=18)) == {"sxr", "hxr"}
    with pytest.raises(SourceUnavailable):
        parse_swpc_records(recs, satellite_number=16)  # feed only holds GOES-18
    with pytest.raises(SourceUnavailable):
        parse_swpc_records([])
    with pytest.raises(SourceUnavailable, match="lack column"):
        parse_swpc_records([{"time_tag": "2024-05-01T00:00:00Z"}])


def test_swpc_source_from_saved_json_and_over_http(tmp_path, serve):
    root, url = serve
    t, soft, hard, _ = goes_like(2)
    recs = swpc_records(t[:1500], soft[:1500], hard[:1500])
    saved = json_dump(recs, tmp_path / "feed.json")
    raw = make_source(
        SourceSpec(name="s", kind="nasa_goes_swpc", role="eval", params={"json_path": str(saved)})
    ).fetch()
    (root / "json" / "goes" / "primary").mkdir(parents=True)
    json_dump(recs, root / "json" / "goes" / "primary" / "xrays-1-day.json")
    live = make_source(SourceSpec(name="s", kind="nasa_goes_swpc", role="eval",
                                  params={"base_url": url, "product": "xrays-1-day"}))  # fmt: skip
    raw2 = live.fetch()
    assert np.array_equal(raw.channels["sxr"].t_unix, raw2.channels["sxr"].t_unix)
    assert live.fingerprint().endswith("-live")  # a live feed is never served from cache
    with pytest.raises(SourceUnavailable, match="product must be"):
        make_source(
            SourceSpec(name="s", kind="nasa_goes_swpc", params={"product": "bogus"})
        ).fetch()


# ---- NCEI archive (directory listing + download + cache) ----------------------------------------


def _ncei_layout(root, files):
    d = root / "goes16" / "l2" / "data" / "xrsf-l2-avg1m_science" / "2024" / "05"
    d.mkdir(parents=True)
    for f in files:
        (d / f.name).write_bytes(f.read_bytes())
        # an older version of the same day must lose against v2-2-0
        (d / f.name.replace("v2-2-0", "v1-0-0")).write_bytes(b"not a netcdf file")
    return d


def test_ncei_downloads_picks_latest_version_and_reuses_cache(tmp_path, serve, goes_files):
    root, url = serve
    d, *_ = goes_files
    _ncei_layout(root, sorted(d.glob("*.nc"))[:3])
    local = tmp_path / "dl"
    params = {"satellite": 16, "start": "2024-05-01", "end": "2024-05-03", "local_dir": str(local),
              "base_url": url + "/goes{sat}/l2/data/xrsf-l2-avg1m_science/{yyyy}/{mm}/"}  # fmt: skip
    raw = make_source(SourceSpec(name="n", kind="nasa_goes_ncei", params=params)).fetch()
    assert len(list(local.glob("*.nc"))) == 3 and not list(local.glob("*.part"))
    assert raw.channels["sxr"].t_unix[0] >= T0_GOES - 1 and raw.meta["missing_days"] == []
    # second call works offline: everything comes from local_dir
    offline = make_source(
        SourceSpec(
            name="n",
            kind="nasa_goes_ncei",
            params={**params, "offline": True, "base_url": "http://127.0.0.1:1/"},
        )
    )
    assert len(offline.fetch().channels["sxr"].t_unix) == len(raw.channels["sxr"].t_unix)
    assert len(offline.input_files()) == 3


def test_ncei_missing_days_tolerated_then_fail(tmp_path, serve, goes_files):
    root, url = serve
    d, *_ = goes_files
    _ncei_layout(root, sorted(d.glob("*.nc"))[:2])
    base = url + "/goes{sat}/l2/data/xrsf-l2-avg1m_science/{yyyy}/{mm}/"
    ok = {"satellite": 16, "start": "2024-05-01", "end": "2024-05-04", "local_dir": str(tmp_path / "a"),
          "base_url": base, "max_missing_days": 2}  # fmt: skip
    raw = make_source(SourceSpec(name="n", kind="nasa_goes_ncei", params=ok)).fetch()
    assert raw.meta["missing_days"] == ["2024-05-03", "2024-05-04"]
    with pytest.raises(SourceUnavailable, match="more than 1 days unavailable"):
        make_source(SourceSpec(name="n", kind="nasa_goes_ncei",
                               params={**ok, "local_dir": str(tmp_path / "b"), "max_missing_days": 1})).fetch()  # fmt: skip
    with pytest.raises(SourceUnavailable, match="start and params.end"):
        make_source(SourceSpec(name="n", kind="nasa_goes_ncei", params={})).fetch()


def test_http_errors_and_retry_message(serve):
    _root, url = serve
    with pytest.raises(SourceUnavailable, match="HTTP 404"):
        http_get(url + "/nope.json", retries=1)
    with pytest.raises(SourceUnavailable, match="network access"):
        http_get("http://127.0.0.1:1/x", retries=1, timeout=1)


# ---- DONKI --------------------------------------------------------------------------------------


def test_parse_donki_flares_drops_incomplete_and_dedupes():
    recs = [
        {
            "flrID": "a",
            "beginTime": "2024-05-01T10:00Z",
            "peakTime": "2024-05-01T10:10Z",
            "endTime": None,
            "classType": "M1.2",
        },
        {
            "flrID": "a",
            "beginTime": "2024-05-01T10:00Z",
            "peakTime": "2024-05-01T10:10Z",
            "endTime": None,
            "classType": "M1.2",
        },
        {"flrID": "b", "beginTime": None, "peakTime": None, "endTime": None, "classType": "C1.0"},
        {
            "flrID": "c",
            "beginTime": "2024-05-01T08:00Z",
            "peakTime": "2024-05-01T08:05Z",
            "endTime": "2024-05-01T08:20Z",
            "classType": "",
        },
    ]
    df = parse_donki_flares(recs)
    assert list(df.flr_id) == ["a"] and np.isnan(df.end_unix[0])
    assert len(parse_donki_flares([])) == 0


def test_donki_catalog_over_http_chunks_long_ranges(tmp_path, serve, synthetic_small):
    root, url = serve
    _df, _ev, fdb = synthetic_small
    recs = donki_records(fdb, T0_GOES)
    json_dump(recs, root / "FLR")
    cat = make_catalog("donki", {"base_url": url + "/FLR", "api_key": "k"})
    out = cat.fetch(T0_GOES, T0_GOES + 30 * 86400)
    assert 0 < len(out) <= len(recs) and out.begin_unix.is_monotonic_increasing
    # a 70-day request makes several 30-day calls; duplicates across chunks are removed
    long = cat.fetch(T0_GOES, T0_GOES + 70 * 86400)
    assert long.flr_id.is_unique


# ---- GOES data through preprocessing: labels are calibrated classes ---------------------------


def test_goes_netcdf_to_labelled_lightcurve_matches_truth_events(goes_files):
    d, *_rest, truth = goes_files
    spec = SourceSpec(name="g", kind="nasa_goes_file", params={"paths": [str(d / "*.nc")]})
    raw = make_source(spec).fetch()
    bundle = process_raw_series(spec, raw, seed=0)
    assert bundle.t0_unix == pytest.approx(T0_GOES, abs=60)
    assert set(["time_min", "sxr", "hxr", "flare_class", "flare_id"]) <= set(bundle.df.columns)
    truth_db = build_flare_db(truth, label="truth")
    truth_mx = truth_db[truth_db.class_idx >= 3]
    got_mx = bundle.flare_db[bundle.flare_db.class_idx >= 3]
    assert len(got_mx) > 0
    # The synthetic Sun is ~17 M/X flares/day and they overlap, so a NOAA-style detector merges
    # events (fewer, longer). What must hold: nearly every true M/X peak lies inside a detected event.
    off = (bundle.t0_unix - T0_GOES) / 60.0  # detector minutes are relative to the series start
    covered = [
        (
            (got_mx.onset_min + off - 5 <= r.peak_min) & (r.peak_min <= got_mx.end_min + off + 5)
        ).any()
        for r in truth_mx.itertuples()
    ]
    assert np.mean(covered) > 0.8
    assert bundle.meta["label_kind"].startswith("GOES-calibrated")


def test_download_file_retries_when_connection_breaks_mid_file(tmp_path, monkeypatch):
    import requests

    from flarenet.sources.nasa import http as H

    calls = {"n": 0}

    class Resp:
        def iter_content(self, chunk_size):
            calls["n"] += 1
            yield b"abc"
            if calls["n"] == 1:
                raise requests.exceptions.ChunkedEncodingError("Connection broken: BrokenPipe")
            yield b"def"

    monkeypatch.setattr(H, "http_get", lambda url, **kw: Resp())
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    out = H.download_file("http://x/f.nc", tmp_path / "f.nc", retries=3)
    assert out.read_bytes() == b"abcdef" and calls["n"] == 2
    assert not list(tmp_path.glob("*.part"))

    class Broken:
        def iter_content(self, chunk_size):
            yield b"abc"
            raise requests.exceptions.ChunkedEncodingError("Connection broken")

    monkeypatch.setattr(H, "http_get", lambda url, **kw: Broken())
    import pytest

    from flarenet.sources.base import SourceUnavailable

    with pytest.raises(SourceUnavailable, match="failed after 2 attempts"):
        H.download_file("http://x/g.nc", tmp_path / "g.nc", retries=2)
    assert not (tmp_path / "g.nc").exists()

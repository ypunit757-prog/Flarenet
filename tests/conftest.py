"""Shared fixtures. Real NASA/NOAA servers are never contacted: GOES / DONKI / SWPC payloads are
generated locally from the synthetic lightcurve (which lives on the same W/m^2 scale)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from flarenet.core.config import Config, SourceSpec
from flarenet.core.utils import to_unix_seconds
from flarenet.pipelines.preprocessing.flare_db import build_flare_db
from flarenet.sources.synthetic import generate_synthetic

T0_GOES = 1_714_521_600.0  # 2024-05-01T00:00:00Z
J2000_UNIX = 946_728_000.0  # 2000-01-01T12:00:00Z


@pytest.fixture(scope="session")
def synthetic_small():
    df, events = generate_synthetic(n_days=30, seed=7)
    return df, events, build_flare_db(df, label="synthetic")


def make_lightcurve(n: int = 1000) -> pd.DataFrame:
    """Flat, gap-free, flare-free lightcurve on the processed-bundle schema."""
    return pd.DataFrame(
        {
            "time_min": np.arange(n, dtype=float),
            "sxr": np.full(n, 3e-9),
            "hxr": np.full(n, 5e-10),
            "sxr_raw": np.full(n, 3e-9),
            "hxr_raw": np.full(n, 5e-10),
            "flare_class": np.zeros(n, dtype=int),
            "flare_id": -np.ones(n, dtype=int),
            "active_regime": np.zeros(n, dtype=int),
        }
    )


def goes_like(n_days: int = 20, seed: int = 3):
    """(t_unix, soft W/m^2, hard W/m^2, truth DataFrame) with 1-minute cadence and gaps."""
    df, _ = generate_synthetic(n_days, seed)
    t = T0_GOES + df["time_min"].to_numpy() * 60.0
    return t, df["sxr"].to_numpy().copy(), df["hxr"].to_numpy().copy(), df


def write_goes_netcdf(
    path, t_unix, soft, hard, flag=None, units="seconds since 2000-01-01 12:00:00"
):
    import netCDF4

    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("time", len(t_unix))
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = units
        tv[:] = (t_unix - J2000_UNIX) if units.startswith("seconds since 2000") else t_unix
        for name, vals in (("xrsb_flux", soft), ("xrsa_flux", hard)):
            v = ds.createVariable(name, "f8", ("time",), fill_value=-9999.0)
            v[:] = np.where(np.isfinite(vals), vals, -9999.0)
        if flag is not None:
            f = ds.createVariable("xrsb_flag", "i4", ("time",))
            f[:] = flag
    return path


def write_goes_days(directory, t_unix, soft, hard, satellite=16):
    """One NCEI-style file per UTC day: sci_xrsf-l2-avg1m_g16_dYYYYMMDD_v2-2-0.nc."""
    directory.mkdir(parents=True, exist_ok=True)
    day = np.floor(t_unix / 86400).astype(int)
    paths = []
    for d in np.unique(day):
        m = day == d
        ymd = pd.Timestamp(d * 86400, unit="s").strftime("%Y%m%d")
        p = directory / f"sci_xrsf-l2-avg1m_g{satellite}_d{ymd}_v2-2-0.nc"
        write_goes_netcdf(p, t_unix[m], soft[m], hard[m])
        paths.append(p)
    return paths


def swpc_records(t_unix, soft, hard, satellite=18):
    """NOAA SWPC xrays JSON records (two energy bands per minute)."""
    rows = []
    for ti, s, h in zip(t_unix, soft, hard, strict=True):
        tag = pd.Timestamp(ti, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")
        if np.isfinite(s):
            rows.append(
                {"time_tag": tag, "satellite": satellite, "flux": float(s), "energy": "0.1-0.8nm"}
            )
        if np.isfinite(h):
            rows.append(
                {"time_tag": tag, "satellite": satellite, "flux": float(h), "energy": "0.05-0.4nm"}
            )
    return rows


def donki_records(flare_db: pd.DataFrame, t0_unix: float):
    """DONKI FLR-style JSON for the M/X/C flares of an event table (class from the peak flux)."""
    from flarenet.core.goes import flux_to_class_type

    iso = lambda m: pd.Timestamp(t0_unix + m * 60, unit="s").strftime("%Y-%m-%dT%H:%MZ")  # noqa: E731
    out = []
    for r in flare_db[flare_db.class_idx >= 2].itertuples():
        out.append({"flrID": f"F{r.event_id}", "beginTime": iso(r.onset_min), "peakTime": iso(r.peak_min),
                    "endTime": iso(r.end_min), "classType": flux_to_class_type(r.peak_sxr)})  # fmt: skip
    return out


@pytest.fixture(scope="session")
def goes_files(tmp_path_factory):
    """20 days of GOES-like NetCDF files + the arrays they were written from."""
    t, soft, hard, truth = goes_like(20)
    d = tmp_path_factory.mktemp("goes_nc")
    write_goes_days(d, t, soft, hard)
    return d, t, soft, hard, truth


# ---- fake Aditya-L1 FITS (SoLEXS + HEL1OS) ------------------------------------------------

UNIX_T0 = 1_782_000_000.0  # arbitrary observation start (2026-06-21 UTC)


def _dummy_hdu():
    return fits.BinTableHDU.from_columns([fits.Column(name="X", format="D", array=np.zeros(1))])


def write_fake_aditya(root, hours: float = 14.0, seed: int = 0):
    """SoLEXS ``*.lc`` (HDU 1: TIME unix s, COUNTS) and two HEL1OS orbit dirs (HDU 5: MJD, CTR)."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    t = UNIX_T0 + np.arange(n, dtype=float)
    minute = np.arange(n) / 60.0
    soft = np.full(n, 100.0)
    hard = np.full(n, 20.0)
    for onset_min, amp in [(200, 3000.0), (450, 800.0), (600, 20000.0)]:
        burst = np.where(minute >= onset_min, np.exp(-(minute - onset_min) / 12.0), 0.0)
        soft += amp * burst
        hard += 0.1 * amp * burst
    soft, hard = rng.poisson(soft).astype(float), rng.poisson(hard).astype(float)

    root.mkdir(parents=True, exist_ok=True)
    solexs = root / "AL1_SOLEXS_fake_SDD2_L1.lc"
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="COUNTS", format="D", array=soft),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        solexs, overwrite=True
    )

    mjd = t / 86400.0 + 40587.0
    half = n // 2
    dirs = []
    for k, sl in enumerate([slice(0, half), slice(half, n)]):
        d = root / f"HLS_orbit{k}"
        (d / "czt").mkdir(parents=True, exist_ok=True)
        hcols = [
            fits.Column(name="MJD", format="D", array=mjd[sl]),
            fits.Column(name="CTR", format="D", array=hard[sl]),
        ]
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                *[_dummy_hdu() for _ in range(4)],
                fits.BinTableHDU.from_columns(hcols),
            ]
        )
        hdul.writeto(d / "czt" / "lightcurve_czt1.fits", overwrite=True)
        dirs.append(str(d))
    return str(solexs), dirs


# ---- config helpers -------------------------------------------------------------------------


def tiny_cfg(tmp_path, sources: list[SourceSpec] | None = None, epochs: int = 2) -> Config:
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.runs_dir = str(tmp_path / "runs")
    cfg.run_name = "t"
    cfg.data.block_days = 2
    cfg.training.epochs = epochs
    cfg.training.device = "cpu"
    cfg.training.batch_size = 64
    cfg.evaluation.bootstrap_n = 20
    cfg.sources = sources or [SourceSpec(params={"n_days": 30})]
    cfg.validate()
    return cfg


def json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


__all__ = ["to_unix_seconds"]

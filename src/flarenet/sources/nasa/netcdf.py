"""Reader for NOAA/NCEI GOES-R XRS level-2 science files (``sci_xrsf-l2-avg1m_g16_d*.nc``)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.utils import get_logger
from flarenet.sources.base import SourceUnavailable

log = get_logger(__name__)

# Variable-name candidates, first match wins. GOES-R L2: xrsb_flux = 0.1-0.8 nm (long channel,
# the one flare classes are defined on), xrsa_flux = 0.05-0.4 nm (short channel).
SOFT_VARS = ["xrsb_flux", "b_flux", "xrsb", "B_AVG", "flux_b"]
HARD_VARS = ["xrsa_flux", "a_flux", "xrsa", "A_AVG", "flux_a"]
TIME_VARS = ["time", "TIME", "Time"]
J2000 = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")
_UNITS_RE = re.compile(r"^\s*(\w+)\s+since\s+(.+?)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {
    "second": 1.0, "seconds": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0,
    "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
}  # fmt: skip


def cf_time_to_unix(values: np.ndarray, units: str | None) -> np.ndarray:
    """Convert CF-convention time values ('seconds since 2000-01-01 12:00:00') to unix seconds."""
    if not units:
        log.warning("time variable has no units attribute; assuming seconds since J2000")
        base, scale = J2000, 1.0
    else:
        m = _UNITS_RE.match(units)
        if not m or m.group(1).lower() not in _UNIT_SECONDS:
            raise SourceUnavailable(f"Unsupported NetCDF time units: {units!r}")
        text = re.sub(r"\s*(UTC|Z)$", "", m.group(2).strip(), flags=re.IGNORECASE)
        base = pd.Timestamp(text.replace(" ", "T"), tz="UTC")
        scale = _UNIT_SECONDS[m.group(1).lower()]
    base_unix = (base - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    return base_unix + np.asarray(values, dtype=float) * scale


def _first_present(names: list[str], available) -> str | None:
    for n in names:
        if n in available:
            return n
    return None


def read_goes_xrs_netcdf(
    path: str | Path, use_quality_flags: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (t_unix, soft_flux, hard_flux or None) in W/m^2; bad/flagged samples are NaN."""
    try:
        import netCDF4
    except ImportError as e:
        raise SourceUnavailable(
            "Reading NetCDF needs the optional dependency: pip install 'flarenet[netcdf]'"
        ) from e

    with netCDF4.Dataset(str(path)) as ds:
        names = set(ds.variables)
        tname = _first_present(TIME_VARS, names)
        sname = _first_present(SOFT_VARS, names)
        hname = _first_present(HARD_VARS, names)
        if tname is None or sname is None:
            raise SourceUnavailable(
                f"{path}: expected a time variable {TIME_VARS} and a soft-flux variable "
                f"{SOFT_VARS}; found {sorted(names)}"
            )
        tvar = ds.variables[tname]
        t_unix = cf_time_to_unix(
            np.ma.filled(tvar[:].astype(float), np.nan), getattr(tvar, "units", None)
        )

        def read_flux(var_name: str) -> np.ndarray:
            flux = np.ma.filled(ds.variables[var_name][:].astype("float64"), np.nan)
            flux = np.where(np.isfinite(flux) & (flux > 0), flux, np.nan)
            flag_name = re.sub(r"_?flux$", "", var_name) + "_flag"
            if use_quality_flags and flag_name in names:
                flag = np.ma.filled(ds.variables[flag_name][:], 0)
                flux = np.where(np.asarray(flag) != 0, np.nan, flux)
            return flux

        soft = read_flux(sname)
        hard = read_flux(hname) if hname else None
    return t_unix, soft, hard

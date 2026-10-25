"""NOAA SWPC GOES X-ray JSON feeds (the live, near-real-time GOES XRS data NASA/NOAA publish).

Endpoint pattern: ``https://services.swpc.noaa.gov/json/goes/<primary|secondary>/<product>.json``
with products ``xrays-6-hour``, ``xrays-1-day``, ``xrays-3-day``, ``xrays-7-day``. Each record has
``time_tag`` (UTC), ``satellite``, ``flux`` (W/m^2) and ``energy`` (``0.1-0.8nm`` = soft channel the
flare classes are defined on, ``0.05-0.4nm`` = short channel).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.core.utils import get_logger, to_unix_seconds
from flarenet.sources.base import DataSource, SourceUnavailable
from flarenet.sources.nasa.http import get_json

log = get_logger(__name__)

SWPC_PRODUCTS = ("xrays-6-hour", "xrays-1-day", "xrays-3-day", "xrays-7-day")
SOFT_ENERGY = "0.1-0.8nm"
HARD_ENERGY = "0.05-0.4nm"


def parse_swpc_records(
    records: list[dict], satellite_number: int | None = None
) -> dict[str, Channel]:
    """Turn SWPC X-ray JSON records into ``{"sxr": Channel, "hxr": Channel}`` (hxr optional)."""
    if not records:
        raise SourceUnavailable("SWPC X-ray feed returned no records")
    df = pd.DataFrame(records)
    for col in ("time_tag", "flux", "energy"):
        if col not in df.columns:
            raise SourceUnavailable(f"SWPC records lack column {col!r}; got {list(df.columns)}")
    if satellite_number is not None and "satellite" in df.columns:
        df = df[pd.to_numeric(df["satellite"], errors="coerce") == satellite_number]
    t = to_unix_seconds(df["time_tag"])
    flux = pd.to_numeric(df["flux"], errors="coerce").to_numpy(dtype=float)
    flux = np.where(np.isfinite(flux) & (flux > 0), flux, np.nan)
    energy = df["energy"].astype(str).str.replace(" ", "").to_numpy()

    out: dict[str, Channel] = {}
    for key, name in (("sxr", SOFT_ENERGY), ("hxr", HARD_ENERGY)):
        m = (energy == name) & np.isfinite(t)
        if not m.any():
            continue
        tt, ff = t[m], flux[m]
        order = np.argsort(tt, kind="stable")
        tt, ff = tt[order], ff[order]
        keep = np.concatenate([[True], np.diff(tt) > 0])  # drop duplicate timestamps
        out[key] = Channel(tt[keep], ff[keep])
    if "sxr" not in out:
        raise SourceUnavailable(f"No {SOFT_ENERGY} records in SWPC feed (energies: {set(energy)})")
    return out


def fetch_swpc_series(
    name: str,
    product: str = "xrays-7-day",
    satellite: str = "primary",
    satellite_number: int | None = None,
    base_url: str = "https://services.swpc.noaa.gov",
    json_path: str | None = None,
    timeout_s: float = 30.0,
    retries: int = 3,
    describe: dict | None = None,
) -> RawSeries:
    """Fetch (or read from ``json_path``) a SWPC feed and wrap it as a RawSeries."""
    if json_path:
        with open(json_path) as f:
            records = json.load(f)
    else:
        if product not in SWPC_PRODUCTS:
            raise SourceUnavailable(f"product must be one of {SWPC_PRODUCTS}, got {product!r}")
        url = f"{base_url.rstrip('/')}/json/goes/{satellite}/{product}.json"
        records = get_json(url, timeout=timeout_s, retries=retries)
    channels = parse_swpc_records(records, satellite_number)
    meta = {**(describe or {}), "product": product, "satellite": satellite}
    log.info(
        "[%s] SWPC %s: %d soft samples, hard channel %s",
        name,
        product,
        len(channels["sxr"]),
        "present" if "hxr" in channels else "absent",
    )
    return RawSeries(name, "nasa_goes_swpc", channels, meta)


@SOURCES.register("nasa_goes_swpc")
class GoesSwpcSource(DataSource):
    """Last 6 hours to 7 days of GOES XRS data from NOAA SWPC (calibrated, W/m^2)."""

    @dataclass
    class Params:
        product: str = "xrays-7-day"
        satellite: str = "primary"  # primary | secondary
        satellite_number: int | None = None  # keep only this GOES number (e.g. 18)
        base_url: str = "https://services.swpc.noaa.gov"
        json_path: str | None = None  # read a saved feed instead of the network
        timeout_s: float = 30.0
        retries: int = 3

    default_labeler = "goes_events"
    label_kind = "GOES-calibrated classes from NOAA-style event detection on the 0.1-0.8 nm flux"
    soft_name = "GOES XRS-B (0.1-0.8 nm)"
    hard_name = "GOES XRS-A (0.05-0.4 nm)"

    def input_files(self) -> list[Path]:
        return [Path(self.params.json_path)] if self.params.json_path else []

    def fingerprint(self) -> str:
        # A live feed changes every minute: never treat it as cached unless read from a file.
        base = super().fingerprint()
        return base if self.params.json_path else base + "-live"

    def fetch(self) -> RawSeries:
        p = self.params
        return fetch_swpc_series(
            self.name,
            p.product,
            p.satellite,
            p.satellite_number,
            p.base_url,
            p.json_path,
            p.timeout_s,
            p.retries,
            self.describe(),
        )

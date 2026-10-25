"""Historical GOES-R XRS 1-minute science data from NOAA NCEI (NetCDF), one file per day.

Typical use: pull a solar-maximum stretch (e.g. May 2024) to train on real, calibrated GOES
labels. Files are downloaded once into ``local_dir`` and re-used.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.core.utils import get_logger
from flarenet.sources.base import DataSource, SourceUnavailable
from flarenet.sources.nasa.http import download_file, list_directory
from flarenet.sources.nasa.netcdf import read_goes_xrs_netcdf

log = get_logger(__name__)

# Layout of the NCEI archive. Override ``base_url`` if NOAA moves it; {sat} = 16, {yyyy}, {mm}.
DEFAULT_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
    "goes{sat}/l2/data/xrsf-l2-avg1m_science/{yyyy}/{mm}/"
)
FILE_RE = r"sci_xrsf-l2-avg1m_g{sat}_d{ymd}_v[\w\-.]+\.nc$"


def day_range(start: str, end: str) -> list[date]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        raise SourceUnavailable(f"end {end} is before start {start}")
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def find_local_file(local_dir: Path, sat: int, d: date) -> Path | None:
    pattern = f"sci_xrsf-l2-avg1m_g{sat}_d{d:%Y%m%d}_v*.nc"
    hits = sorted(local_dir.glob(pattern))
    return hits[-1] if hits else None  # highest version string last


@SOURCES.register("nasa_goes_ncei")
class GoesNceiSource(DataSource):
    @dataclass
    class Params:
        satellite: int = 16
        start: str | None = None  # YYYY-MM-DD, inclusive
        end: str | None = None  # YYYY-MM-DD, inclusive
        local_dir: str = "data/downloads/goes_xrs"
        offline: bool = False  # True: only use files already in local_dir
        base_url: str = DEFAULT_BASE_URL
        use_quality_flags: bool = True  # flagged samples -> NaN
        max_missing_days: int = 10  # tolerate this many unavailable days before failing
        timeout_s: float = 120.0
        retries: int = 4
        workers: int = 4  # parallel downloads (the NCEI server is slow, so this helps a lot)

    default_labeler = "goes_events"
    label_kind = "GOES-calibrated classes from NOAA-style event detection on the 0.1-0.8 nm flux"
    soft_name = "GOES XRS-B (0.1-0.8 nm)"
    hard_name = "GOES XRS-A (0.05-0.4 nm)"

    def input_files(self) -> list[Path]:
        p = self.params
        if not (p.start and p.end):
            return []
        d = Path(p.local_dir)
        return [
            f for day in day_range(p.start, p.end) if (f := find_local_file(d, p.satellite, day))
        ]

    def _obtain(self, d: date, listings: dict | None = None) -> Path | None:
        """Local file for day ``d``, downloading it if needed (listings are cached per month)."""
        p = self.params
        local = Path(p.local_dir)
        hit = find_local_file(local, p.satellite, d)
        if hit or p.offline:
            return hit
        base = p.base_url.format(sat=p.satellite, yyyy=f"{d:%Y}", mm=f"{d:%m}")
        listings = listings if listings is not None else {}
        if base not in listings:  # one directory listing per month, not per day
            try:
                listings[base] = list_directory(
                    base, r"\.nc$", timeout=p.timeout_s, retries=p.retries
                )
            except SourceUnavailable as e:
                listings[base] = e
        names_all = listings[base]
        if isinstance(names_all, SourceUnavailable):
            raise names_all
        rx = re.compile(FILE_RE.format(sat=p.satellite, ymd=f"{d:%Y%m%d}"))
        names = [n for n in names_all if rx.search(n)]
        if not names:
            return None
        return download_file(
            base + names[-1], local / names[-1], timeout=p.timeout_s, retries=p.retries
        )

    def fetch(self) -> RawSeries:
        p = self.params
        if not (p.start and p.end):
            raise SourceUnavailable(
                f"[{self.name}] params.start and params.end (YYYY-MM-DD) required"
            )
        t_parts, s_parts, h_parts, missing = [], [], [], []
        days = day_range(p.start, p.end)
        listings: dict = {}
        errors: dict[date, str] = {}
        done = [0]

        def get(d: date) -> Path | None:
            try:
                return self._obtain(d, listings)
            except SourceUnavailable as e:
                errors[d] = str(e)
                return None
            finally:
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == len(days):
                    log.info("[%s] %d/%d days fetched", self.name, done[0], len(days))

        # Month listings first (sequential, cached), then the files in parallel.
        local = Path(p.local_dir)
        todo = [d for d in days if find_local_file(local, p.satellite, d) is None]
        if todo and not p.offline:
            for d in todo[:1]:
                get(d)  # fail-fast probe: surfaces a systemic network problem early
            done[0] -= 1  # the probe day is fetched again (instantly) below
        with ThreadPoolExecutor(max_workers=max(1, p.workers)) as pool:
            files = list(pool.map(get, days))
        if len(errors) == len(days):
            raise SourceUnavailable(
                f"[{self.name}] every day failed, e.g. {next(iter(errors.values()))}"
            )
        for d, f in zip(days, files, strict=True):
            if d in errors:
                log.warning("[%s] %s: %s", self.name, d, errors[d])
            if f is None:
                missing.append(d.isoformat())
                if len(missing) > p.max_missing_days:
                    raise SourceUnavailable(
                        f"[{self.name}] more than {p.max_missing_days} days unavailable "
                        f"(re-run to retry only those; finished downloads are kept): {missing}"
                    )
                continue
            t, soft, hard = read_goes_xrs_netcdf(f, p.use_quality_flags)
            t_parts.append(t)
            s_parts.append(soft)
            h_parts.append(hard if hard is not None else np.full_like(soft, np.nan))
        if not t_parts:
            raise SourceUnavailable(
                f"[{self.name}] no GOES-{p.satellite} data for {p.start}..{p.end}"
            )
        t = np.concatenate(t_parts)
        order = np.argsort(t, kind="stable")
        t = t[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        soft = np.concatenate(s_parts)[order][keep]
        hard = np.concatenate(h_parts)[order][keep]
        t = t[keep]
        if missing:
            log.warning("[%s] %d day(s) unavailable: %s", self.name, len(missing), missing)
        channels = {"sxr": Channel(t, soft)}
        if np.isfinite(hard).any():
            channels["hxr"] = Channel(t, hard)
        meta = {**self.describe(), "satellite": p.satellite, "missing_days": missing}
        return RawSeries(self.name, self.kind, channels, meta)

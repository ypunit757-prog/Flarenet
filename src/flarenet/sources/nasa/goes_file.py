"""Local GOES-like files: the 'bring your own NASA/NOAA data' route.

Accepts CSV, Parquet, NOAA SWPC JSON dumps and GOES-R XRS NetCDF files (one or many, globs allowed).
For CSV/Parquet, point ``time_column`` / ``sxr_column`` / ``hxr_column`` at your columns; this is
the quickest way to use data exported from SunPy/Fido, CDAWeb, a spreadsheet, or another mission.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.core.utils import get_logger, to_unix_seconds
from flarenet.sources.base import DataSource, SourceUnavailable
from flarenet.sources.nasa.goes_swpc import parse_swpc_records
from flarenet.sources.nasa.netcdf import read_goes_xrs_netcdf

log = get_logger(__name__)
MJD_UNIX_EPOCH, JD_UNIX_EPOCH = 40587.0, 2440587.5


def times_to_unix(col: pd.Series, fmt: str) -> np.ndarray:
    """Convert a time column to unix seconds. fmt: auto | unix | unix_ms | mjd | jd | iso."""
    if fmt == "iso":
        return to_unix_seconds(col)
    num = pd.to_numeric(col, errors="coerce")
    if fmt == "auto":
        if num.notna().mean() < 0.9:
            return to_unix_seconds(col)
        med = float(num.median())
        fmt = "unix_ms" if med > 1e11 else "unix" if med > 1e8 else "mjd" if med > 1e4 else "jd"
        if fmt == "jd" and med < 2e6:
            raise SourceUnavailable(f"Cannot guess the numeric time format (median {med})")
    v = num.to_numpy(dtype=float)
    if fmt == "unix":
        return v
    if fmt == "unix_ms":
        return v / 1000.0
    if fmt == "mjd":
        return (v - MJD_UNIX_EPOCH) * 86400.0
    if fmt == "jd":
        return (v - JD_UNIX_EPOCH) * 86400.0
    raise SourceUnavailable(f"Unknown time_format {fmt!r}")


@SOURCES.register("nasa_goes_file")
class GoesFileSource(DataSource):
    @dataclass
    class Params:
        paths: list[str] = field(default_factory=list)  # files or globs
        format: str = "auto"  # auto | csv | parquet | swpc_json | netcdf
        time_column: str = "time"
        time_format: str = "auto"  # auto | unix | unix_ms | mjd | jd | iso
        sxr_column: str = "xrsb"  # 0.1-0.8 nm flux, W/m^2
        hxr_column: str | None = "xrsa"  # 0.05-0.4 nm flux, W/m^2; null if you have none
        sxr_scale: float = 1.0  # multiply fluxes (e.g. 1/0.7 for legacy GOES scaling)
        hxr_scale: float = 1.0
        use_quality_flags: bool = True
        csv_sep: str = ","

    default_labeler = "goes_events"
    label_kind = "GOES-calibrated classes from NOAA-style event detection on the soft-channel flux"
    soft_name = "soft X-ray (0.1-0.8 nm)"
    hard_name = "short X-ray (0.05-0.4 nm)"

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for pat in self.params.paths:
            hits = sorted(glob.glob(str(Path(pat).expanduser())))
            out += [Path(h) for h in hits]
        return out

    def input_files(self) -> list[Path]:
        return self._files()

    @staticmethod
    def _guess(path: Path) -> str:
        suf = path.suffix.lower()
        return {
            ".csv": "csv",
            ".txt": "csv",
            ".parquet": "parquet",
            ".json": "swpc_json",
            ".nc": "netcdf",
            ".nc4": "netcdf",
        }.get(suf, "csv")

    def _read_one(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        p = self.params
        fmt = self._guess(path) if p.format == "auto" else p.format
        if fmt == "netcdf":
            return read_goes_xrs_netcdf(path, p.use_quality_flags)
        if fmt == "swpc_json":
            import json

            with open(path) as f:
                ch = parse_swpc_records(json.load(f))
            hard = ch["hxr"] if "hxr" in ch else None
            if hard is not None and not np.array_equal(hard.t_unix, ch["sxr"].t_unix):
                # align the short channel onto the soft channel's timestamps
                hv = pd.Series(hard.values, index=hard.t_unix).reindex(ch["sxr"].t_unix).to_numpy()
            else:
                hv = hard.values if hard is not None else None
            return ch["sxr"].t_unix, ch["sxr"].values, hv
        if fmt in ("csv", "parquet"):
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, sep=p.csv_sep)
            for col in (p.time_column, p.sxr_column):
                if col not in df.columns:
                    raise SourceUnavailable(
                        f"{path}: column {col!r} not found in {list(df.columns)}"
                    )
            t = times_to_unix(df[p.time_column], p.time_format)
            soft = pd.to_numeric(df[p.sxr_column], errors="coerce").to_numpy(float) * p.sxr_scale
            soft = np.where(np.isfinite(soft) & (soft > 0), soft, np.nan)
            hard = None
            if p.hxr_column:
                if p.hxr_column not in df.columns:
                    raise SourceUnavailable(f"{path}: column {p.hxr_column!r} not found")
                hard = (
                    pd.to_numeric(df[p.hxr_column], errors="coerce").to_numpy(float) * p.hxr_scale
                )
                hard = np.where(np.isfinite(hard) & (hard > 0), hard, np.nan)
            return t, soft, hard
        raise SourceUnavailable(f"Unknown format {fmt!r}")

    def fetch(self) -> RawSeries:
        files = self._files()
        if not files:
            raise SourceUnavailable(
                f"[{self.name}] no files match params.paths={self.params.paths}"
            )
        ts, ss, hs = [], [], []
        for f in files:
            t, s, h = self._read_one(f)
            m = np.isfinite(t)
            ts.append(t[m])
            ss.append(s[m])
            hs.append((h if h is not None else np.full_like(s, np.nan))[m])
        t = np.concatenate(ts)
        order = np.argsort(t, kind="stable")
        t, soft, hard = t[order], np.concatenate(ss)[order], np.concatenate(hs)[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, soft, hard = t[keep], soft[keep], hard[keep]
        channels = {"sxr": Channel(t, soft)}
        if np.isfinite(hard).any():
            channels["hxr"] = Channel(t, hard)
        log.info("[%s] read %d file(s), %d samples", self.name, len(files), len(t))
        return RawSeries(self.name, self.kind, channels, {**self.describe(), "n_files": len(files)})

"""Aditya-L1 soft X-ray (SoLEXS) and hard X-ray (HEL1OS) level-1 lightcurves from ISRO PRADAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.core.utils import get_logger
from flarenet.sources.base import DataSource, SourceUnavailable

log = get_logger(__name__)
MJD_UNIX_EPOCH = 40587.0  # MJD of 1970-01-01


@SOURCES.register("aditya_l1")
class AdityaL1Source(DataSource):
    """SoLEXS ``*.lc`` (HDU 1: TIME unix s, COUNTS) + HEL1OS CZT1 (HDU 5: MJD, CTR) lightcurves."""

    @dataclass
    class Params:
        solexs_path: str | None = None
        hel1os_dirs: list[str] = field(default_factory=list)
        hel1os_file: str = "czt/lightcurve_czt1.fits"
        hel1os_hdu: int = 5

    default_labeler = "relative_percentile"
    default_scaling = "background_normalised"
    label_kind = (
        "relative proxy (multiples of the 10th-percentile rate), not calibrated GOES classes"
    )
    units = "counts/s"
    soft_name = "SoLEXS"
    hard_name = "HEL1OS"

    def input_files(self) -> list[Path]:
        p = self.params
        files = [Path(p.solexs_path)] if p.solexs_path else []
        files += [Path(d) / p.hel1os_file for d in p.hel1os_dirs]
        return files

    def fetch(self) -> RawSeries:
        p = self.params
        if not p.solexs_path:
            raise SourceUnavailable(f"[{self.name}] params.solexs_path is required for aditya_l1")
        if not p.hel1os_dirs:
            raise SourceUnavailable(f"[{self.name}] params.hel1os_dirs is required for aditya_l1")
        for f in self.input_files():
            if not f.exists():
                raise SourceUnavailable(f"[{self.name}] file not found: {f}")

        with fits.open(p.solexs_path) as hdul:
            data = hdul[1].data
            t_s = np.array(data["TIME"], dtype=float)  # already unix seconds
            c_s = np.array(data["COUNTS"], dtype=float)

        t_parts, c_parts = [], []
        for orbit_dir in p.hel1os_dirs:
            with fits.open(Path(orbit_dir) / p.hel1os_file) as hdul:
                data = hdul[p.hel1os_hdu].data
                t_parts.append((np.array(data["MJD"], dtype=float) - MJD_UNIX_EPOCH) * 86400.0)
                c_parts.append(np.array(data["CTR"], dtype=float))
        t_h = np.concatenate(t_parts)
        c_h = np.concatenate(c_parts)
        order = np.argsort(t_h)
        log.info("[%s] SoLEXS %d samples, HEL1OS %d samples", self.name, len(t_s), len(t_h))
        return RawSeries(
            self.name,
            self.kind,
            {"sxr": Channel(t_s, c_s), "hxr": Channel(t_h[order], c_h[order])},
            self.describe(),
        )

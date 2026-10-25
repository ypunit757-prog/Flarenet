"""NASA CCMC DONKI solar-flare catalog (``https://api.nasa.gov/DONKI/FLR``).

Free API key from https://api.nasa.gov (``DEMO_KEY`` works with low rate limits). Pass the key as
``api_key`` or set ``api_key: env`` to read the ``NASA_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flarenet.core.registry import CATALOGS
from flarenet.core.utils import get_logger, to_unix_seconds
from flarenet.sources.base import SourceUnavailable
from flarenet.sources.catalogs import CATALOG_COLUMNS, EventCatalog
from flarenet.sources.nasa.http import get_json

log = get_logger(__name__)
CHUNK_DAYS = 30


def parse_donki_flares(records: list[dict]) -> pd.DataFrame:
    """DONKI FLR JSON -> catalog frame. Records without a begin time or class are dropped."""
    rows = []
    for r in records or []:
        rows.append(
            {
                "flr_id": str(r.get("flrID", "")),
                "begin_unix": to_unix_seconds([r.get("beginTime")])[0],
                "peak_unix": to_unix_seconds([r.get("peakTime")])[0],
                "end_unix": to_unix_seconds([r.get("endTime")])[0],
                "class_type": str(r.get("classType") or ""),
            }
        )
    df = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    for col in ("begin_unix", "peak_unix", "end_unix"):  # empty responses give object dtype
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df = df[np.isfinite(df.begin_unix) & (df.class_type != "")]
    return df.drop_duplicates("flr_id").sort_values("begin_unix").reset_index(drop=True)


@CATALOGS.register("donki")
class DonkiCatalog(EventCatalog):
    @dataclass
    class Params:
        api_key: str = "DEMO_KEY"  # or "env" -> $NASA_API_KEY
        base_url: str = "https://api.nasa.gov/DONKI/FLR"
        json_path: str | None = None  # read a saved response instead of calling the API
        timeout_s: float = 60.0
        retries: int = 3

    def fetch(self, t_start_unix: float, t_end_unix: float) -> pd.DataFrame:
        p = self.params
        if p.json_path:
            with open(p.json_path) as f:
                records = json.load(f)
        else:
            key = os.environ.get("NASA_API_KEY", "DEMO_KEY") if p.api_key == "env" else p.api_key
            records, t = [], t_start_unix
            while t <= t_end_unix:
                t_next = min(t + CHUNK_DAYS * 86400, t_end_unix)
                params = {
                    "startDate": pd.Timestamp(t, unit="s").strftime("%Y-%m-%d"),
                    "endDate": pd.Timestamp(t_next, unit="s").strftime("%Y-%m-%d"),
                    "api_key": key,
                }
                data = get_json(p.base_url, params=params, timeout=p.timeout_s, retries=p.retries)
                if not isinstance(data, list):
                    raise SourceUnavailable(f"Unexpected DONKI response: {str(data)[:200]}")
                records += data
                t = t_next + 86400
        df = parse_donki_flares(records)
        m = (df.peak_unix.fillna(df.begin_unix) >= t_start_unix) & (df.begin_unix <= t_end_unix)
        out = df[m].reset_index(drop=True)
        log.info("DONKI: %d flares in the requested window", len(out))
        return out

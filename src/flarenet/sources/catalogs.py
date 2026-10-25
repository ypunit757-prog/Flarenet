"""Event catalogs: official flare lists used as label truth instead of detecting flares in the flux.

A catalog returns a DataFrame with columns
``begin_unix, peak_unix, end_unix (NaN allowed), class_type ('M5.2'), flr_id``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from flarenet.core.config import build_dataclass
from flarenet.core.registry import CATALOGS
from flarenet.core.utils import fingerprint, to_unix_seconds
from flarenet.sources.base import NoParams, SourceUnavailable

CATALOG_COLUMNS = ["flr_id", "begin_unix", "peak_unix", "end_unix", "class_type"]


class EventCatalog(ABC):
    Params: ClassVar[type] = NoParams

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = build_dataclass(self.Params, params, "catalog_params.")

    @abstractmethod
    def fetch(self, t_start_unix: float, t_end_unix: float) -> pd.DataFrame: ...

    def fingerprint(self) -> str:
        from dataclasses import asdict

        return fingerprint(type(self).__name__, asdict(self.params))


def make_catalog(name: str, params: dict[str, Any] | None = None) -> EventCatalog:
    import flarenet.sources  # noqa: F401  (registers the built-in catalogs)

    return CATALOGS.get(name)(params)


@CATALOGS.register("csv")
class CsvCatalog(EventCatalog):
    """Any CSV with begin/peak/end times and a GOES class string such as 'M5.2'."""

    @dataclass
    class Params:
        path: str | None = None
        begin_column: str = "begin_time"
        peak_column: str = "peak_time"
        end_column: str | None = "end_time"
        class_column: str = "class_type"
        id_column: str | None = None

    def fetch(self, t_start_unix: float, t_end_unix: float) -> pd.DataFrame:
        p = self.params
        if not p.path:
            raise SourceUnavailable("csv catalog needs catalog_params.path")
        df = pd.read_csv(p.path)
        for col in (p.begin_column, p.peak_column, p.class_column):
            if col not in df.columns:
                raise SourceUnavailable(f"{p.path}: column {col!r} not found in {list(df.columns)}")
        out = pd.DataFrame(
            {
                "flr_id": df[p.id_column].astype(str)
                if p.id_column
                else np.arange(len(df)).astype(str),
                "begin_unix": to_unix_seconds(df[p.begin_column]),
                "peak_unix": to_unix_seconds(df[p.peak_column]),
                "end_unix": to_unix_seconds(df[p.end_column])
                if p.end_column in df.columns
                else np.nan,
                "class_type": df[p.class_column].astype(str),
            }
        )
        m = (out.peak_unix >= t_start_unix) & (out.peak_unix <= t_end_unix)
        return out[m].reset_index(drop=True)[CATALOG_COLUMNS]

"""Labeler interface: turn a 1-minute lightcurve into per-minute class / event id / regime.

To add a labeler:

    @LABELERS.register("my_labeler")
    class MyLabeler(Labeler):
        @dataclass
        class Params:
            threshold: float = 3.0
        def label(self, inp: LabelInput) -> dict[str, np.ndarray]: ...

and select it per source with ``labeler: my_labeler`` / ``labeler_params: {...}`` in the config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from flarenet.core.config import build_dataclass
from flarenet.core.utils import fingerprint
from flarenet.sources.base import NoParams


@dataclass
class LabelInput:
    df: pd.DataFrame  # time_min, sxr, hxr, sxr_raw, hxr_raw on a 1-minute grid
    t0_unix: float
    aligned: dict[str, np.ndarray] = field(default_factory=dict)


class Labeler(ABC):
    Params: ClassVar[type] = NoParams

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = build_dataclass(self.Params, params, "labeler_params.")

    @abstractmethod
    def label(self, inp: LabelInput) -> dict[str, np.ndarray]:
        """Return arrays ``flare_class`` (0..4), ``flare_id`` (-1 = none) and ``active_regime``."""

    def fingerprint(self) -> str:
        return fingerprint(type(self).__name__, asdict(self.params), "v1")


def events_to_arrays(n: int, events: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """events = [(first_minute, last_minute_inclusive, class_idx)] in time order -> label arrays.
    Overlaps keep the higher class and, there, the stronger event's id."""
    cls = np.zeros(n, dtype=int)
    fid = -np.ones(n, dtype=int)
    for k, (i0, i1, c) in enumerate(events):
        i0, i1 = max(0, i0), min(n - 1, i1)
        if i1 < i0 or c <= 0:
            continue
        sl = slice(i0, i1 + 1)
        stronger = c >= cls[sl]
        cls[sl] = np.where(stronger, c, cls[sl])
        fid[sl] = np.where(stronger, k, fid[sl])
    # re-number ids densely so they are 0..n_events-1 in time order
    ids = np.unique(fid[fid >= 0])
    if len(ids):
        fid = np.where(fid >= 0, np.searchsorted(ids, fid), -1)
    return {"flare_class": cls, "flare_id": fid, "active_regime": (cls >= 2).astype(int)}

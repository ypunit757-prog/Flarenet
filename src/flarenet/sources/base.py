"""Data-source interface. One class per instrument/provider; each returns a ``RawSeries``.

To add a new instrument:

    @SOURCES.register("my_instrument")
    class MySource(DataSource):
        @dataclass
        class Params:
            path: str | None = None
        default_labeler = "goes_events"       # how flares get labelled
        default_scaling = "native"
        def fetch(self) -> RawSeries: ...

then reference ``kind: my_instrument`` in a config. Nothing else in the project changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from flarenet.core.config import SourceSpec, build_dataclass
from flarenet.core.registry import SOURCES
from flarenet.core.schema import RawSeries
from flarenet.core.utils import fingerprint


class SourceUnavailable(RuntimeError):
    """The data could not be obtained (network blocked, files missing, bad credentials, ...)."""


@dataclass
class NoParams:
    pass


class DataSource(ABC):
    kind: ClassVar[str] = ""
    Params: ClassVar[type] = NoParams
    # What the downstream pipelines should do with this source unless the config overrides it.
    default_labeler: ClassVar[str] = "goes_events"
    default_scaling: ClassVar[str] = "native"
    # Human-readable meaning of the labels the default labeler produces (goes into reports).
    label_kind: ClassVar[str] = "unspecified"
    units: ClassVar[str] = "W/m2"
    uses_seed: ClassVar[bool] = False
    soft_name: ClassVar[str] = "soft X-ray"
    hard_name: ClassVar[str] = "hard X-ray"

    def __init__(self, name: str, params: dict[str, Any] | None = None, seed: int = 42):
        self.name = name
        self.seed = seed
        self.params = build_dataclass(self.Params, params, f"sources[{name}].params.")

    # ----- to implement ------------------------------------------------------------------
    @abstractmethod
    def fetch(self) -> RawSeries:
        """Return raw samples at native cadence. Raise SourceUnavailable if it cannot."""

    def input_files(self) -> list[Path]:
        """Local files this source reads; their size/mtime go into the cache fingerprint."""
        return []

    # ----- provided ----------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "units": self.units,
            "label_kind": self.label_kind,
            "soft_name": self.soft_name,
            "hard_name": self.hard_name,
        }

    def fingerprint(self) -> str:
        files = []
        for p in self.input_files():
            st = p.stat() if p.exists() else None
            files.append([str(p), st.st_size if st else -1, st.st_mtime_ns if st else -1])
        return fingerprint(
            self.kind, asdict(self.params), self.seed if self.uses_seed else None, files
        )


def make_source(spec: SourceSpec, seed: int = 42) -> DataSource:
    # Importing the package registers every built-in source.
    import flarenet.sources  # noqa: F401

    return SOURCES.get(spec.kind)(spec.name, spec.params, seed)

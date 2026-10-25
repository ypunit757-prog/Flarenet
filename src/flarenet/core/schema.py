"""Data contracts passed between pipelines.

    DataSource --fetch--> RawSeries --preprocess--> LightcurveBundle --features--> WindowSet --> model

Every instrument (Aditya-L1, GOES, a synthetic generator, ...) is reduced to the same two
objects, so nothing downstream knows which spacecraft produced the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flarenet.core.constants import LIGHTCURVE_COLUMNS


class SchemaError(ValueError):
    """Raised when data does not satisfy the lightcurve / window contract."""


@dataclass
class Channel:
    """One measured quantity on its own (possibly irregular) time axis."""

    t_unix: np.ndarray  # seconds since 1970-01-01 UTC, float64, ascending
    values: np.ndarray  # same length; NaN = missing

    def __post_init__(self) -> None:
        self.t_unix = np.asarray(self.t_unix, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.t_unix.shape != self.values.shape or self.t_unix.ndim != 1:
            raise SchemaError("Channel t_unix and values must be 1-D arrays of equal length")

    def __len__(self) -> int:
        return len(self.t_unix)


@dataclass
class RawSeries:
    """Raw instrument output at native cadence, before resampling or labelling.

    ``channels`` must contain ``"sxr"`` (soft X-ray, the channel flares are labelled on) and may
    contain ``"hxr"`` (hard X-ray or the nearest equivalent). ``aligned`` holds optional
    per-sample arrays aligned to the ``sxr`` samples (e.g. ground-truth labels of synthetic data).
    """

    source: str
    kind: str
    channels: dict[str, Channel]
    meta: dict[str, Any] = field(default_factory=dict)
    aligned: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "sxr" not in self.channels:
            raise SchemaError(f"Source {self.source!r}: RawSeries needs an 'sxr' channel")
        n = len(self.channels["sxr"])
        for k, a in self.aligned.items():
            if len(a) != n:
                raise SchemaError(f"aligned[{k!r}] has {len(a)} samples, sxr has {n}")

    @property
    def span_days(self) -> float:
        t = self.channels["sxr"].t_unix
        return float((t[-1] - t[0]) / 86400.0) if len(t) > 1 else 0.0


@dataclass
class LightcurveBundle:
    """A labelled, 1-minute lightcurve plus its event table. ``time_min`` is minutes since t0."""

    name: str
    kind: str
    df: pd.DataFrame
    t0_unix: float
    meta: dict[str, Any] = field(default_factory=dict)
    flare_db: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_minutes(self) -> int:
        return len(self.df)

    def unix_of(self, minutes) -> np.ndarray:
        return self.t0_unix + np.asarray(minutes, dtype=float) * 60.0


def validate_lightcurve(df: pd.DataFrame, name: str = "") -> None:
    """Check the processed-lightcurve contract; raise SchemaError with a precise message."""
    tag = f"[{name}] " if name else ""
    missing = [c for c in LIGHTCURVE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{tag}lightcurve is missing columns {missing}")
    if len(df) < 2:
        raise SchemaError(f"{tag}lightcurve has {len(df)} rows")
    t = df["time_min"].to_numpy()
    if not np.allclose(np.diff(t), 1.0):
        raise SchemaError(f"{tag}time_min must be a regular 1-minute grid")
    cls = df["flare_class"].to_numpy()
    if cls.min() < 0 or cls.max() > 4:
        raise SchemaError(f"{tag}flare_class must be in 0..4, got [{cls.min()}, {cls.max()}]")
    fid = df["flare_id"].to_numpy()
    if fid.min() < -1:
        raise SchemaError(f"{tag}flare_id must be -1 (no flare) or >= 0")
    if np.any((fid == -1) & (cls > 0)):
        raise SchemaError(f"{tag}flare_class > 0 on rows with flare_id == -1")
    for col in ("sxr", "hxr"):
        v = df[col].to_numpy()
        finite = v[np.isfinite(v)]
        if len(finite) == 0:
            raise SchemaError(f"{tag}column {col!r} has no finite values")
        if np.any(finite <= 0):
            raise SchemaError(f"{tag}column {col!r} must be strictly positive (log features)")


@dataclass
class WindowSet:
    """Model-ready windows. ``t_end`` is the minute index of each window's last sample."""

    X: np.ndarray  # (n, channels, window_min) float32, raw (un-normalised) features
    y_now: np.ndarray  # (n,) int64, nowcast class at the window's last minute
    y_fc: np.ndarray  # (n,) float32, 1 if an M/X flare starts within the horizon
    t_end: np.ndarray  # (n,) float64
    missing_frac: np.ndarray  # (n,) float32, share of missing input samples in the window
    channels: list[str]
    source_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    source_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.X)
        if len(self.source_idx) != n:
            self.source_idx = np.zeros(n, dtype=np.int16)
        if not self.source_names:
            self.source_names = ["_"]

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask_or_idx) -> WindowSet:
        return WindowSet(
            self.X[mask_or_idx],
            self.y_now[mask_or_idx],
            self.y_fc[mask_or_idx],
            self.t_end[mask_or_idx],
            self.missing_frac[mask_or_idx],
            list(self.channels),
            self.source_idx[mask_or_idx],
            list(self.source_names),
        )

    def tagged(self, source_name: str) -> WindowSet:
        """Label every window as coming from ``source_name`` (before concatenation)."""
        out = self.subset(slice(None))
        out.source_idx = np.zeros(len(self), dtype=np.int16)
        out.source_names = [source_name]
        return out

    @staticmethod
    def concat(sets: list[WindowSet]) -> WindowSet:
        if not sets:
            raise SchemaError("Cannot concatenate zero WindowSets")
        chans = sets[0].channels
        names: list[str] = []
        idx_parts = []
        for ws in sets:
            if ws.channels != chans:
                raise SchemaError("Cannot concatenate WindowSets with different channels")
            remap = np.zeros(len(ws.source_names), dtype=np.int16)
            for i, nm in enumerate(ws.source_names):
                if nm not in names:
                    names.append(nm)
                remap[i] = names.index(nm)
            idx_parts.append(remap[ws.source_idx])
        return WindowSet(
            np.concatenate([w.X for w in sets]),
            np.concatenate([w.y_now for w in sets]),
            np.concatenate([w.y_fc for w in sets]),
            np.concatenate([w.t_end for w in sets]),
            np.concatenate([w.missing_frac for w in sets]),
            list(chans),
            np.concatenate(idx_parts),
            names,
        )

    # ----- persistence -------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y_now=self.y_now,
            y_fc=self.y_fc,
            t_end=self.t_end,
            missing_frac=self.missing_frac,
            source_idx=self.source_idx,
            channels=np.array(self.channels),
            source_names=np.array(self.source_names),
        )

    @staticmethod
    def load(path: str | Path) -> WindowSet:
        z = np.load(path, allow_pickle=False)
        return WindowSet(
            z["X"],
            z["y_now"],
            z["y_fc"],
            z["t_end"],
            z["missing_frac"],
            [str(c) for c in z["channels"]],
            z["source_idx"],
            [str(s) for s in z["source_names"]],
        )

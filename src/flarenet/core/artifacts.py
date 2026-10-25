"""On-disk artifact store with fingerprint-based caching.

Layout (``paths.data_dir``)::

    raw/<source>/            sxr.parquet, hxr.parquet, aligned.parquet, meta.json
    processed/<source>/      lightcurve.parquet, flare_db.csv, meta.json
    features/<source>/       windows.npz, meta.json

Each directory's ``meta.json`` stores the fingerprint of everything that produced it; a stage
re-runs only when the fingerprint changed or ``--force`` is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.schema import Channel, LightcurveBundle, RawSeries, WindowSet
from flarenet.core.utils import load_json, save_json

META = "meta.json"


class DataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ----- paths -------------------------------------------------------------------------
    def raw_dir(self, name: str) -> Path:
        return self.root / "raw" / name

    def processed_dir(self, name: str) -> Path:
        return self.root / "processed" / name

    def features_dir(self, name: str) -> Path:
        return self.root / "features" / name

    # ----- freshness ---------------------------------------------------------------------
    @staticmethod
    def is_fresh(directory: Path, fp: str) -> bool:
        meta = directory / META
        if not meta.exists():
            return False
        try:
            return load_json(meta).get("fingerprint") == fp
        except Exception:  # corrupt / half-written meta -> recompute
            return False

    @staticmethod
    def write_meta(directory: Path, fp: str, meta: dict) -> None:
        save_json({**meta, "fingerprint": fp}, directory / META)  # written last = commit marker

    @staticmethod
    def read_meta(directory: Path) -> dict:
        return load_json(directory / META)

    # ----- raw series --------------------------------------------------------------------
    def save_raw(self, series: RawSeries, fp: str) -> Path:
        d = self.raw_dir(series.source)
        d.mkdir(parents=True, exist_ok=True)
        (d / META).unlink(missing_ok=True)  # invalidate while rewriting
        for ch_name, ch in series.channels.items():
            pd.DataFrame({"t_unix": ch.t_unix, "value": ch.values}).to_parquet(
                d / f"{ch_name}.parquet", index=False
            )
        if series.aligned:
            pd.DataFrame(series.aligned).to_parquet(d / "aligned.parquet", index=False)
        self.write_meta(
            d,
            fp,
            {
                "kind": series.kind,
                "channels": sorted(series.channels),
                "has_aligned": bool(series.aligned),
                "series_meta": series.meta,
            },
        )
        return d

    def load_raw(self, name: str) -> RawSeries:
        d = self.raw_dir(name)
        meta = self.read_meta(d)
        channels = {}
        for ch_name in meta["channels"]:
            f = pd.read_parquet(d / f"{ch_name}.parquet")
            channels[ch_name] = Channel(f["t_unix"].to_numpy(), f["value"].to_numpy())
        aligned = {}
        if meta.get("has_aligned"):
            aligned = {c: v.to_numpy() for c, v in pd.read_parquet(d / "aligned.parquet").items()}
        return RawSeries(name, meta["kind"], channels, meta["series_meta"], aligned)

    # ----- processed lightcurve ----------------------------------------------------------
    def save_bundle(self, bundle: LightcurveBundle, fp: str) -> Path:
        d = self.processed_dir(bundle.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / META).unlink(missing_ok=True)
        bundle.df.to_parquet(d / "lightcurve.parquet", index=False)
        bundle.flare_db.to_csv(d / "flare_db.csv", index=False)
        self.write_meta(
            d, fp, {"kind": bundle.kind, "t0_unix": bundle.t0_unix, "bundle_meta": bundle.meta}
        )
        return d

    def load_bundle(self, name: str) -> LightcurveBundle:
        d = self.processed_dir(name)
        meta = self.read_meta(d)
        try:
            fdb = pd.read_csv(d / "flare_db.csv")
        except pd.errors.EmptyDataError:
            fdb = pd.DataFrame()
        return LightcurveBundle(
            name,
            meta["kind"],
            pd.read_parquet(d / "lightcurve.parquet"),
            float(meta["t0_unix"]),
            meta["bundle_meta"],
            fdb,
        )

    # ----- windows -----------------------------------------------------------------------
    def save_windows(self, name: str, ws: WindowSet, fp: str, extra: dict | None = None) -> Path:
        d = self.features_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / META).unlink(missing_ok=True)
        ws.save(d / "windows.npz")
        self.write_meta(
            d,
            fp,
            {
                "n_windows": len(ws),
                "forecast_positive_frac": float(np.mean(ws.y_fc)) if len(ws) else None,
                "channels": ws.channels,
                **(extra or {}),
            },
        )
        return d

    def load_windows(self, name: str) -> WindowSet:
        return WindowSet.load(self.features_dir(name) / "windows.npz")

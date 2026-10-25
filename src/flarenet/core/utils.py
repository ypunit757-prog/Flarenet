"""Small utilities: seeding, logging, device selection, JSON helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once with a compact format."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def resolve_device(name: str = "auto") -> torch.device:
    """Map 'auto' | 'cpu' | 'cuda' | 'mps' to a torch device, falling back to CPU."""
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        get_logger(__name__).warning("CUDA requested but unavailable, using CPU")
        return torch.device("cpu")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            get_logger(__name__).warning("MPS requested but unavailable, using CPU")
            return torch.device("cpu")
    return torch.device(name)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_json(obj, path: str | Path) -> None:
    """Write JSON, converting numpy scalars/arrays and NaN to valid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_nan_to_none(obj), f, indent=2, default=_json_default)


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def fingerprint(*parts) -> str:
    """Stable 16-hex-digit hash of JSON-serialisable parts (used for cache invalidation)."""
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def to_unix_seconds(values) -> np.ndarray:
    """Convert timestamps to float unix seconds (UTC). Accepts ISO strings (``2024-05-10T06:27Z``),
    datetimes, pandas Series/Index or a list; None / unparseable entries become NaN."""
    ts = pd.to_datetime(
        pd.Series(list(values) if not isinstance(values, pd.Series) else values),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    out = np.full(len(ts), np.nan)
    ok = ts.notna().to_numpy()
    if ok.any():
        out[ok] = ts[ok].astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9
    return out

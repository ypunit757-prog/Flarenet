"""Settings read from environment variables at the moment they are needed (so tests and operators can change them)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def data_file(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name) or ROOT / "data" / default)
    return path if path.is_absolute() else ROOT / path


def database_path() -> str | None:
    """File (or ``:memory:``) of the SQLite store, from ``DATABASE_URL`` (``sqlite:///data/flarenet.db`` by default).
    ``DATABASE_URL=off`` disables the store. Other databases are not supported by this service; see docs/DEPLOY.md."""
    url = (os.environ.get("DATABASE_URL") or "sqlite:///data/flarenet.db").strip()
    if url.lower() in ("off", "none", ""):
        return None
    if not url.startswith("sqlite:///"):
        raise ValueError(
            "DATABASE_URL must be sqlite:///<file> (or off). Only SQLite is built in; "
            "see docs/DEPLOY.md for what moving to PostgreSQL involves."
        )
    rest = url[len("sqlite:///") :]
    if rest == ":memory:":
        return rest
    p = Path(rest)
    return str(p if p.is_absolute() else ROOT / p)


def admin_key() -> str | None:
    """The admin key, or None when the admin endpoints are disabled (unset, or shorter than 16 characters)."""
    key = os.environ.get("ADMIN_API_KEY", "").strip()
    return key if len(key) >= 16 else None


def background_enabled() -> bool:
    flag = os.environ.get("FLARENET_BACKGROUND", "").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    return bool(
        os.environ.get("FORECAST_URL")
        or os.environ.get("ALERTS_URL")
        or flag in ("1", "true", "on", "yes")
    )

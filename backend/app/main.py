"""FlareNet web API (runs on Render).

A deliberately light service: it serves the forecast payload that `scripts/export_web.py` writes, it does not
run the model itself. That keeps it small enough for a free Render instance (no PyTorch). A scheduled job
(see .github/workflows/refresh-forecast.yml) runs the model on live NOAA data and publishes the JSON; this
API picks it up. Where the payload comes from, in order:

  1. FORECAST_URL   a URL to the JSON (e.g. the raw GitHub URL of the `data` branch). Cached for FORECAST_CACHE_S
                    seconds. If the URL is down, the last good copy is served instead of an error.
  2. FORECAST_FILE  a local file (default: data/forecast.json, which ships with a simulated demo).

Endpoints:  GET /api/health   GET /api/forecast   GET /
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("source", "model", "period", "series", "forecast", "flares")

app = FastAPI(title="FlareNet API", version="1.0.0", docs_url="/docs", redoc_url=None)

# FRONTEND_ORIGIN: comma-separated list of allowed sites, e.g. "https://flarenet.vercel.app".
# FRONTEND_ORIGIN_REGEX: optional, e.g. r"https://flarenet-.*\.vercel\.app" to allow Vercel preview URLs.
_origins = [
    o.strip()
    for o in os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=os.environ.get("FRONTEND_ORIGIN_REGEX") or None,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_cache: dict = {"body": None, "fetched": 0.0, "origin": None}


def _check(body: bytes) -> dict:
    d = json.loads(body)
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"payload is missing {missing}")
    return d


def _from_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flarenet-api"})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (URL is operator-configured)
        return r.read()


def load_payload() -> tuple[bytes, str]:
    """Return (json bytes, origin) where origin is "url", "url-stale" or "file"."""
    url = os.environ.get("FORECAST_URL", "").strip()
    ttl = float(os.environ.get("FORECAST_CACHE_S", "120"))
    if url:
        with _lock:
            fresh = _cache["body"] is not None and time.time() - _cache["fetched"] < ttl
            if fresh:
                return _cache["body"], _cache["origin"]
            try:
                body = _from_url(url)
                _check(body)
                _cache.update(body=body, fetched=time.time(), origin="url")
                return body, "url"
            except (
                Exception
            ):  # network error, bad JSON, missing keys: keep serving the last good copy
                if _cache["body"] is not None:
                    return _cache["body"], "url-stale"
    path = Path(os.environ.get("FORECAST_FILE") or ROOT / "data" / "forecast.json")
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise HTTPException(503, "No forecast available yet")
    body = path.read_bytes()
    _check(body)
    return body, "file"


@app.get("/")
def index():
    return {
        "service": "flarenet-api",
        "docs": "/docs",
        "forecast": "/api/forecast",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    body, origin = load_payload()
    d = json.loads(body)
    try:
        end = datetime.strptime(d["period"]["end"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        age_h = round((datetime.now(timezone.utc) - end).total_seconds() / 3600, 2)
    except Exception:
        age_h = None
    return {
        "status": "ok",
        "origin": origin,
        "demo": bool(d.get("demo")),
        "data_age_hours": age_h,
        "generated": d.get("generated"),
    }


@app.get("/api/forecast")
def forecast():
    body, origin = load_payload()
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=60", "X-Forecast-Origin": origin},
    )

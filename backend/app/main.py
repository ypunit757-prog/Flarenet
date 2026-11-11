"""FlareNet web API (runs on Render).

A deliberately light service: it serves what the scheduled GitHub job publishes, it does not run the model itself. That keeps it
small enough for a free Render instance (no PyTorch). The job (.github/workflows/refresh-forecast.yml) runs the model on live
NOAA data and writes three files to the `data` branch; this API picks them up:

  forecast.json   the forecast payload (scripts/export_web.py)                   FORECAST_URL   or FORECAST_FILE
  alerts.json     predicted-risk and observed-flare events (process_alerts.py)   ALERTS_URL     or ALERTS_FILE
  forward_log.csv the daily forecast record, resolved against outcomes           FORWARD_LOG_URL or FORWARD_LOG_FILE

Each source is cached for a while (FORECAST_CACHE_S, AUX_CACHE_S); if the URL is down the last good copy is served instead of an
error. Alert events are also copied into a small SQLite store (DATABASE_URL) so they can be filtered and paged.

Public endpoints:   GET /api/health  /api/forecast  /api/alerts  /api/forward-log  /docs
Admin endpoints:    GET /api/admin/status   POST /api/admin/refresh      (header X-Admin-Key; off unless ADMIN_API_KEY is set)

Errors are always JSON: {"error": {"code", "message", "request_id"}}. Every request is logged with an id (X-Request-ID).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app import settings
from app.security import require_admin
from app.store import Store

ROOT = settings.ROOT
REQUIRED = ("source", "model", "period", "series", "forecast", "flares")
VERSION = "2.0.0"

log = logging.getLogger("flarenet.api")


def configure_logging() -> None:
    """Log to stderr at LOG_LEVEL (default INFO) unless the host process already configured logging."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


class PayloadError(ValueError):
    """A source answered but its content is unusable (not JSON, wrong shape)."""


# ---------------------------------------------------------------- the forecast payload


def _check(body: bytes) -> dict:
    try:
        d = json.loads(body)
    except ValueError as e:
        raise PayloadError(f"payload is not valid JSON: {e}") from e
    if not isinstance(d, dict):
        raise PayloadError("payload is not a JSON object")
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise PayloadError(f"payload is missing {missing}")
    return d


def _from_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flarenet-api"})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (URL is operator-configured)
        return r.read(20_000_000)


_lock = threading.Lock()
_cache: dict = {"body": None, "fetched": 0.0, "origin": None}


def load_payload() -> tuple[bytes, str]:
    """Return (json bytes, origin) where origin is "url", "url-stale" or "file"."""
    url = os.environ.get("FORECAST_URL", "").strip()
    ttl = settings.env_float("FORECAST_CACHE_S", 120)
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
            ) as e:  # network error, bad JSON, missing keys: keep serving the last good copy
                log.warning("forecast source failed (%s: %s)", type(e).__name__, e)
                if _cache["body"] is not None:
                    return _cache["body"], "url-stale"
    path = settings.data_file("FORECAST_FILE", "forecast.json")
    if not path.exists():
        raise HTTPException(503, "No forecast available yet")
    body = path.read_bytes()
    _check(body)
    return body, "file"


# ---------------------------------------------------------------- alerts and the forward log (optional sources)

_aux: dict[str, dict] = {
    "alerts": {"doc": None, "fetched": 0.0, "origin": "none", "digest": None},
    "forward_log": {"doc": None, "fetched": 0.0, "origin": "none", "digest": None},
}
_aux_lock = threading.Lock()


def _parse_alerts(body: bytes) -> dict:
    try:
        d = json.loads(body)
    except ValueError as e:
        raise PayloadError(f"alerts are not valid JSON: {e}") from e
    events = d.get("events") if isinstance(d, dict) else d
    if not isinstance(events, list) or not all(
        isinstance(e, dict) and "id" in e and "kind" in e and "time" in e for e in events
    ):
        raise PayloadError("alerts must be a list of events with id, kind and time")
    return {"updated": d.get("updated") if isinstance(d, dict) else None, "events": events}


FORWARD_COLUMNS = (
    "for_day",
    "issued_utc",
    "target",
    "p_flarenet",
    "threshold",
    "alert",
    "y",
    "y_cov",
    "resolved_utc",
)


def _num(v: str):
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _parse_forward(body: bytes) -> dict:
    """The daily forward-test record: our forecast and the outcome only (other columns of the file are not published here)."""
    try:
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as e:
        raise PayloadError(f"forward log is not readable CSV: {e}") from e
    if rows and not {"for_day", "p_flarenet"} <= set(rows[0]):
        raise PayloadError("forward log lacks for_day / p_flarenet columns")
    out = []
    for r in rows:
        y = _num(r.get("y"))
        out.append(
            {
                "day": r["for_day"], "issued": r.get("issued_utc") or None, "target": r.get("target") or "M",
                "p": _num(r.get("p_flarenet")), "threshold": _num(r.get("threshold")),
                "alert": None if _num(r.get("alert")) is None else bool(_num(r.get("alert"))),
                "resolved": y is not None, "flare": None if y is None else bool(y),
                "coverage": _num(r.get("y_cov")), "resolved_utc": r.get("resolved_utc") or None,
            }
        )  # fmt: skip
    return {"rows": out}


def _load_aux(
    name: str, url_env: str, file_env: str, default_file: str, parse
) -> tuple[dict | None, str]:
    """``(parsed document, origin)``; origin is url, url-stale, file or none. Never raises for a missing source."""
    ttl = settings.env_float("AUX_CACHE_S", 120)
    c = _aux[name]
    url = os.environ.get(url_env, "").strip()
    with _aux_lock:
        if url:
            if c["doc"] is not None and time.time() - c["fetched"] < ttl:
                return c["doc"], c["origin"]
            try:
                body = _from_url(url)
                c.update(
                    doc=parse(body),
                    fetched=time.time(),
                    origin="url",
                    digest=hashlib.sha256(body).hexdigest(),
                )
                return c["doc"], "url"
            except Exception as e:  # noqa: BLE001
                log.warning("%s source failed (%s: %s)", name, type(e).__name__, e)
                if c["doc"] is not None:
                    return c["doc"], "url-stale"
        path = settings.data_file(file_env, default_file)
        if path.exists():
            try:
                body = path.read_bytes()
                c.update(
                    doc=parse(body),
                    fetched=time.time(),
                    origin="file",
                    digest=hashlib.sha256(body).hexdigest(),
                )
                return c["doc"], "file"
            except Exception as e:  # noqa: BLE001
                log.warning("%s file unreadable (%s: %s)", name, type(e).__name__, e)
    return None, "none"


def load_alerts() -> tuple[dict | None, str]:
    return _load_aux("alerts", "ALERTS_URL", "ALERTS_FILE", "alerts.json", _parse_alerts)


def load_forward_log() -> tuple[dict | None, str]:
    return _load_aux(
        "forward_log", "FORWARD_LOG_URL", "FORWARD_LOG_FILE", "forward_log.csv", _parse_forward
    )


# ---------------------------------------------------------------- the SQLite store

_store_state: dict = {"path": None, "store": None, "error": None}
_store_lock = threading.Lock()


def db_configured() -> bool:
    """Whether a database is asked for (an unusable DATABASE_URL counts as asked for, and shows as unavailable)."""
    try:
        return settings.database_path() is not None
    except ValueError:
        return True


def get_store() -> Store | None:
    """The store for the current DATABASE_URL, opened on first use; None when disabled or unusable (the API then filters in memory)."""
    try:
        path = settings.database_path()
    except ValueError as e:
        _store_state["error"] = str(e)
        return None
    with _store_lock:
        if path != _store_state["path"]:
            if _store_state["store"] is not None:
                _store_state["store"].close()
            _store_state.update(path=path, store=None, error=None)
            if path:
                try:
                    if path != ":memory:":
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                    _store_state["store"] = Store(path)
                except Exception as e:  # noqa: BLE001 - read-only disk etc.: serve without the store
                    log.error("could not open the database %s (%s: %s)", path, type(e).__name__, e)
                    _store_state["error"] = f"{type(e).__name__}: {e}"
        return _store_state["store"]


def _sync_alerts(doc: dict | None) -> int:
    store = get_store()
    if store is None or not doc:
        return 0
    new = store.add_events(doc["events"])
    if new:
        log.info("stored %d new alert event(s)", new)
    return new


# ---------------------------------------------------------------- app, errors, logging


def clear_caches() -> None:
    _cache.update(fetched=0.0)
    for c in _aux.values():
        c.update(fetched=0.0)


def refresh_all() -> dict:
    """Fetch every source now (used by the background loop and the admin endpoint). Never raises."""
    out: dict = {}
    clear_caches()
    for name, fn in (
        ("forecast", lambda: load_payload()[1]),
        ("alerts", lambda: load_alerts()[1]),
        ("forward_log", lambda: load_forward_log()[1]),
    ):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[name] = f"error: {type(e).__name__}"
    doc, _ = load_alerts()
    out["new_alert_events"] = _sync_alerts(doc)
    return out


def _background(stop: threading.Event) -> None:
    interval = max(30.0, settings.env_float("INGEST_INTERVAL_S", 300))
    while not stop.wait(interval):
        try:
            log.info("background refresh: %s", refresh_all())
        except Exception:  # noqa: BLE001
            log.exception("background refresh failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    stop = threading.Event()
    thread = None
    if settings.background_enabled():
        thread = threading.Thread(
            target=_background, args=(stop,), name="flarenet-ingest", daemon=True
        )
        thread.start()
        log.info("background ingestion every %ds", settings.env_float("INGEST_INTERVAL_S", 300))
    if settings.admin_key() is None:
        log.info(
            "admin endpoints are off (set ADMIN_API_KEY, 16 or more characters, to enable them)"
        )
    yield
    stop.set()


app = FastAPI(
    title="FlareNet API", version=VERSION, docs_url="/docs", redoc_url=None, lifespan=lifespan
)

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

_RID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@app.middleware("http")
async def access_log(request: Request, call_next):
    sent = request.headers.get("x-request-id", "")
    rid = (
        sent if _RID.match(sent) else uuid.uuid4().hex[:12]
    )  # an unchecked id could forge log lines
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s failed [%s]", request.method, request.url.path, rid)
        raise
    response.headers["X-Request-ID"] = rid
    log.info(
        "%s %s -> %d in %.0f ms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
        rid,
    )
    return response


def _error(
    request: Request,
    status: int,
    code: str,
    message: str,
    headers: dict | None = None,
    details=None,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status, headers=headers)


_CODES = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed", 503: "unavailable"}


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return _error(
        request,
        exc.status_code,
        _CODES.get(exc.status_code, "error"),
        str(exc.detail),
        getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "problem": e["msg"]} for e in exc.errors()
    ]
    return _error(
        request, 422, "validation_error", "The request has invalid parameters", details=details
    )


@app.exception_handler(PayloadError)
async def payload_error(request: Request, exc: PayloadError):
    log.error("unusable payload: %s", exc)
    return _error(request, 502, "bad_payload", f"The upstream data is unusable: {exc}")


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return _error(
        request, 500, "internal", "Internal error; see the server log with this request id"
    )


# ---------------------------------------------------------------- public endpoints


@app.get("/")
def index():
    return {
        "service": "flarenet-api", "version": VERSION, "docs": "/docs", "forecast": "/api/forecast",
        "alerts": "/api/alerts", "forward_log": "/api/forward-log", "health": "/api/health",
    }  # fmt: skip


@app.get("/api/health")
def health():
    body, origin = load_payload()
    d = json.loads(body)
    try:
        end = datetime.strptime(d["period"]["end"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        age_h = round((datetime.now(timezone.utc) - end).total_seconds() / 3600, 2)
    except Exception:  # noqa: BLE001
        age_h = None
    store = get_store()
    return {
        "status": "ok", "origin": origin, "demo": bool(d.get("demo")), "data_age_hours": age_h, "generated": d.get("generated"),
        "database": "ok" if store else ("unavailable" if db_configured() else "off"),
    }  # fmt: skip


@app.get("/api/forecast")
def forecast():
    body, origin = load_payload()
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=60", "X-Forecast-Origin": origin},
    )


@app.get("/api/alerts")
def alerts(
    limit: int = Query(50, ge=1, le=500, description="events to return"),
    offset: int = Query(0, ge=0, le=100_000),
    kind: Literal["predicted_risk", "observed_flare"] | None = Query(
        None, description="only this kind of event"
    ),
    since: str | None = Query(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$",
        description="UTC date or timestamp",
    ),
):
    """Alert events, newest first. ``predicted_risk`` is a model forecast; ``observed_flare`` is a flare seen in the live GOES
    X-ray reading (not an official NOAA record)."""
    doc, origin = load_alerts()
    _sync_alerts(doc)
    store = get_store()
    if store is not None:
        total, events = store.query(limit, offset, kind, since)
    else:
        rows = [
            e
            for e in (doc or {"events": []})["events"]
            if (not kind or e["kind"] == kind) and (not since or str(e["time"]) >= since)
        ]
        rows.sort(key=lambda e: (str(e["time"]), str(e["id"])), reverse=True)
        total, events = len(rows), rows[offset : offset + limit]
    return {
        "updated": (doc or {}).get("updated"),
        "origin": origin,
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": events,
    }


@app.get("/api/forward-log")
def forward_log(
    limit: int = Query(120, ge=1, le=2000),
    resolved: bool | None = Query(None, description="only resolved (true) or pending (false) days"),
):
    """The daily forward test: each day's forecast and, once the day is over, whether an M flare occurred. Newest first."""
    doc, origin = load_forward_log()
    rows = list((doc or {"rows": []})["rows"])
    if resolved is not None:
        rows = [r for r in rows if r["resolved"] == resolved]
    rows.sort(key=lambda r: r["day"], reverse=True)
    return {"origin": origin, "total": len(rows), "rows": rows[:limit]}


# ---------------------------------------------------------------- admin endpoints


def _age(c: dict) -> float | None:
    return round(time.time() - c["fetched"], 1) if c.get("fetched") else None


@app.get("/api/admin/status", dependencies=[Depends(require_admin)])
def admin_status():
    store = get_store()
    return {
        "version": VERSION, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast": {"configured_url": bool(os.environ.get("FORECAST_URL")), "origin": _cache["origin"], "age_s": _age(_cache)},
        "alerts": {"configured_url": bool(os.environ.get("ALERTS_URL")), "origin": _aux["alerts"]["origin"], "age_s": _age(_aux["alerts"])},
        "forward_log": {"configured_url": bool(os.environ.get("FORWARD_LOG_URL")), "origin": _aux["forward_log"]["origin"], "age_s": _age(_aux["forward_log"])},
        "database": {"enabled": db_configured(), "open": store is not None, "events": store.counts() if store else {}, "error": _store_state["error"]},
        "background_ingestion": settings.background_enabled(),
    }  # fmt: skip


@app.post("/api/admin/refresh", dependencies=[Depends(require_admin)])
def admin_refresh():
    """Drop the caches and fetch every source again now."""
    return {"refreshed": refresh_all()}

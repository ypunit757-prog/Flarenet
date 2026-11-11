# FlareNet API (FastAPI, for Render or Docker)

Serves the forecast, the alert history and the daily forward test to the dashboard. It does **not** run the model, so it
needs no PyTorch and fits a small instance. The model runs in the scheduled GitHub job and publishes its files to the
`data` branch; this service reads those files, validates them, and caches them.

```
GET  /api/forecast        the payload written by scripts/export_web.py (checked before it is served)
GET  /api/alerts          alert events, newest first (?limit, ?offset, ?kind, ?since)
GET  /api/forward-log     the daily forward test (?limit, ?resolved)
GET  /api/health          status, where the data came from, how old it is, database state
GET  /docs                interactive API documentation (OpenAPI)
GET  /api/admin/status    caches, database counts          (needs the admin key)
POST /api/admin/refresh   drop the caches, fetch again    (needs the admin key)
```

Alert events have `kind: "predicted_risk"` (a model forecast) or `kind: "observed_flare"` (a flare seen in the live GOES
X-ray reading, not an official NOAA record). The two are never mixed.

## What it does

| Requirement | How |
|---|---|
| Validation | a source must be valid JSON of the expected shape (the forecast must hold the keys the dashboard reads) before it is cached or served; a bad one is rejected and the last good copy kept. Query parameters are validated by FastAPI (a bad one gets a 422 in the common error shape) |
| Errors | one JSON shape for every error: `{"error": {"code", "message", "request_id"}}`; no stack traces are returned |
| Caching | each source is cached for `FORECAST_CACHE_S`; if a source goes down the last good copy is served and `X-Forecast-Origin` says so |
| Logging | one line per request with a request id (`X-Request-ID`), plus warnings for source problems, to stderr |
| Admin | `/api/admin/*` need the header `X-Admin-Key` equal to `ADMIN_API_KEY` (at least 16 characters). With no key set the endpoints answer 404, so a deployment does not advertise them; a wrong or missing key gets 401. The key is compared in constant time |
| Database | alert events are stored in SQLite (Python standard library), parameterised queries only. The JSON on the `data` branch stays the source of truth; SQLite is a queryable copy that is rebuilt from it |
| Background job | when `FORECAST_URL` or `ALERTS_URL` is set, a thread refreshes the sources every `INGEST_INTERVAL_S` seconds (minimum 30) |

### Why SQLite and not PostgreSQL

The alert history is small (capped at 500 events), written by one job, and rebuilt from the `data` branch at any time. Render's
free plan has no persistent disk, so a SQLite file there is a cache that may be empty after a restart, and that is fine
because it refills from `ALERTS_URL`. PostgreSQL would add a paid managed database and a driver for data that is already
safely stored in Git. If you later need several API instances or user accounts, move `app/store.py` to PostgreSQL; its
methods (`add_events`, `query`, `counts`) are the whole interface.

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
python -m pytest tests -q
```

With no settings it serves `data/forecast.json`, a simulated demo (`"demo": true`), and the page says so.

## Run in Docker

```bash
docker build -t flarenet-api backend
docker run -p 8000:8000 -e FRONTEND_ORIGIN=http://localhost:5173 flarenet-api
# or the whole site (API on :8000, dashboard on :8080):
docker compose up --build        # from the project root
```

Mount a volume at `/srv/db` to keep the SQLite file across restarts (the compose file does).

## Settings (environment variables)

Copy `.env.example` to `.env` for a starting point. Nothing is required.

| Variable | Meaning |
|---|---|
| `FORECAST_URL` | URL of the real forecast JSON, e.g. the raw GitHub URL on the `data` branch. If it is down the last good copy is served |
| `ALERTS_URL` | URL of `alerts.json` (the alert history) on the `data` branch |
| `FORWARD_LOG_URL` | URL of `forward_log.csv` (the daily forward test), if published |
| `FORECAST_CACHE_S` | cache time for the forecast, default 120 |
| `AUX_CACHE_S` | cache time for the alert history and forward log |
| `FORECAST_FILE` | local forecast file instead, default `data/forecast.json` |
| `FRONTEND_ORIGIN` | allowed website(s), comma separated, e.g. `https://flarenet.vercel.app` |
| `FRONTEND_ORIGIN_REGEX` | optional, to allow Vercel preview URLs |
| `ADMIN_API_KEY` | enables `/api/admin/*`; 16 characters or more |
| `DATABASE_URL` | `sqlite:///data/flarenet.db` (default), an absolute path as `sqlite:////srv/db/flarenet.db`, or `off` |
| `FLARENET_BACKGROUND` | `1` or `0` to force the background refresh on or off |
| `INGEST_INTERVAL_S` | seconds between background refreshes, default 300 |
| `LOG_LEVEL` | `INFO` by default |

```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" https://<service>.onrender.com/api/admin/status
```

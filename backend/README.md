# FlareNet API (FastAPI, for Render)

Serves the forecast JSON to the frontend. It does not run the model, so it needs no PyTorch and fits a small Render instance.

```
GET /api/forecast   the payload written by scripts/export_web.py
GET /api/health     status, where the data came from, how old it is
GET /docs           interactive API docs
```

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
pytest                      # 5 tests
```

With no settings it serves `data/forecast.json`, a simulated demo (`"demo": true`).

## Settings (environment variables)

| Variable | Meaning |
|---|---|
| `FORECAST_URL` | URL of the real JSON, e.g. the raw GitHub URL on the `data` branch. Cached; the last good copy is served if the URL is down |
| `FORECAST_CACHE_S` | cache time for `FORECAST_URL`, default 120 |
| `FORECAST_FILE` | local file instead, default `data/forecast.json` |
| `FRONTEND_ORIGIN` | allowed website(s), comma separated, e.g. `https://flarenet.vercel.app` |
| `FRONTEND_ORIGIN_REGEX` | optional, to allow Vercel preview URLs |

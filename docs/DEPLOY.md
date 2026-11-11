# Deploying FlareNet: frontend on Vercel, API on Render

```
GitHub Actions (every 30 min)         Render (FastAPI)              Vercel (React)
 runs the model on live NOAA data  ->  serves /api/forecast  <-----  the dashboard
 publishes forecast.json to the        reads it via FORECAST_URL
 `data` branch
```

The API is light on purpose (no PyTorch). The model runs in the scheduled GitHub job, which has the memory for it.

## 1. Put the project on GitHub
```bash
git init && git add . && git commit -m "FlareNet"
git branch -M main
git remote add origin https://github.com/<you>/flarenet.git
git push -u origin main
```
Add your trained model first: `cp runs/goes_recent/checkpoint.pt models/flarenet.pt` (the `.gitignore` allows this file).
Use a public repository if you want the raw `data` branch URL to work without a token.

## 2. Publish live forecasts (GitHub Actions)
Repo > Actions > "Refresh forecast" > Run workflow. It creates the `data` branch with `forecast.json`.
Its raw URL is `https://raw.githubusercontent.com/<you>/flarenet/data/forecast.json`.

## 3. API on Render
1. Render > New > Blueprint > pick the repo (it reads `render.yaml`), or New > Web Service with root directory `backend`,
   build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
2. Environment variables:
   - `FORECAST_URL` = the raw URL above
   - `ALERTS_URL` = `https://raw.githubusercontent.com/<you>/flarenet/data/alerts.json` (the alert history, written by the same job)
   - `FRONTEND_ORIGIN` = your Vercel URL (set after step 4), e.g. `https://flarenet.vercel.app`
   - `ADMIN_API_KEY` = a long random string (the Blueprint generates one); it protects `/api/admin/*`. Leave it out to disable them
   - everything else is optional; see `backend/README.md` and `backend/.env.example`
3. Check `https://<service>.onrender.com/api/health`.
The free plan sleeps after idle time, so the first request can take up to a minute.

## 4. Frontend on Vercel
1. Vercel > Add New > Project > pick the repo. Set **Root Directory** to `frontend` (Vite is detected).
2. Environment variable `VITE_API_URL` = `https://<service>.onrender.com`.
3. Deploy. Then put the Vercel URL into Render's `FRONTEND_ORIGIN` and redeploy the API.

## 5. Alerts (optional)
The refresh job decides alerts with the rules in `src/flarenet/alerts/engine.py`: no alert from stale, simulated or invalid data,
one alert per episode, a cool-down between announcements, and forecasts kept apart from observed flares. It writes
`alerts.json` (the history the Alerts page shows) and `alert_state.json` (what the next run needs) to the `data` branch.
Nothing else is needed for the history to work.

**Browser notifications** need no setup. They work only while a FlareNet tab is open, because a web page cannot run in the background.

**E-mail** needs a mail account the job can log in to. In the repository: Settings > Secrets and variables > Actions.

| Secrets | |
|---|---|
| `SMTP_HOST`, `SMTP_PORT` | your mail server, e.g. `smtp.gmail.com`, `587` |
| `SMTP_USER`, `SMTP_PASSWORD` | the login (for Gmail, an app password, not your normal password) |
| `SMTP_SECURITY` | `starttls` (the default) or `ssl` (the default when the port is 465) |
| `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO` | sender, and recipients separated by commas |

| Variables (optional) | |
|---|---|
| `ALERT_LEVEL` | `sensitive`, `balanced` (default) or `strict` |
| `ALERT_THRESHOLD` | a score from 0 to 1; overrides the level |
| `ALERT_COOLDOWN_MIN` | minutes between forecast alerts, default 180 |
| `SITE_URL` | your dashboard URL, linked in the e-mail |

Without the secrets nothing is sent and nothing fails. The e-mail is sent only after the alert state has been saved, so one alert
is never sent twice, and the job queues runs instead of cancelling them for the same reason. GitHub may run the 30-minute schedule
late, so an alert can arrive late; the e-mail says when the forecast was made.

## 6. Docker (optional)
```bash
docker compose up --build          # API on :8000, dashboard on :8080
docker build -t flarenet-api backend      # the API alone
```
`backend/Dockerfile` runs as a non-root user, has a health check, and keeps the SQLite file in `/srv/db` (a volume).
The dashboard image (`frontend/Dockerfile`) bakes in the API address at build time (`--build-arg VITE_API_URL=...`).
Render and Vercel do not need Docker; it is for running everything on one machine.

## Database
The alert history is stored in SQLite inside the API (Python standard library). The JSON on the `data` branch is the source of
truth and the SQLite file is rebuilt from it, so Render's free plan (no persistent disk) loses nothing when it restarts.
PostgreSQL is deliberately not used: see `backend/README.md` for why and for what moving to it would involve.

## Adding the model comparison and explanation to the site
These are produced on your own computer, because they need the real data (see `docs/PROJECT_REPORT.md`, "Reproducing the results").
```bash
uv run python scripts/compare_models.py -c configs/noaa_all_chrono.yaml
cp runs/<run>_compare/model_comparison.json frontend/public/model_comparison.json

uv run python scripts/explain_model.py -c configs/noaa_all_chrono.yaml --checkpoint models/flarenet.pt --stack models/gbm6h_live.pkl
# commit models/explain_global.json; the refresh job puts it on the Explainable AI page
```
Until you do, the Results page says the comparison has not been run, and shows no numbers for it.

## What to expect
- Until the first live run succeeds, the API serves its bundled **simulated** demo and the page says so.
- If the model checkpoint is not committed, the refresh job skips and the demo stays.
- The headline numbers on the Results page are fixed values in `frontend/src/lib/validation.js`; update them when you re-evaluate. The model comparison appears when `frontend/public/model_comparison.json` exists.
- If the API is not set up (`VITE_API_URL` empty), the site looks for a `data.json` next to it and otherwise shows the bundled demo; the alert history and forward test show a note that they need the API.

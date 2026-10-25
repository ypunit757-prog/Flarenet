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
   - `FRONTEND_ORIGIN` = your Vercel URL (set after step 4), e.g. `https://flarenet.vercel.app`
3. Check `https://<service>.onrender.com/api/health`.
The free plan sleeps after idle time, so the first request can take up to a minute.

## 4. Frontend on Vercel
1. Vercel > Add New > Project > pick the repo. Set **Root Directory** to `frontend` (Vite is detected).
2. Environment variable `VITE_API_URL` = `https://<service>.onrender.com`.
3. Deploy. Then put the Vercel URL into Render's `FRONTEND_ORIGIN` and redeploy the API.

## What to expect
- Until the first live run succeeds, the API serves its bundled **simulated** demo and the page says so.
- If the model checkpoint is not committed, the refresh job skips and the demo stays.
- Results shown on the dashboard's Results page are fixed numbers in `frontend/src/lib/validation.js`; update them when you re-evaluate.

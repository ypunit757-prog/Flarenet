# FlareNet frontend (React + Vite, JavaScript)

A dashboard for the FlareNet model: current risk, flux and forecast timeline with zoom, a flare table showing what the model said beforehand, measured results, and a page to look at your own X-ray CSV.

## Run it

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Without data it shows a clearly labelled **demo** (simulated flux, not the real model).

## Connect the API

Set `VITE_API_URL` (copy `.env.example` to `.env.local`) to read from the Render API. If it is empty the app reads `public/data.json`.

## Show real model output

From the project root, export the model's forecast to the frontend's public folder, then reload the page:

```bash
uv run python scripts/export_web.py -c configs/nasa_goes_ncei.yaml \
  --checkpoint runs/goes_recent/checkpoint.pt --source goes_recent --days 14 \
  --metrics runs/goes_recent/metrics_report.json --out frontend/public/data.json
```

Use the same `--set sources.0.params.start=... end=...` date overrides you used for the case study.

Measured results shown on the Results page live in `src/lib/validation.js`. Edit them when you re-run an evaluation.

## Build and deploy

```bash
npm run build      # writes dist/
```

`dist/` is a static site. Routing uses the URL hash, so no server rewrites are needed.

- **Vercel:** root directory `frontend` (Vite is detected). Add `VITE_API_URL` as an environment variable. See `../docs/DEPLOY.md`.

## Layout

```
src/
  App.jsx                shell: sidebar, top bar, routes
  views/                 Forecast, Results, Analyze, About
  components/            Chart (SVG, drag to zoom), Gauge, Kpis, Chip, Banner
  lib/                   format, summarize, validation numbers, data loading, contexts
  assets/demo-data.json  simulated demo
  styles.css             design tokens (dark default, light theme) and components
```

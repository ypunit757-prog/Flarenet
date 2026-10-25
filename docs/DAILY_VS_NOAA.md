# Daily forecast vs NOAA, on the same days

`flarenet daily` is a second forecaster that is built to be compared with NOAA's own daily forecast.
It exists because the 6-hour deep model is a different product from NOAA's day-ahead probabilities,
and because the earlier daily test (TSS 0.36, Brier skill about -0.02) showed the limit was *information*,
not model size: NOAA's forecasters see sunspot regions; a flux-only model does not.

**Status: run once on real data (2019-03 to 2026-09). Result: level with NOAA through 2024, ahead of NOAA in 2025-26,
not established as better overall. See "First real run" below, and [`FORWARD_TEST.md`](FORWARD_TEST.md) for the test that
settles it.**

## What it does

For every UTC day T it makes one forecast at 22:00 UTC on day T-1 (NOAA's issue time) of "an M-class (or X-class)
flare starts during day T", using only data available at that moment:

| feature group | what | prefix |
|---|---|---|
| flux | log GOES flux now, mean / max / spread / quantiles over windows from 3 h to 7 days, 6 h trend, data coverage | `f_` |
| flare history | counts of C/M/X in the last 24 h / 3 d / 7 d, strongest flare, hours since the last M/X | `h_` |
| rotation | C/M flares and peak flux 27 days ago (+-2 days), when the same longitudes face Earth again | `rot_` |
| sunspot regions | NOAA Solar Region Summary of that day: number of regions, total and largest area, area growth, new regions, magnetic (Alpha/Beta/Gamma/Delta) counts, McIntosh class components (Z/penumbra/compactness), spot counts | `r_` |

Model: three shallow gradient-boosting classifiers blended with a regularised logistic regression, then Platt
calibration (fit on blocked out-of-fold predictions) and an alert threshold chosen on training days only.

## The test (why a win would mean something)

- **Walk-forward:** a test day is predicted by a model trained only on days before it (minus a 3-day gap). The first
  730 days are training only; after that the model is refit every 90 days.
- **Same days:** scored on the days where NOAA's own forecast (the RSGA product, "Event probabilities", issued 22:00 UTC)
  exists, so both are judged on identical days.
- **A fair NOAA baseline:** NOAA is scored as published, and also *recalibrated on the same earlier days and thresholded the
  same way* as FlareNet, so a win cannot come from NOAA being cut at an arbitrary 0.5 or being mis-calibrated for the
  current solar cycle. The recalibrated row is the one to take seriously.
- **Paired block bootstrap** over 14-day blocks (flare days cluster in time) for the difference FlareNet minus NOAA in AUC,
  Brier score and TSS.
- **Ablations:** flux only, + flare history, + rotation, + sunspot regions, so you can see what actually helps.

## Run it

```bash
uv sync --python 3.12 --extra netcdf
uv run flarenet noaa-text --start 2017-03-01 --end 2026-10-01    # downloads NOAA's SRS + RSGA text; prints how many parsed
uv run flarenet daily -c configs/daily_noaa.yaml                 # the full walk-forward test
```

First run downloads GOES daily files and about 7,000 small NOAA text files (cached in `data/`; re-run the same command if it
stops). Output in `runs/daily_noaa/`:

- `daily_report.md`: read this first.
- `daily_metrics.json`, `predictions.csv` (every out-of-sample day with FlareNet, NOAA, climatology), `day_table.csv`.
- `daily_model.pkl` and `daily_model_live.pkl` (for tomorrow's forecast, below).

Check `noaa-text` first: if it reports many unreadable files, the NOAA text layout has changed and the parsers need a fix
before any result means anything. Run the X-class version with `--set daily.target=X` (few X-class days, so wide intervals).

## How to read the report

1. Look at the **same-days table**. NOAA's published verification for earlier periods had an M-class AUC near 0.87 (a different
   period, so a rough sanity check only). If NOAA scores far from that here, suspect the data or labels first.
2. Look at **paired differences**, then at the section **"Is the edge consistent?"**. A pooled win (AUC and "Brier vs
   recalibrated NOAA" both in FlareNet's favour) is only a *consistent* win if it also holds inside each year, in both halves
   of the period, and against NOAA recalibrated on a rolling year. The section's first line says which of these happened.
   "No established difference" is the honest reading otherwise, and a tie with NOAA from public data is itself a good result.
3. Look at the **ablation table**. If "+ sunspot regions" does not beat "flux only", the SRS features are not adding anything
   and the model has nothing NOAA lacks.
4. Do **not** change settings after seeing the test period and then quote the new number. If you change the model, treat the
   result as exploratory and confirm on a later period (data after the first run).

## Tomorrow's forecast

```bash
uv run python scripts/daily_predict.py --model runs/daily_noaa/daily_model_live.pkl --out daily_outlook.json
```

Reads the last 7 days of GOES X-rays and the current sunspot-region summary from NOAA SWPC, prints the probability next to
NOAA's published 1-day number, and (with `--merge backend/data/forecast.json`) adds it to the dashboard JSON under `daily`.
The live model leaves out the rotation features, which need more than 7 days of history. Run it shortly after 22:00 UTC.
Retrain (`flarenet daily`) every few months so the model includes recent days.

## First real run (2019-03 to 2026-09, 2,740 days with both forecasts, 725 M-class days)

| | AUC FlareNet - NOAA | Brier FlareNet - NOAA recalibrated |
|---|---|---|
| all days | +0.016 [+0.006, +0.026] | -0.008 [-0.014, -0.003] |
| 2019-2024 (516 flare days) | +0.001 [-0.008, +0.010] | 0.000 [-0.005, +0.005] |
| 2025-2026 (209 flare days) | +0.066 [+0.039, +0.096] | -0.037 [-0.050, -0.024] |

- The pooled win comes from 2025-26. Through 2024 FlareNet matched NOAA. NOAA was significantly better in 2022 (AUC).
- AUC computed inside each year: +0.010 [-0.013, +0.031], no established difference. TSS at validated thresholds: tie.
- The 2025-26 gap is already there in Jan-Apr 2025, before the data source changed satellite, and it survives recalibrating
  NOAA on a rolling year (Brier -0.030 [-0.043, -0.018]). NOAA's own AUC fell from about 0.79 (2022-24) to 0.65-0.70.
- Sunspot-region features did not help (AUC -0.003 [-0.009, +0.001] versus flux only); the files parsed fine, so this is a
  finding, not a parser failure. FlareNet and NOAA probabilities correlate at 0.92; averaging them beats NOAA but not FlareNet.
- Not established: whether 2025-26 reflects a better model, a worse NOAA in the declining phase, or label differences
  (labels are this project's flare detection, NOAA verifies against its own list). Hence the forward test.

To re-check any run: `uv run python scripts/daily_consistency.py runs/daily_noaa/predictions.csv`.

## Limits

- The sunspot features come from NOAA's text summary (area, magnetic class, McIntosh class). NOAA's forecasters also use
  magnetograms, H-alpha and judgment. The next information to add, if the report shows the regions help but NOAA still
  wins, is SDO/HMI magnetic-field parameters per region (SHARP keywords from JSOC); they are not wired in here.
- Labels are this project's flare detection on the GOES flux; NOAA verifies against its own event list. Flares near midnight
  or inside data gaps can differ.
- One station of evidence: a result on one 2017-2026 stretch (mostly the rise of cycle 25) may not hold for the next cycle phase.

# Forward test: scoring forecasts made before the day happened

The walk-forward test in [`DAILY_VS_NOAA.md`](DAILY_VS_NOAA.md) is retrospective: every forecast was made by code
we could still change, scored on labels we defined, over data that is now final. A forward test removes those
doubts. A frozen model writes tomorrow's forecast into a log every day; days later the outcome is filled in; NOAA's
forecast for the same day sits next to it. Nobody can tune anything after seeing the result.

## What stops it being gamed

- **One frozen model.** Every row carries the model file's fingerprint. If it ever changes, the scorer says so and
  refuses a verdict. Retrained models get a new log.
- **Append-only.** A day already in the log is never rewritten (`append_forecast` refuses). Outcomes are only
  written into rows whose outcome is empty.
- **Logged in time.** Each row records when it was written. A forecast written after its own UTC day had begun is
  excluded from the score and counted.
- **Gaps stay visible.** Days with no forecast, and days with too little X-ray data to know the outcome, are counted
  in the report, never filled in.
- **Same definitions as the walk-forward test:** same features, same flare detection, NOAA scored as published and
  with the frozen recalibration fitted on the training days.

## Set it up (about 15 minutes)

```bash
# 1. Freeze the model from the run you already did (no downloads; seconds to a minute)
uv run python scripts/daily_freeze.py -c configs/daily_noaa.yaml --table runs/daily_noaa/day_table.csv --out models

# 2. Check that NOAA's live forecast can be read
uv run python scripts/daily_predict.py --check-noaa

# 3. Try one day by hand (after 22:00 UTC = 03:30 IST)
uv run python scripts/daily_predict.py --model models/daily_model_live.pkl --log forward_log.csv
```

Step 2 should print `OK` with NOAA's issue time and the M and X probabilities. The live source is NOAA's
"3-day Space Weather Predictions" file (`Whole_Disk_Flare_Prob`, class M and X, for the next three UTC days). The
walk-forward test used the archived daily report (RSGA), so `daily_forward_score.py` also fetches the archive for
logged days, fills any NOAA number the live read missed, and reports how many days the two sources disagree. If it
says they differ, trust the archive and tell whoever maintains this.

Run step 3 between 22:00 and 24:00 UTC (03:30-05:30 IST). A forecast written after its own UTC day began is
excluded from scoring, so a run at 00:04 UTC is only a test: delete `forward_log.csv` before the real start.

## Run it every day

Best: let GitHub do it, so a sleeping laptop cannot leave gaps.

1. Commit `models/daily_model_live.pkl` and push the repository to GitHub.
2. The workflow `.github/workflows/daily-forward-log.yml` runs at 22:20 UTC (03:50 IST) and publishes
   `forward_log.csv` and `forward_report.md` to the `data` branch. Start it once with "Run workflow" in the Actions tab.
   This workflow has not been run on GitHub yet; check the first run in the Actions tab.

Or on your own machine, a daily cron job (the machine must be awake):

```
20 22 * * * cd /path/to/flarenet && uv run python scripts/daily_predict.py --model models/daily_model_live.pkl --log forward_log.csv
```
(cron uses the machine's local time; 22:20 UTC is 03:50 IST.)

The log resolves outcomes from the NOAA X-ray feed, which holds only the last 7 days: run the job at least weekly or
older days stay unresolved.

## Read it

```bash
uv run python scripts/daily_forward_score.py forward_log.csv --model models/daily_model_live.pkl
```

The first line is the verdict. It says **"Too early to read"** until there are at least 120 resolved days with at
least 30 M-class days. At a flare rate near 20% of days that is roughly five to six months. After that the verdict is
"FlareNet is ahead", "NOAA is ahead" or "No established difference", using the same paired intervals as before.

## Limits

- Outcomes use the live feed's preliminary X-ray data and this project's flare detection, not NOAA's event list.
- The fixed model gets older as the solar cycle changes; that is part of what the test measures.
- A few months of one regime can still mislead; a second period (or X-class) adds evidence.

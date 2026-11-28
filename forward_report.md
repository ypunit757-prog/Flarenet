# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 33 and 6.**

- Forecast days logged: 35; resolved (outcome known): 33; unresolved: 1; written up to 6 h after their day began (counted): 34; written later than that (excluded): 1; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-27; flare days: 6.

## Days with both forecasts (33 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.802 | 0.128 | 0.140 | 0.407 |
| NOAA as published (TSS at 0.5) | 0.830 | 0.122 | 0.180 | 0.259 |
| NOAA recalibrated (frozen mapping) | 0.830 | 0.123 | 0.170 | 0.500 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 24 and 5.**

- Forecast days logged: 25; resolved (outcome known): 24; unresolved: 1; written up to 6 h after their day began (counted): 25; written later than that (excluded): 0; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-17; flare days: 5.

## Days with both forecasts (24 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.779 | 0.146 | 0.117 | 0.337 |
| NOAA as published (TSS at 0.5) | 0.789 | 0.141 | 0.144 | 0.295 |
| NOAA recalibrated (frozen mapping) | 0.789 | 0.144 | 0.127 | 0.326 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

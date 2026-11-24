# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 30 and 6.**

- Forecast days logged: 31; resolved (outcome known): 30; unresolved: 0; written up to 6 h after their day began (counted): 30; written later than that (excluded): 1; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-23; flare days: 6.

## Days with both forecasts (30 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.778 | 0.141 | 0.121 | 0.375 |
| NOAA as published (TSS at 0.5) | 0.809 | 0.134 | 0.161 | 0.250 |
| NOAA recalibrated (frozen mapping) | 0.809 | 0.136 | 0.151 | 0.458 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

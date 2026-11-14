# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 20 and 4.**

- Forecast days logged: 21; resolved (outcome known): 20; unresolved: 1; written up to 6 h after their day began (counted): 21; written later than that (excluded): 0; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-13; flare days: 4.

## Days with both forecasts (20 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.812 | 0.137 | 0.146 | 0.500 |
| NOAA as published (TSS at 0.5) | 0.828 | 0.125 | 0.219 | 0.375 |
| NOAA recalibrated (frozen mapping) | 0.828 | 0.128 | 0.201 | 0.625 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

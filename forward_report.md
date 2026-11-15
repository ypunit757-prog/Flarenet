# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 21 and 4.**

- Forecast days logged: 22; resolved (outcome known): 21; unresolved: 1; written up to 6 h after their day began (counted): 22; written later than that (excluded): 0; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-14; flare days: 4.

## Days with both forecasts (21 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.809 | 0.133 | 0.139 | 0.456 |
| NOAA as published (TSS at 0.5) | 0.838 | 0.121 | 0.215 | 0.382 |
| NOAA recalibrated (frozen mapping) | 0.838 | 0.124 | 0.196 | 0.588 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

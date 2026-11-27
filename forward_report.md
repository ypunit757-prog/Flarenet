# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 32 and 6.**

- Forecast days logged: 34; resolved (outcome known): 32; unresolved: 1; written up to 6 h after their day began (counted): 33; written later than that (excluded): 1; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-26; flare days: 6.

## Days with both forecasts (32 days)

| forecaster | AUC | Brier | Brier skill | TSS (own threshold) |
|---|---|---|---|---|
| FlareNet | 0.795 | 0.132 | 0.134 | 0.397 |
| NOAA as published (TSS at 0.5) | 0.824 | 0.126 | 0.174 | 0.256 |
| NOAA recalibrated (frozen mapping) | 0.824 | 0.127 | 0.164 | 0.487 |

Paired intervals need at least 60 days with both forecasts.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

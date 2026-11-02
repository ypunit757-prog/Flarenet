# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**Too early to read: need at least 120 resolved days with at least 30 flare days; have 8 and 0.**

- Forecast days logged: 9; resolved (outcome known): 8; unresolved: 1; written up to 6 h after their day began (counted): 9; written later than that (excluded): 0; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.
- Period: 2026-10-25 to 2026-11-01; flare days: 0.

Fewer than 20 resolved days have both a FlareNet and a NOAA forecast logged, so there is no comparison yet.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

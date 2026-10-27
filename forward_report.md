# Forward test: FlareNet vs NOAA on days that had not happened when the forecast was made

**No scored days yet.**

- Forecast days logged: 3; resolved (outcome known): 0; unresolved: 0; logged after their day began (excluded): 3; calendar days with no forecast logged: 0.
- Model fingerprints in the log: 2e0f2e52aa.

Fewer than 20 resolved days have both a FlareNet and a NOAA forecast logged, so there is no comparison yet.

## Notes

- Outcomes come from the live X-ray feed (preliminary data), with the same flare detection as training.
- A day is resolved only if enough X-ray data covers it; gaps are left empty, never guessed.
- Do not retrain or change the model during the test. If you do, start a new log.

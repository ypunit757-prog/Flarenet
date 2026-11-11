# FlareNet: presentation outline, demo script and viva questions

About 15 minutes: 12 slides, a 3-minute live demo, then questions. Numbers below are the ones measured in `docs/PROJECT_REPORT.md`;
do not add others. The diagram images are in `docs/diagrams/`.

## Slides

**1. Title.** FlareNet: forecasting and alerting for solar flares from GOES X-ray data. Your name, guide, college.

**2. Why it matters.** The Sun's strongest flares (class M and X) can disturb radio communication and satellite operations.
Agencies forecast them; the question here is how much a small, public-data system can do, and how honestly it can be tested.
*Say:* flare classes are ten times apart: M from 1e-5 W/m², X from 1e-4 W/m².

**3. Problem and objectives.** Forecast "M or X flare within 6 hours" every 30 minutes; explain it; alert without ever alerting on bad data; test on data the model never saw.
*Show:* the six objectives from report section 1.3.

**4. What exists, and the gap.** NOAA's forecasts use sunspot and magnetic information. Research models use magnetogram images.
FlareNet uses only the public X-ray flux, so it can run live; its contribution is the end-to-end, honestly tested system.
*Show:* the comparison table from report section 2.2.

**5. Architecture.** `diagrams/architecture.png`. Walk left to right: SWPC feed, scheduled GitHub job (model, explanation, alert rules), `data` branch, API, dashboard, e-mail.
*Say:* the model runs in the scheduled job, not in the API, which keeps the API cheap.

**6. Data and the test that cannot cheat.** The split table: train earliest 78% of GOES-16, validate on a later year, final test on GOES-18, a different satellite, after April 2025.
*Say:* random splits leak between neighbouring minutes. A chronological split with a gap does not.

**7. The model.** The CNN-BiLSTM diagram, 122,087 parameters; calibrated probability (Platt); three alert levels; blend with a tree model.
*Say:* why TSS and not accuracy: "never alert" is already over 90% accurate on a rare event.

**8. Results (measured).** On GOES-18: ROC-AUC 0.83, TSS 0.49, 64% of pre-flare times alerted, 61% of alerts false, calibration error 0.014.
Network alone 0.81 / 0.45; the blend's gain is +0.022 AUC (+0.011 to +0.032).
*Say clearly:* the gain on a weaker GOES-16 check was only +0.009, so read it as +0.01 to +0.02.
**[After you run the comparison:]** a slide with the LSTM / Random Forest / XGBoost table and the ROC curves. Use only numbers from `model_comparison.md`.

**9. Explainable AI.** The Explain page: bars for each input group, sign written, "typical window" baseline, exact Shapley values; global importance from `explain_model.py`.
*Say:* it explains the model, not the Sun. **[After you run Appendix A:]** name the top three input groups.

**10. Alerts.** The rules: no alert from demo, stale (3 hours) or invalid data; one per episode; cool-down; forecast and observed flares kept apart; browser and e-mail; history.
*Show:* the sequence diagram or the Alerts page.

**11. Engineering.** API with validation, JSON errors, admin key, SQLite, cache, background refresh; Docker; CI; 283 + 33 + 40 tests.
*Show:* `diagrams/er.png` briefly; say why SQLite (small, rebuildable) rather than PostgreSQL.

**12. Limits, future work, conclusion.** Flux only; not shown to beat NOAA; no X-class forecast (nowcast found 0 of 13); alerts are not official; Aditya-L1 untested.
Future: magnetogram features, Aditya-L1 test, web push, X-class model. Conclusion: measured skill, calibrated, honest, reproducible.

## Live demo (3 minutes)

1. **Forecast page.** Point at the state word, the probability against the alert line, the 7-day chart, the list of flares with "warned / not warned".
2. **Explain page.** "These bars are why the score is what it is." Read the biggest raising group and the biggest lowering group.
3. **Alerts page.** Turn alerts on, choose a level, press "Send a test notification". Show the server alert history and say what "forecast" and "observed" mean.
4. **Results page.** Show the "not shown" items (X-class nowcast, Aditya-L1) before anyone asks.
5. **Fallback if the internet or the API is down:** the site shows the simulated demo and says so. Say that this is by design; do not present demo numbers as real.

## Likely questions, and honest answers

**Is it better than NOAA?** Not shown. For the 6-hour forecast there is no like-for-like NOAA product. For the daily forecaster, one retrospective run was level with NOAA through 2024 and ahead in 2025 and 2026, but not established as better overall; the forward test decides it.

**Why only X-ray data?** It is public, live and small, so the whole system can run for free. The cost is that the model cannot see sunspot regions, which limits its skill.

**Why TSS?** It is hit rate minus false-alarm rate, so it does not reward a model for saying "no flare" on a rare event, and it does not change with how rare flares are (Bloomfield et al., 2012).

**What does 61% false alarms mean?** About six in ten alerts were not followed by an M/X flare within 6 hours at the default level. The sensitive level catches more flares with more false alarms; the strict level is the reverse.

**How do you know it is not overfitted?** Training, calibration and test are different periods; the test is another satellite. Intervals come from block resampling. The weaker GOES-16 check shows a smaller gain, which is stated.

**What is "calibrated"?** Among times the model scores 30%, about 30% are followed by a flare. Calibration error was 0.014 on GOES-18.

**Why is the model not in the API?** The API would need PyTorch and memory; the scheduled job has both. The API only serves files, so it is small.

**What if the data is old or broken?** The page says "out of date" or shows demo; the alert rules refuse data older than 3 hours, simulated data, or an invalid score. Nothing is announced.

**Why SQLite?** The history is small, written by one job and rebuilt from the `data` branch, and the free host has no persistent disk. PostgreSQL would add a database for data already stored in Git. The store is three methods, so it can be replaced.

**Why does the notification not come when the browser is closed?** A web page cannot run in the background. E-mail does, from the scheduled job. Web push would be the next step.

**How does the explanation work?** Exact Shapley values over about a dozen groups of inputs: each group is switched from a typical value to its real value in every order, and the average change in the score is its contribution; they add up exactly to the difference from the typical score.

**What would you do next?** Add magnetogram features and measure the gain; test Aditya-L1 data once there are weeks of it; build an X-class model; add web push.

## Checklist before the day
- Run the comparison and explanation (report Appendix A) and fill the blanks in report 9.2 and 9.3.
- Open the site once beforehand (free hosting may be asleep) and have a screenshot of each page as a backup.
- Know the three numbers: AUC 0.83, TSS 0.49, 61% false alarms. Know what is not shown.

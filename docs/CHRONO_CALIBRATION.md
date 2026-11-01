# A fairer validation set and calibrated probabilities for the 6-hour model

## Why

The first 6-hour model (`configs/noaa_all.yaml`) chooses its alert threshold on a validation set made by shuffling
3-day blocks over eight years of GOES-16. Those blocks sit inside the same active periods as the training blocks, so the
validation set is easier than a new period: validation TSS was 0.78, the same model scored TSS 0.44 on GOES-18. The
threshold (0.48) was tuned on the optimistic set. Training also uses re-balanced batches, so the raw output is not a
probability: a score of 0.60 does not mean 60 % of such windows were followed by an M/X flare.

## What changed

| | before | now (`configs/noaa_all_chrono.yaml`) |
|---|---|---|
| validation | shuffled 3-day blocks | a later year: train = earliest 78 % of the time span, validation = next 12 %, test = last 10 %, with a 13.5-hour gap (window + horizon) cut off the end of train and validation |
| alert threshold | TSS-optimal on that validation set | TSS-optimal on the later validation year, curve smoothed over +-2 steps |
| probabilities | raw model output | Platt scaling fitted on the validation windows (monotone: AUC and the ranking of windows are unchanged) |
| alert levels | one threshold | three, fixed on validation and stored in the checkpoint: `sensitive` (catches at least 90 % of flare windows), `balanced` (best TSS, the default), `strict` (at most 50 % false alarms) |

A level is stored as "not reachable" rather than faked when no threshold with positive skill meets its target.
Old checkpoints (no calibration, one threshold) load and score exactly as before; the new options default to the old
behaviour (`data.split: block`, `training.calibrate: none`).

## Run it

```bash
uv run flarenet run -c configs/noaa_all_chrono.yaml      # reuses the data already downloaded; only retrains
uv run python scripts/compare_runs.py runs/goes_all runs/goes_all_chrono
```

Outputs are in `runs/goes_all_chrono/` (`report.md` has the alert-level table and the usual metrics, `reliability.png`
the calibration plot, `checkpoint.pt` the model).

## Reading the comparison

Compare on `goes18_test` only: both runs never trained on it. The held-out "goes16_all" rows are not comparable (a
shuffled-block test against the latest 10 % of the series).

- **AUC and TSS intervals overlap:** ranking skill is not shown to differ. That is a fine result: the point is an
  honest threshold and meaningful probabilities, not a higher score.
- **Brier score and calibration error (ECE) lower:** the probabilities are closer to the real flare frequency.
- **FAR lower at a similar TSS:** fewer false alarms for the same skill.
- **TSS clearly lower:** the chronological split costs training data (the most active months are used for validation and
  test instead of training). Say so; do not publish the new model as an improvement.

Nothing here has been run on the real GOES data from the build environment (the NOAA/NCEI servers are not reachable
there): only on simulated data and unit tests. The numbers come from your run.

## First real run (GOES-16 2017-03 to 2025-04 for training, GOES-18 2025-04 to 2026-09 for testing)

Same data and model; only the split, the calibration and the threshold rule differ. Both runs scored on GOES-18, which neither
trained on (393 M/X flares):

| | block split, raw (`goes_all`) | chronological, calibrated (`goes_all_chrono`) |
|---|---|---|
| validation TSS (the number the threshold was tuned on) | 0.78 | 0.41 |
| forecast AUC | 0.808 [0.77, 0.84] | 0.810 [0.77, 0.84] |
| forecast TSS | 0.443 [0.37, 0.50] | 0.446 [0.36, 0.52] |
| HSS | 0.224 [0.18, 0.27] | 0.374 [0.30, 0.43] |
| alert threshold | 0.48 | 0.19 |
| windows alerted among forecast-positive (POD) / false alarms among alerts (FAR) | 0.81 / 0.75 | 0.58 / 0.60 |
| M/X flares warned in time | 364 of 393 | 309 of 393 |
| Brier score | 0.201 | 0.090 |
| calibration error (ECE) | 0.321 | 0.010 |

What it shows:

- The diagnosis held. The block-split validation TSS (0.78) was far above what a new satellite and period gave (0.44); the
  chronological validation TSS (0.41) agrees with it (0.45).
- Ranking skill is unchanged (AUC and TSS identical within their intervals), as expected from a monotone calibration.
- The probabilities are now meaningful: a calibration error of 0.01 on an unseen satellite and period, from 0.32.
- The new default alert level is more conservative: fewer false alarms (60% against 75%) and fewer flares warned (309 against
  364). That is a trade, not a free gain; TSS is the same. HSS is higher mostly because it rewards fewer false alarms.
- Levels fixed on validation do not fully transfer: the balanced level promised POD 0.72 on validation and gave 0.58 on
  GOES-18, the sensitive level promised 0.90 and gave 0.73. The false-alarm rates transferred (0.60 to 0.60 for balanced).
  Expect a further drop in flares caught on a period that differs from validation.
- The nowcast also scored higher on GOES-18 in this run (accuracy 0.974, macro-F1 0.55, against 0.94 and 0.42), but it was
  one run, and this project's multi-seed tests (on simulated data) showed a large seed-to-seed spread in nowcast macro-F1, so that is not shown to be a real gain.

## Promoting the new model

If the candidate is level or better on GOES-18:

```bash
cp runs/goes_all_chrono/checkpoint.pt models/flarenet.pt
git add models/flarenet.pt frontend/src/lib/validation.js && git commit -m "promote the chronological, calibrated 6 h model" && git push
```

Commit the checkpoint and the dashboard numbers (`frontend/src/lib/validation.js`) together, so the site never shows one
model's numbers beside the other's forecast. The next Refresh forecast run publishes calibrated scores and the three alert
levels (the dashboard lists them in "How to read this page"). The daily model and the forward test are separate and are not
affected.

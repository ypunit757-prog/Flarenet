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

## Promoting the new model

If the candidate is level or better on GOES-18:

```bash
cp runs/goes_all_chrono/checkpoint.pt models/flarenet.pt
git add models/flarenet.pt && git commit -m "6 h model: chronological validation, calibrated probabilities" && git push
```

The next Refresh forecast run publishes calibrated scores and the three alert levels (the dashboard shows them in "How to
read this page"). The daily model and the forward test are separate and are not affected.

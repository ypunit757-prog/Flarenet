Put the trained checkpoint used for the live forecast here as `flarenet.pt`:

```bash
cp runs/goes_all_chrono/checkpoint.pt models/flarenet.pt   # chronological validation, calibrated (see docs/CHRONO_CALIBRATION.md)
```

It is small (about 1.5 MB), and `.gitignore` allows this one file. The scheduled job in
`.github/workflows/refresh-forecast.yml` skips itself until the file exists.

## Frozen daily model (forward test)

`flarenet daily` and `scripts/daily_freeze.py` write `daily_model_live.pkl`. To run the forward test on GitHub, commit
that one file here (it is small). Do not replace it while the test runs; see `docs/FORWARD_TEST.md`.

## Tree-model blend for the 6-hour forecast

`gbm6h_live.pkl` is written by `scripts/gbm_6h.py --live-compatible --install models/gbm6h_live.pkl` (see
`docs/IMPROVING_FURTHER.md`). It belongs to exactly one `flarenet.pt`: it records that file's SHA-256, and the live export
ignores it (showing the network alone, with the reason) for any other. Re-create it whenever `flarenet.pt` changes.

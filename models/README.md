Put the trained checkpoint used for the live forecast here as `flarenet.pt`:

```bash
cp runs/goes_recent/checkpoint.pt models/flarenet.pt
```

It is small (about 1.5 MB), and `.gitignore` allows this one file. The scheduled job in
`.github/workflows/refresh-forecast.yml` skips itself until the file exists.

## Frozen daily model (forward test)

`flarenet daily` and `scripts/daily_freeze.py` write `daily_model_live.pkl`. To run the forward test on GitHub, commit
that one file here (it is small). Do not replace it while the test runs; see `docs/FORWARD_TEST.md`.

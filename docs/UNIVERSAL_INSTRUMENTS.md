# One model for ISRO, NASA/NOAA and other agencies' X-ray data

FlareNet only needs a **soft X-ray lightcurve** (plus, optionally, a hard channel). It does not care
which spacecraft produced it. What *does* differ between instruments is the absolute flux level
(gain, background, spectral band), the noise, and whether a hard band exists. This page lists what the
code does about that, how to plug in a new agency's data, and how to compare against NOAA honestly.

> **Status.** The robustness features are tested and measured on a *simulated* detector shift
> (numbers in `docs/ROBUSTNESS_RESULTS.md`). No real GOES / Aditya-L1 / NOAA result exists in this
> repository: the NASA/NOAA servers were unreachable from the build environment. "Better than NOAA"
> is therefore **not demonstrated**; the commands below produce that evidence on your machine.

## What makes the model instrument-independent

| Option | Where | Effect |
|---|---|---|
| `features.relative: true` | `pipelines/features/channels.py` | log-flux, rolling max, EWMA, fluence and hardness are measured in decades above the detector's own *trailing* 10th-percentile baseline (causal, so it works live). A gain change or constant background offset no longer moves the inputs. |
| `training.aug_level_jitter` | `pipelines/training/dataset.py` | random per-window gain shift (log10 units) of soft and hard level channels. For absolute-feature models; not allowed together with `relative`. |
| `training.aug_hxr_dropout` | same | with this probability the hard channel is replaced by the soft one, so the model also works for instruments with no hard band (the pipeline substitutes it the same way at inference). |
| `training.aug_noise_max` | same | extra noise with a random sigma per window. |
| `scaling: goes_calibrated` | `pipelines/preprocessing/scaling.py` | maps an instrument's counts onto GOES W/m2: `reference_csv` (fit `log10 GOES = a + b log10 counts` on overlapping minutes, R2 stored in the bundle meta), `gain/power/background`, or a one-number `quiet_flux` anchor. |
| `synthetic_instrument` source | `sources/synthetic.py` | the synthetic Sun seen through another detector (gain, power, background, noise, no/own hard channel) for controlled shift tests. |

The evaluation refuses to score a checkpoint with different `features.relative` settings than it was
trained with (otherwise the inputs would be silently mis-scaled).

## Adding another agency's data

1. Put the lightcurve in a CSV/Parquet/NetCDF with a time column and a soft-band flux column.
2. Use the generic reader:
   ```yaml
   - name: other_agency
     kind: nasa_goes_file
     role: eval
     params: {paths: [/data/xrs/*.csv], time_column: time, time_format: unix,
              sxr_column: flux_soft, hxr_column: null}
   ```
   `hxr_column: null` is fine: the hard channel is then a copy of the soft one.
3. If the units are not W/m2, add `scaling: goes_calibrated` with `scaling_params` (above).
4. If the instrument is a new file format, write a small `DataSource` (`docs/ADDING_A_SOURCE.md`).

Labels for non-GOES instruments are either your own catalogue (`labeler: catalog`) or the relative
proxy; neither equals NOAA's verified classes, so cross-agency scores are *not* the same quantity as
GOES scores. Reports say so.

## Comparing with NOAA

NOAA SWPC publishes one probability per UTC day for "an M-class (or X) flare in the next 24 h".
`flarenet noaa-compare` reproduces that protocol: the model (trained with `data.horizon_min: 1440`)
is queried at 00:00 UTC using only the previous 90 minutes, and scored with TSS/HSS/AUC/Brier at
theta 0.5 and at its own tuned threshold, with the climatology Brier skill.

```bash
flarenet train -c configs/noaa_daily.yaml --run-name noaa_daily
flarenet noaa-compare -c configs/noaa_daily.yaml --checkpoint runs/noaa_daily/checkpoint.pt \
    --source goes18_2024 --noaa-csv noaa_daily_probs.csv
```

`noaa_daily_probs.csv` (columns `date`, `prob_m`) holds NOAA's own forecasts for the same days, taken
from SWPC's published forecast archive. With it you get a **same-days** comparison. Without it you only
get the model's numbers next to published NOAA figures from a different period (TSS 0.44, HSS 0.47,
AUC 0.87, Brier 0.11 for daily M-class, arXiv:2508.01114), which is context, not a verdict.

Things to keep in mind before claiming to beat NOAA:
* NOAA forecasters use active-region magnetograms and images; this model sees 90 minutes of X-ray flux.
  X-ray history alone may well not match them for 24 h forecasts.
* Use a held-out *period* and another satellite, never the training days.
* Report intervals: a few dozen M-flare days give wide bootstrap intervals.
* Climatology, persistence and the flux-rule nowcast (`baselines` in `report.md`) must be beaten first.

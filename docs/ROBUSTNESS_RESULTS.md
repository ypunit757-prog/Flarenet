# Instrument-shift benchmark (simulated detectors)

**What this is.** Train on the synthetic detector, then score the *same kind of synthetic Sun* seen through
other simulated detectors (never trained on, different seeds). 4 variants x 4 seeds, 150 training days,
single model (no ensemble), alert threshold fixed from training/validation. Reproduce:

    flarenet experiment -c configs/instrument_shift.yaml -e configs/experiments/instrument_shift.yaml --out runs/shift --jobs 2

**What this is not.** Real ISRO / NASA data, or a comparison with NOAA. The shifts are my own simple
models of "another instrument" (gain, nonlinearity, background, noise, no/own hard channel).

Detectors: `ctrl_same_detector` (control), `shift_gain_noise` (gain x40, noise 5 %), `shift_no_hard_channel`
(gain x0.02, flux^1.15, no hard band), `shift_noisy_background` (gain x5, +1x background, noise 25 %, hard gain x30).
Mean +- std over 4 seeds.

**Forecast TSS (6 h, M/X)**

| variant | in-distribution test | control | gain+noise | no hard channel | noisy background |
|---|---|---|---|---|---|
| baseline (absolute features) | 0.546 | 0.650 | 0.538 +- 0.043 | 0.313 +- 0.172 | 0.054 +- 0.065 |
| augment (level jitter, hxr dropout, noise) | 0.557 | 0.654 | 0.569 +- 0.016 | 0.297 +- 0.133 | 0.138 +- 0.100 |
| **relative** (`features.relative`) | 0.515 | 0.626 | 0.628 +- 0.039 | 0.472 +- 0.083 | 0.515 +- 0.048 |
| **relative + augment** | 0.510 | 0.626 | 0.639 +- 0.022 | 0.509 +- 0.012 | 0.504 +- 0.023 |

**Forecast AUC**: baseline 0.889 / 0.859 / 0.809 / 0.777; relative 0.887 / 0.877 / 0.807 / 0.844;
relative+augment 0.887 / 0.883 / 0.830 / 0.831 (control / gain+noise / no hard / noisy background).

**Nowcast macro-F1**: baseline 0.613 / 0.301 / 0.247 / 0.147; relative 0.559 / 0.566 / 0.426 / 0.449;
relative+augment 0.562 / 0.557 / 0.527 / 0.429.

Full per-run table: `docs/shift_experiment_results.csv`, paired summary: `docs/shift_experiment_summary.md`.

**Reading it honestly**
* Relative features remove most of the collapse under detector change (TSS 0.05 -> 0.50, macro-F1 0.15 -> 0.43).
* The price is about 0.03 TSS (and about 0.06 nowcast macro-F1) on the training detector itself; with 4 seeds
  and +-0.05 spread this is suggestive, not established. Absolute features remain the better choice when
  training and deployment use the same instrument.
* Level-jitter/dropout augmentation alone helps little; the gain comes from the feature change.
* All of this is on a simulated Sun. How much transfers to Aditya-L1 SoLEXS / HEL1OS is unknown until you run
  `configs/universal_multi_agency.yaml` on real files.

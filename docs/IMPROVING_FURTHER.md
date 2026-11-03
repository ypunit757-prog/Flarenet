# Improving the 6-hour model further, and measuring whether it worked

Every tool below produces a number on data the model has not seen; a change is kept only if that number says so. The
real GOES data could not be reached from the environment this was written in, so every figure in this file comes from
a run on your machine. The first results are recorded at the end.

## What was added

| tool | question it answers | cost | command |
|---|---|---|---|
| paired comparison | is run B really better than run A, on the same windows? | seconds | `scripts/compare_runs.py A B --paired` |
| tuner | which training settings work best, judged on the validation period only? | 21 training runs with the defaults (13 trials, then 8 re-runs) | `scripts/tune_6h.py` |
| five-network average | does averaging seeds beat one network? | 5 x one training run | `configs/noaa_all_chrono_ens5.yaml` |
| tree model + stack | do long-memory features add to the 90-minute window? | 5 to 10 min, no new download | `scripts/gbm_6h.py` |

### 1. Paired comparison

The first chronological run had the same AUC and TSS as the old one, with intervals of about +-0.07 on 393 flares. Two
overlapping intervals do not show two models are equal, only that this sample cannot tell them apart; a gain of 0.02
AUC would be invisible. UFCORIN, a published GOES-flux flare predictor, found TSS moving by 0.1 to 0.2 between
cross-validation splits, more than the differences between its input strategies, and argued that comparisons must be
made on the same data over several splits ([Muranushi et al., 2015](https://arxiv.org/abs/1507.08011)).
`--paired` scores both models on identical windows and bootstraps whole 72-hour blocks of windows, so the interval is
for the *difference*. Each run must have saved its window predictions, which the evaluation now does; a run made
before that is re-scored without retraining:

```bash
uv run flarenet evaluate -c configs/noaa_all.yaml        --checkpoint runs/goes_all/checkpoint.pt        --run-name goes_all
uv run flarenet evaluate -c configs/noaa_all_chrono.yaml --checkpoint runs/goes_all_chrono/checkpoint.pt --run-name goes_all_chrono
uv run python scripts/compare_runs.py runs/goes_all runs/goes_all_chrono --paired
```

### 2. Tuning on validation only

```bash
uv run python scripts/tune_6h.py -c configs/noaa_all_chrono.yaml --trials 12 --confirm 3
```

Random search over learning rate, dropout, weight decay, label smoothing, focal-loss gamma, forecast loss weight, batch
balancing, weight averaging, network width and the checkpoint-selection metric. Windows are built once and reused.

- Each trial is trained on the training period and scored on the **validation** period, after the same Platt
  calibration the real run uses. GOES-18 is never read, so it stays an honest final test.
- Trials are ranked by validation AUC (no threshold involved; the threshold-dependent TSS is the noisiest metric),
  with Brier score next to it.
- The best of many noisy trials looks better than it is. So trial 0 (your config as it stands) and the top three are
  re-run with two more seeds, and **only a setting run with every seed can win**; the table shows the seed spread. A
  gain smaller than that spread is noise.
- Output: `runs/tune_6h/trials.csv`, `tuning_report.md`, `best_config.yaml`. `--resume` continues an interrupted search.

Then run the winner once and test it against the current model on GOES-18, where the verdict is made:

```bash
uv run flarenet run -c runs/tune_6h/best_config.yaml
uv run python scripts/compare_runs.py runs/goes_all_chrono runs/goes_all_chrono_tuned --paired
```

### 3. Five-network average

```bash
uv run flarenet run -c configs/noaa_all_chrono_ens5.yaml
uv run python scripts/compare_runs.py runs/goes_all_chrono runs/goes_all_chrono_ens5 --paired
```

The committed model is a single network, and the earlier "three-model ensemble" wording on the About page was wrong and
has been removed. A small network trained on a few hundred flares varies from seed to seed; averaging is the standard
remedy. A recent magnetogram-transformer system reviewed by [Shao et al.](https://arxiv.org/pdf/2511.20465) averages ten
models, but that review gives no head-to-head number for averaging against a single model, so the size of the gain here
is unknown until the paired table says.

### 4. Tree model on long-memory features, stacked with the network

```bash
uv run python scripts/gbm_6h.py -c configs/noaa_all_chrono.yaml
uv run python scripts/gbm_6h.py -c configs/noaa_all_chrono.yaml --live-compatible   # only features the live feed can give
```

The network sees the last 90 minutes. Whether an M/X flare starts in the next six hours also depends on the last days:
how active the Sun has been, how long since the last M flare, what happened one solar rotation (27 days) earlier. The
daily forecaster already turns that into features and a small gradient-boosting model. This computes the same features
at the end of every network window (plus 30-minute to 2-hour terms and a rotation-slot count), fits the tree model on
the training period, and combines it with the network by a logistic regression on the two logits, fitted on validation.

Rules built in: the tree model never sees validation or test windows; the stacking weights and all thresholds come from
validation; a flare enters the history only 10 minutes after its peak (a peak is not known to be the peak until the flux
falls); a stack in which either model gets a non-positive weight is refused and the network stays alone. A test checks
that features do not change when data after the window is altered. The report compares network, tree model and stack on
validation, on the later part of GOES-16, and on GOES-18, with paired intervals. It is not wired into the live forecast:
adopt it only if the "stack minus network" row on `goes18_test` excludes zero in the right direction.

A caution from the test itself: on the synthetic Sun the stack put most weight on the weaker tree model on a small
validation set and then did *worse* than the network on the later period. The report exists to catch this; on real data
the validation set is two orders of magnitude larger, but read the stack-minus-network table, not the validation line.

## What the references say, and what they do not

- Ignoring time order inflates flare-model scores, and rebalancing choices should be measured, not assumed
  ([Ahmadzadeh et al., 2021](https://arxiv.org/abs/2103.07542v1)). The chronological split, the gap between periods and
  the calibration on validation windows come from this. A neighbouring paper lists improper data treatment, class
  weighting and re-sampling among the ways models look accurate while being weak
  ([Mistakes and Pitfalls in Training Flare Forecasting Models](https://essopenarchive.org/doi/full/10.1002/essoar.10508359.1)).
- A published LSTM flare predictor used the same layout as the chronological run (train 2010-2013, validate 2014,
  test 2015-2018) and found that selecting 14 to 22 of its 40 features scored better than using all of them
  ([Liu et al., 2019](https://arxiv.org/pdf/1905.07095)). That argues for small, selected feature sets in the tree model
  rather than every feature available.
- The review by [Shao et al.](https://arxiv.org/pdf/2511.20465) warns that random cross-validation can leak between
  correlated samples, says deep learning generally outperforms older machine learning, and cautions that results of
  different studies are not directly comparable because data, splits and preprocessing differ. It does not compare
  gradient boosting with deep learning head to head; nor do the other papers read for this. Whether trees beat, match or
  help the network on GOES flux is an open question for this project, and `gbm_6h.py` is how to answer it.
- NOAA's own daily M/X forecasts, verified over 1998-2024, do not outperform simple baselines (climatology,
  persistence, logistic regression and others) on key classification and probabilistic metrics, and are poorly calibrated
  with many false alarms ([Camporeale and Berger, 2025](https://arxiv.org/html/2508.01114v1)). That is consistent with
  the daily forecaster being level with NOAA through 2024. It also means "as good as NOAA" is a weaker bar than "better
  than a good simple baseline".
- The strongest systems in these papers use SDO/HMI magnetograms of active regions. FlareNet uses X-ray flux only, so
  published TSS values (for example 0.79 for >=M in the LSTM paper) are not a target it can be compared with: different
  inputs, a different forecast window (24 hours per active region against 6 hours for the whole disk) and different
  splits. The daily forecaster's sunspot-region features (the Solar Region Summary) added nothing measurable.

## Deliberately not done

- **Tuning against GOES-18.** That would turn the only unbiased test into a training signal.
- **A bigger or different architecture.** No source here shows one beating the current model at this data size, and a
  change would be a guess. The tuner already varies width and dropout; add architectures only after those results.
- **Magnetogram inputs.** The largest likely gain in the literature, but a new data source (JSOC/HMI) with its own
  pipeline and its own leakage traps; worth a separate project step, not a quick edit.
- **Touching the frozen daily model.** `models/daily_model_live.pkl` stays unchanged while the forward test runs.

## Order to run things

1. `compare_runs.py --paired` on the two existing runs (minutes): is the chronological model better, equal or worse?
2. `gbm_6h.py` (minutes): cheapest test of a different kind of information.
3. `tune_6h.py`, then the best config on GOES-18 (an hour or two).
4. The five-network config, on the best configuration from step 3 (copy the tuned values into it first).

Keep any change only if its paired interval on GOES-18 excludes zero in the right direction; otherwise the current model
stands, and that is a result worth recording too.

## First real results (run on GOES-16 training data, tested on GOES-18)

**Old model against the chronological, calibrated model** (`compare_runs.py --paired`, 12,668 windows, 393 M/X flares):
AUC +0.002 (95 % interval -0.004 to +0.007), Brier -0.111 (-0.123 to -0.100), TSS +0.004 (-0.053 to +0.057). The
calibrated model ranks windows as well as the old one, to within about 0.007 AUC, and its probabilities are far
better; at its default alert it warns 58 % of flares against 81 % and 60 % of its alerts are false against 75 %.
That is a different operating point on the same ranking, not a gain in skill. The pairing made the AUC interval about
six times narrower than the separate intervals (about +-0.035).

**Tree model and stack against the network alone** (`gbm_6h.py`, same GOES-18 windows, 1,646 followed by an M/X flare;
stack weights 0.34 on the network and 0.66 on the trees, fitted on 8,258 validation windows):

| GOES-18 | network | tree model | stack | stack minus network (95 % interval) |
|---|---|---|---|---|
| AUC | 0.810 | 0.828 | 0.828 | +0.0186 (0.0095 to 0.0284) |
| Brier | 0.0896 | 0.0875 | 0.0865 | -0.0031 (-0.0045 to -0.0017) |
| TSS | 0.446 | 0.471 | 0.479 | +0.0325 (0.0082 to 0.0576) |
| POD | 0.578 | 0.601 | 0.606 | +0.0286 (0.0024 to 0.0568) |
| FAR | 0.603 | 0.591 | 0.584 | -0.0191 (-0.0382 to -0.0018) |

All five intervals exclude zero in the stack's favour on the satellite neither model trained on. The tree model alone
also beats the network on AUC (+0.0187, 0.0064 to 0.0316). On the weaker check, the later part of GOES-16 (6,878
windows), the AUC difference is +0.0067 (-0.0016 to +0.0149), not established; Brier improves (-0.0040), POD improves
(+0.029) and FAR worsens (+0.012).

What this does and does not show:

- It is one test period on one satellite. The gain on GOES-18 (+0.019 AUC) is larger than on the later GOES-16 period
  (+0.007, not established), so "about +0.01 to +0.02 AUC" is the honest range until more data says otherwise.
- The tree model used 27-day rotation features and 168-hour windows. The live NOAA feed reaches back only 7 days, so a
  live forecast cannot compute them. `--live-compatible` refits without them; the stack goes into the live forecast only
  if the gain survives that.
- GOES-18 has now been used for one adopt-or-reject decision. Each further candidate compared on it raises the chance
  that one looks better by luck. Data after the config's end date (2026-09-30) is untouched and is the cleanest final
  check; it will take months of activity to mean anything, as the Sun has been quiet.

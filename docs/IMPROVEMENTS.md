# Model improvements: what was tried, what helped, what did not

Every candidate was compared with the original model on the **same synthetic worlds** (each seed is a
different generated lightcurve, split and initialisation), pairing results seed by seed, and was kept
only if it helped consistently. Tooling: `flarenet experiment` (see below). All numbers here are on
**synthetic data**; no real GOES or Aditya-L1 data has been used, so none of this proves a gain on the real Sun.

## Result

`configs/default.yaml` now turns on three changes. `configs/fast.yaml` keeps the original behaviour.

| | original (`fast.yaml`) | improved (`default.yaml`) |
|---|---|---|
| Nowcast macro-F1, 8-seed mean | 0.650 | **0.764** (+0.115 ± 0.042 s.e., better on 6 of 8 seeds) |
| Nowcast accuracy, 8-seed mean | 0.866 | **0.893** (+0.027 ± 0.009, 7 of 8 seeds) |
| Recall of M-class windows, 8-seed mean | 0.575 | **0.767** (+0.193 ± 0.084, 6 of 8 seeds) |
| Forecast AUC, 8-seed mean | 0.850 | 0.854 (+0.003 ± 0.002) |
| Forecast TSS, 8-seed mean | 0.573 | 0.572 (-0.001 ± 0.009) |
| Training time (2 CPU cores, one run) | 49 s | about 4 min |

So the improvement is in the **nowcast** (the class of the current minute, especially the rarer B, C
and M classes). The **6-hour forecast did not improve**: its changes are inside the noise.

Seed 42, the default seed shown in the README and the dashboard: nowcast macro-F1 0.374 → 0.748,
accuracy 0.771 → 0.876, forecast TSS 0.630 → 0.656, AUC 0.878 → 0.880. That jump is larger than the
typical gain because seed 42's original model happened to have a weak nowcast; quote the 8-seed
means above, not the seed-42 numbers.

Spread between seeds is large (original forecast TSS ranges about 0.45 to 0.80 across seeds), and on
the three seeds that were **not** used to choose this configuration (6, 7, 8) the nowcast gain was
+0.075 ± 0.057 macro-F1 (better on 2 of 3), which on its own is inconclusive. Gains were biggest where
the original nowcast was weakest (seed 1: 0.415 → 0.697; seed 5: 0.516 → 0.769) and the new
configuration was slightly worse on seeds 2 and 6.

## The three changes

| Setting | What it does |
|---|---|
| `data.train_stride_min: 30` | Training windows are cut every 30 min instead of 60: twice as many (overlapping) training windows. Validation and test keep the 60-minute grid, so scores stay comparable, and the block split still keeps neighbouring days together. |
| `training.nowcast_bias_tuning: true` | After training, per-class offsets are added to the nowcast logits, chosen on the **validation** set to maximise macro-F1. This corrects the bias towards QUIET/X that class imbalance causes. The offsets are saved in the checkpoint and added to the nowcast logits whenever the model predicts (evaluation, `predict`, `monitor`). |
| `training.ensemble_size: 3` | Three members trained with different seeds; probabilities are averaged. The saved model is the ensemble (3 x 122,087 = 366,261 parameters). |

Seed 42, per-class detection TSS for C / M / X (from the nowcast): 0.67 / 0.71 / 0.83 originally,
0.86 / 0.87 / 0.87 with these settings (single seed).

## Where the forecast stands, and why

`scripts/headroom_check.py` trains a plain gradient-boosted tree on simple summary statistics of each
window (last value, mean, max, min, spread, drift) on the same splits:

| mean over seeds 1-5 | gradient-boosted trees | FlareNet |
|---|---|---|
| Forecast AUC | 0.872 | 0.864 |
| Forecast TSS | 0.595 | 0.606 |

A simple tree model matches the network, so on this synthetic data **the forecast is limited by how
predictable the generator makes flares, not by network capacity**. Further architecture tuning is
unlikely to help the forecast here. Real forecast skill needs more information (real GOES / Aditya-L1
data, more instruments, extra features) and has to be measured on real data.

## Tried and not kept

Stage 1 compared single changes on seeds 1-3 (the stage was stopped early, so only 3 seeds were
complete for every variant). Paired differences to the original:

| Change | Result |
|---|---|
| `training.forecast_balance=auto` (50/50 forecast sampling) | forecast AUC +0.012, but nowcast macro-F1 -0.17 and worse calibration |
| `training.ema_decay=0.95` (weight averaging) | no gain; forecast TSS slightly lower (-0.005) |
| `training.select_metric=auc` | no clear gain; nowcast macro-F1 lower |
| `training.threshold_smooth=5` | no effect |
| `model.nowcast_skip=true`, `nowcast_pool=final_states` | forecast AUC +0.005, nowcast unchanged within noise |
| `training.nowcast_weight_power=1.0` (stronger class weights) | nowcast macro-F1 -0.13, accuracy -0.14 |
| `data.purge=true` (drop windows that overlap another split) | nowcast macro-F1 -0.14 |
| wider model (48 / 48 / 64) | no gain |
| 240-minute input window (stage 2, 5 seeds) | no gain; slower |
| stride 15 instead of 30 (stage 2, 5 seeds) | similar to stride 30 but slower to train |

`select_metric=tss_auc` produced identical results to the original in the stage-1 runs, so it was not
informative. These are single runs of each change with 3 to 5 seeds; a negative result means "no
consistent gain here", not "never useful".

## Reproduce

```bash
flarenet run -c configs/fast.yaml        # original behaviour, about 1 min on CPU
flarenet run                             # improved defaults, about 4 min on CPU

# the experiment runner: variants x seeds, paired comparison written to summary.md
flarenet experiment -e configs/experiments/stage3_final.yaml --out runs/stage3 --jobs 2
python scripts/headroom_check.py --seeds 1,2,3,4,5 --results runs/stage3/results.jsonl
```

`flarenet experiment` takes a YAML with `seeds`, a `baseline` variant name and `variants` (lists of
`--set`-style overrides). It writes `results.jsonl` (resumable), `results.csv` and `summary.md` with
means, standard deviations, and paired differences with standard errors and win counts. Use
`--jobs 1` on a GPU. The stage 1 / 2 / 3 definitions are in `configs/experiments/`.

## Caveats

* Everything above is synthetic data. The generator is far more flare-active than the real Sun.
* Configuration choices were made on seeds 1-5 and checked on seeds 6-8; that is a small fresh sample.
* The ensemble and finer windows cost about five times the training time.
* Offsets are tuned on the validation split of the same synthetic series; on a new instrument the
  class balance may differ, so re-tune (or disable `nowcast_bias_tuning`) when fine-tuning elsewhere.

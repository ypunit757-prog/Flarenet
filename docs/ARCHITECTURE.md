# Architecture

## Data contracts

Every instrument is reduced to the same objects, so nothing downstream knows which spacecraft
produced the data:

```
DataSource --fetch--> RawSeries --preprocess--> LightcurveBundle --features--> WindowSet --> model
 (plug-in)            native cadence             1-min grid + labels           (N, C, T) windows
                      sxr (+ hxr) channels       + flare database
```

* `RawSeries` (`core/schema.py`): per-channel `(t_unix, values)` at native cadence. `sxr` is
  required (the channel flares are labelled on); `hxr` is optional.
* `LightcurveBundle`: a regular 1-minute table `time_min, sxr, hxr, sxr_raw, hxr_raw, flare_class,
  flare_id, active_regime`, the start time `t0_unix`, the flare database and metadata (label kind,
  units, instrument names).
* `WindowSet`: `X (N, channels, window)`, nowcast class, forecast label (M/X onset within the
  horizon), window end time, missing fraction, and the source each window came from.

## Pipelines

| Package | Stages | Writes |
|---|---|---|
| `pipelines/ingestion` | `fetch` | `data/raw/<source>/` (parquet) |
| `pipelines/preprocessing` | `plan`, `resample`, `scale`, `label`, `bundle` | `data/processed/<source>/` (lightcurve, flare_db.csv) |
| `pipelines/features` | `windows`, `split` | `data/features/<source>/windows.npz` |
| `pipelines/training` | `fit`, `save` | `runs/<run>/checkpoint.pt`, `history.json` |
| `pipelines/evaluation` | `load_model`, `score` | in memory, consumed by reporting |
| `pipelines/reporting` | `metrics`, `markdown`, `figures` | `metrics_report.json`, `report.md`, PNGs, CSVs |
| `pipelines/inference` | `forecast` (offline) / `monitor` (live loop) | `predictions_<source>.csv/.png`, `alerts.csv` |

A `Pipeline` (`core/pipeline.py`) is an ordered list of `Stage` objects sharing a
`PipelineContext` (config, data store, run directory, a `state` dict). It logs and times each
stage, names the failing stage in the exception (`PipelineError`) and always writes
`manifest_<pipeline>.json` (status, per-stage seconds, config fingerprint). `orchestration/runner.py`
composes them into the user-facing workflows; each workflow is `ingestion -> preprocessing ->
features` followed by the pipeline(s) it is about.

## Caching

`core/artifacts.py` stores each stage's output next to a `meta.json` holding a fingerprint of the
inputs: source params, input-file size and mtime, labeler/scaling choice, window settings. A stage
re-runs only when the fingerprint changed or `--force` is given. `meta.json` is written last, so a
half-written directory counts as stale. Live feeds (`nasa_goes_swpc` without `json_path`) are never
cached.

## Roles and splits

Each source has a `role`:

* `train`: windowed, then split into train / val / test by shuffled contiguous multi-day blocks
  (`data.block_days`), per source. Several train sources are concatenated; `weight` sets how often
  each is sampled.
* `eval`: never trained on; scored on all its windows. This is how cross-instrument and
  cross-period results are produced.

## Extension points (registries in `core/registry.py`)

| Registry | Add one by | Example |
|---|---|---|
| `SOURCES` | subclass `DataSource`, decorate `@SOURCES.register("kind")` | [ADDING_A_SOURCE.md](ADDING_A_SOURCE.md) |
| `LABELERS` | subclass `Labeler` | `goes_events`, `relative_percentile`, `catalog` |
| `CATALOGS` | subclass `EventCatalog` | `donki`, `csv` |
| scaling policies | function in `pipelines/preprocessing/scaling.py` | `native`, `background_normalised` |
| `MODELS` | `models/` | `flarenet` |

Unknown config keys and unknown source parameters raise errors that list the valid ones.

# Using NASA / NOAA data

GOES X-ray Sensor (XRS) data is produced by NOAA's GOES-R series, built and launched with NASA, and
is published by NOAA (NCEI, SWPC) and mirrored by NASA (CDAWeb/SPDF). FlareNet reads it in three ways.
Labels from all of them are **calibrated GOES classes** (B/C/M/X from the 0.1-0.8 nm peak flux),
unlike the relative proxy used for Aditya-L1.

> **Not yet verified against the live servers.** These integrations were built in an environment
> that could not reach NOAA or NASA. They are tested against locally generated files in the real
> GOES-R NetCDF, SWPC JSON and DONKI layouts and against a local HTTP server that mimics the NCEI
> directory index, the SWPC feed and the DONKI endpoint. If NOAA changes a URL or file layout, the
> `base_url` options and `nasa_goes_file` let you adapt without code changes. Please run
> `flarenet ingest -c configs/nasa_goes_ncei.yaml` first and report any mismatch.

## 1. NCEI archive: `nasa_goes_ncei`

```yaml
- name: goes16_2024
  kind: nasa_goes_ncei
  role: train
  params:
    satellite: 16                 # 16, 17, 18, 19
    start: 2024-03-01             # inclusive, YYYY-MM-DD
    end: 2024-06-30
    local_dir: data/downloads/goes_xrs
    # offline: true               # use only files already in local_dir
    # max_missing_days: 3         # tolerated unavailable days
    # use_quality_flags: true     # flagged samples become NaN
```

Downloads `sci_xrsf-l2-avg1m_g<sat>_d<YYYYMMDD>_v*.nc` (1-minute averages, W/m2), picking the highest
version for each day, atomically and with retries; re-runs reuse `local_dir`. Needs
`pip install -e ".[netcdf]"`. The default `base_url` is
`https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{sat}/l2/data/xrsf-l2-avg1m_science/{yyyy}/{mm}/`;
change it if NOAA moves the archive.

## 2. Your own files: `nasa_goes_file`

No network needed. Format is guessed from the extension (`.nc`, `.json`, `.csv`, `.parquet`) or set with `format`.

```yaml
- name: goes_csv
  kind: nasa_goes_file
  role: train
  params:
    paths: ["/data/goes/*.csv"]      # globs allowed
    time_column: time_tag            # unix, unix_ms, mjd, jd or ISO text (auto-detected)
    sxr_column: xrsb                 # 0.1-0.8 nm flux, W/m2 (the channel flares are classified on)
    hxr_column: xrsa                 # 0.05-0.4 nm flux; null if you do not have it
    sxr_scale: 1.0                   # multiply fluxes (e.g. to convert legacy GOES scaling)
```

Good for data exported from SunPy/Fido, CDAWeb, NCEI CSV/NetCDF, or SWPC JSON dumps. Non-positive
values and sentinels become NaN.

## 3. Live feed: `nasa_goes_swpc`

Last 6 hours to 7 days from `https://services.swpc.noaa.gov/json/goes/<primary|secondary>/<product>.json`
(products `xrays-6-hour`, `xrays-1-day`, `xrays-3-day`, `xrays-7-day`). Use it as an `eval` source,
or for the live loop:

```bash
flarenet monitor -c configs/live_monitor.yaml --checkpoint runs/nasa_goes/checkpoint.pt
```

It appends `checked_utc, forecast_for_utc, p_mx_6h, alert, nowcast_class, threshold` to
`runs/live/alerts.csv` every `monitor.interval_s` seconds and skips ticks it cannot serve (network
error, less than `monitor.min_context_min` of history). It never trains. Use a checkpoint trained on
GOES data. Set `json_path` to replay a saved feed.

## DONKI flare catalog (NASA CCMC)

```bash
flarenet donki --start 2024-05-01 --end 2024-05-31 --out donki_may.csv     # DEMO_KEY by default
```

Get a free key at <https://api.nasa.gov> (`--api-key`, or `api_key: env` to read `$NASA_API_KEY`).
Two uses in a source config:

```yaml
labeler: catalog                      # official class and times as label truth
labeler_params: {catalog: donki, catalog_params: {api_key: env}}
# or keep the flux detector and report how well it agrees with DONKI:
labeler_params: {cross_check: {catalog: donki, catalog_params: {api_key: env}, tolerance_min: 15}}
```

The cross-check reports recall against the catalog, precision for C-class and above, and class
agreement; it lands in the processed bundle metadata and the logs.

## Caveats

* GOES channel 0.05-0.4 nm is used where Aditya-L1 has HEL1OS hard X-rays. They are different physics
  (hot-plasma soft X-rays vs 20+ keV photons), so expect a domain gap between instruments.
* The detector is NOAA-style (rise of 4 monotonic samples and 1.4x, end at half-decay) and works on
  1-minute averages; it will not reproduce NOAA's official event list exactly. Use the DONKI
  cross-check to measure the difference.
* Missing days and telemetry gaps become NaN; `data.max_missing_frac` drops windows with too many.
* GOES satellite changes (e.g. 16 to 18) alter calibration slightly; treat cross-satellite scores as a
  robustness check.
* Quote results from `report.md`, which lists the caveats that apply to the run.

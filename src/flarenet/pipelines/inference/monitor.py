"""Near-real-time forecasting loop over a live feed (default: NOAA SWPC GOES X-ray JSON).

Each iteration: fetch the latest feed -> resample / label in memory -> forecast the newest window
-> append one row to ``<run_dir>/<monitor.alert_log>``. Network failures are logged and retried
on the next tick; the loop never trains and never writes the data store.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from flarenet.core.config import Config, SourceSpec
from flarenet.core.utils import get_logger, resolve_device
from flarenet.pipelines.inference.predict import forecast_bundle
from flarenet.pipelines.preprocessing.pipeline import process_raw_series
from flarenet.pipelines.training.checkpoint import load_checkpoint
from flarenet.sources.base import SourceUnavailable, make_source

log = get_logger(__name__)
ALERT_COLUMNS = [
    "checked_utc",
    "forecast_for_utc",
    "p_mx_6h",
    "alert",
    "nowcast_class",
    "threshold",
]


def monitor_spec(cfg: Config) -> SourceSpec:
    m = cfg.monitor
    return SourceSpec(
        name="live", kind=m.source_kind, role="eval",
        params={"product": m.product, "satellite": m.satellite},
    )  # fmt: skip


def one_tick(cfg: Config, loaded, device, spec: SourceSpec | None = None) -> dict:
    """Fetch + forecast once. Raises SourceUnavailable / ValueError when no forecast is possible."""
    spec = spec or monitor_spec(cfg)
    raw = make_source(spec, cfg.seed).fetch()
    bundle = process_raw_series(spec, raw, cfg.seed)
    if bundle.n_minutes < cfg.monitor.min_context_min:
        raise ValueError(
            f"Only {bundle.n_minutes} min of history, need monitor.min_context_min="
            f"{cfg.monitor.min_context_min}; use a longer feed product (e.g. xrays-1-day)."
        )
    row = forecast_bundle(loaded, bundle, device, for_latest_only=True).iloc[0]
    return {
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "forecast_for_utc": row.time_utc,
        "p_mx_6h": round(float(row.p_mx_6h), 4),
        "alert": int(row.alert),
        "nowcast_class": row.nowcast_class,
        "threshold": loaded.threshold,
    }


def run_monitor(
    cfg: Config, checkpoint: str | Path, spec: SourceSpec | None = None, sleep=time.sleep
) -> list[dict]:
    device = resolve_device(cfg.training.device)
    loaded = load_checkpoint(checkpoint, device)
    log_path = cfg.run_dir / cfg.monitor.alert_log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    rows: list[dict] = []
    i = 0
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALERT_COLUMNS)
        if new_file:
            w.writeheader()
        while cfg.monitor.max_iterations is None or i < cfg.monitor.max_iterations:
            i += 1
            try:
                row = one_tick(cfg, loaded, device, spec)
            except (SourceUnavailable, ValueError) as e:
                log.warning("monitor tick %d skipped: %s", i, e)
            else:
                w.writerow(row)
                f.flush()
                rows.append(row)
                log.info("%s | P(M/X in %d min)=%.3f | %s | now=%s", row["forecast_for_utc"],
                         cfg.data.horizon_min, row["p_mx_6h"], "ALERT" if row["alert"] else "ok",
                         row["nowcast_class"])  # fmt: skip
            if cfg.monitor.max_iterations is None or i < cfg.monitor.max_iterations:
                sleep(cfg.monitor.interval_s)
    return rows

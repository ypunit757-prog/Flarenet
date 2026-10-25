"""Tomorrow's forecast from live data: the same features and model as the walk-forward test.

Run once a day shortly after 22:00 UTC (when NOAA issues its own forecast). The live model leaves out the
27-day rotation features because the live NOAA X-ray feed only reaches back 7 days.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.schema import LightcurveBundle
from flarenet.daily.features import day_features, events_table
from flarenet.daily.noaa_text import parse_flare_forecast, parse_srs


def load_daily_model(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)  # a file this project wrote; only load files you trust


def predict_tomorrow(
    pkg: dict, bundle: LightcurveBundle, srs_text: str | None = None, now_utc: datetime | None = None,
    issue_tolerance_h: float = 3.0,
) -> dict:  # fmt: skip
    """Probability that the target flare class starts during the next UTC day, from ``bundle`` (live flux)."""
    t0 = bundle.t0_unix
    n = len(bundle.df)
    t_last = t0 + (n - 1) * 60.0
    issue_h = int(pkg["issue_hour_utc"])
    now = now_utc or datetime.fromtimestamp(t_last, tz=timezone.utc)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t_issue = day0.timestamp() + issue_h * 3600.0
    if t_issue > t_last + 60.0:  # before today's issue time: use yesterday's
        t_issue -= 86400.0
    warn = []
    if abs(t_last - t_issue) > issue_tolerance_h * 3600.0:
        warn.append(f"the newest flux sample is {(t_last - t_issue) / 3600:+.1f} h from the {issue_h:02d}:00 UTC issue time; "
                    "the model was trained for that time, so treat this forecast as approximate")  # fmt: skip
    t_day = np.floor(t_issue / 86400.0) * 86400.0 + 86400.0
    flux = (bundle.df["sxr_raw"] if "sxr_raw" in bundle.df.columns else bundle.df["sxr"]).to_numpy(
        float
    )
    lf = np.log10(np.clip(np.where(np.isfinite(flux) & (flux > 0), flux, np.nan), 1e-11, None))
    regions, srs_day = None, date.fromtimestamp(0)
    if srs_text:
        issued, reg, ok = parse_srs(srs_text)
        if ok and issued is not None:
            srs_day = issued.date()
            regions = {srs_day: reg}
        else:
            warn.append(
                "the sunspot-region file could not be read; the forecast uses X-ray data only"
            )
    feats = day_features(
        t_issue, t_day, lf, t0, events_table(bundle), regions, srs_day, rot_ok=False
    )
    cols = pkg["columns"]
    if any(c.startswith("r_") for c in cols) and regions is None:
        warn.append(
            "no sunspot-region data was supplied; its features are missing and the forecast is less reliable"
        )
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
    p = float(pkg["model"].predict_proba(X)[0])
    return {
        "target_class": pkg["target"], "probability": round(p, 3), "alert_threshold": round(float(pkg["model"].threshold), 3),
        "alert": bool(p >= pkg["model"].threshold), "for_day": str(datetime.fromtimestamp(t_day, tz=timezone.utc).date()),
        "issued_utc": datetime.fromtimestamp(t_issue, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions_used": regions is not None, "warnings": warn,
        "trained_days": pkg.get("trained_days"), "trained_until": pkg.get("trained_until"),
    }  # fmt: skip


NOAA_LIVE_URLS = ("https://services.swpc.noaa.gov/text/3-day-solar-geomag-predictions.txt",)


def fetch_noaa_live(timeout: float = 20.0) -> dict | None:
    """NOAA's current flare probabilities for the next three UTC days, parsed with the same code that read
    the archive: {issued, first_day, m, x, source}. None if no live product could be read."""
    from flarenet.sources.base import SourceUnavailable
    from flarenet.sources.nasa.http import http_get

    for url in NOAA_LIVE_URLS:
        try:
            text = http_get(url, timeout=timeout, retries=2).text
        except SourceUnavailable:
            continue
        r = parse_flare_forecast(text)
        if r is not None:
            return {**r, "source": url}
    return None


def check_noaa_live(timeout: float = 20.0) -> list[str]:
    """Human-readable report of what each live NOAA URL returns (for debugging a wrong URL or format)."""
    from flarenet.sources.base import SourceUnavailable
    from flarenet.sources.nasa.http import http_get

    lines = []
    for url in NOAA_LIVE_URLS:
        try:
            text = http_get(url, timeout=timeout, retries=1).text
        except SourceUnavailable as e:
            lines.append(f"{url}: not reachable ({e})")
            continue
        r = parse_flare_forecast(text)
        if r is None:
            head = " | ".join(text.splitlines()[:6])
            lines.append(f"{url}: reachable but no flare probabilities found. First lines: {head}")
        else:
            lines.append(
                f"{url}: OK, issued {r['issued']}, first day {r['first_day']}, M {r['m']}, X {r['x']}"
            )
    return lines


def noaa_prob_for_day(noaa: dict | None, for_day: date, target: str = "M") -> float:
    """NOAA's probability for exactly ``for_day`` (NaN if its forecast does not cover that day)."""
    if not noaa:
        return float("nan")
    k = (for_day - noaa["first_day"]).days
    vals = noaa["m" if target.upper() == "M" else "x"]
    return float(vals[k]) if 0 <= k < len(vals) else float("nan")


def model_id(path: str | Path) -> str:
    """Short fingerprint of a model file, so a forward test can show the model never changed."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


def fetch_live_srs(timeout: float = 20.0) -> str | None:
    from flarenet.sources.base import SourceUnavailable
    from flarenet.sources.nasa.http import http_get

    try:
        return http_get(
            "https://services.swpc.noaa.gov/text/srs.txt", timeout=timeout, retries=2
        ).text
    except SourceUnavailable:
        return None

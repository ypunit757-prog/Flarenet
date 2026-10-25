"""NOAA text products from the NGDC archive: Solar Region Summary (SRS) and the forecast report (RSGA).

SRS  (issued 00:30 UTC daily): the numbered sunspot regions with area, McIntosh class, spot count and magnetic
     type, valid for the previous 24:00 UTC.
RSGA (issued 22:00 UTC daily): NOAA's official forecast, including "Event probabilities" for Class M and Class X
     for each of the next three UTC days. These are the numbers a FlareNet forecast is compared against.

Archive layout (checked against the live index): <base>/<YYYY>/<MM>/<YYYYMMDD>SRS.txt and ...RSGA.txt.
Files are cached on disk and never fetched twice. A day NOAA did not publish (HTTP 404) is remembered so a
re-run does not ask for it again. The parsers return None / flag ``ok=False`` for text they do not understand
instead of guessing, and the callers count those so a format change is visible.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.core.utils import get_logger
from flarenet.sources.base import SourceUnavailable
from flarenet.sources.nasa.http import http_get

log = get_logger(__name__)

BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports"
URLS = {
    "srs": BASE + "/solar_region_summaries/{y}/{m:02d}/{y}{m:02d}{d:02d}SRS.txt",
    "rsga": BASE + "/reports_solar_geophysical_activity/{y}/{m:02d}/{y}{m:02d}{d:02d}RSGA.txt",
}
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}  # fmt: skip

REGION_COLUMNS = ["number", "lat", "lon", "carr_lon", "area", "z", "p", "c", "nn", "mag"]


# ----- parsing ---------------------------------------------------------------------------------

_ISSUED = re.compile(r":Issued:\s*(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2})(\d{2})\s+UTC")
_ROW = re.compile(
    r"^\s*(\d{4,5})\s+([NS])(\d{1,2})([EW])(\d{1,3})\s+(\d{1,3})\s+(\d{1,4})\s+([A-Za-z]{3})"
    r"\s+(\d{1,3})\s+(\d{1,3})\s*([A-Za-z][A-Za-z-]*)?\s*$"
)


def parse_issued(text: str) -> pd.Timestamp | None:
    m = _ISSUED.search(text)
    if not m or m.group(2).title() not in MONTHS:
        return None
    y, mo, d, hh, mm = (
        int(m.group(1)),
        MONTHS[m.group(2).title()],
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
    )
    try:
        return pd.Timestamp(datetime(y, mo, d, hh, mm, tzinfo=timezone.utc))
    except ValueError:
        return None


def parse_srs(text: str) -> tuple[pd.Timestamp | None, pd.DataFrame, bool]:
    """Return (issued, regions, ok). ``ok`` is False when the sunspot section was not found."""
    issued = parse_issued(text)
    rows, in_sec, found = [], False, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("I.") and "Regions with Sunspots" in s:
            in_sec, found = True, True
            continue
        if in_sec and (s.startswith("IA.") or s.startswith("II.") or s.startswith("III.")):
            break
        if not in_sec:
            continue
        m = _ROW.match(line)
        if not m:
            continue
        ns, lat, ew, lon = m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        rows.append(
            {
                "number": int(m.group(1)),
                "lat": lat if ns == "N" else -lat,
                "lon": lon
                if ew == "W"
                else -lon,  # degrees from the central meridian, west positive
                "carr_lon": int(m.group(6)),
                "area": int(m.group(7)),  # millionths of the solar hemisphere
                "z": m.group(8)[0].upper(),
                "p": m.group(8)[1].lower(),
                "c": m.group(8)[2].lower(),
                "nn": int(m.group(10)),  # number of spots
                "mag": (m.group(11) or "").strip(),
            }
        )
    return issued, pd.DataFrame(rows, columns=REGION_COLUMNS), found


_PROB = re.compile(r"Class\s+{c}\s+(\d{{1,3}})\s*/\s*(\d{{1,3}})\s*/\s*(\d{{1,3}})")
_RANGE = re.compile(
    r"Event probabilities\s+(\d{1,2})\s+([A-Za-z]{3})\s*-\s*(\d{1,2})\s+([A-Za-z]{3})"
)


def parse_rsga(text: str) -> dict | None:
    """NOAA's flare forecast: {issued, first_day (date), m: [p1, p2, p3], x: [...]} in 0..1, or None."""
    issued = parse_issued(text)
    h = _RANGE.search(text)
    if issued is None or h is None or h.group(2).title() not in MONTHS:
        return None
    tail = text[h.end() :]
    pm = re.search(_PROB.pattern.format(c="M"), tail)
    px = re.search(_PROB.pattern.format(c="X"), tail)
    if pm is None:
        return None
    day, mon = int(h.group(1)), MONTHS[h.group(2).title()]
    year = issued.year + (
        1 if mon < issued.month - 6 else 0
    )  # a December report forecasting January
    try:
        first = date(year, mon, day)
    except ValueError:
        return None
    vals = lambda m: [int(v) / 100.0 for v in m.groups()] if m else [np.nan] * 3  # noqa: E731
    return {"issued": issued, "first_day": first, "m": vals(pm), "x": vals(px)}


_DAYPRE_DATES = re.compile(r":Prediction_dates:\s+(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})")
_DAYPRE_PROB = r"Class_{c}\s+(\d{{1,3}})\s+(\d{{1,3}})\s+(\d{{1,3}})"


def parse_daypre(text: str) -> dict | None:
    """The live '3-day Space Weather Predictions' product (daypre): whole-disk flare probabilities for the next
    three UTC days, in the same shape as ``parse_rsga``. None if the section is absent."""
    issued = parse_issued(text)
    h = _DAYPRE_DATES.search(text)
    if issued is None or h is None or h.group(2).title() not in MONTHS:
        return None
    tail = text[text.find(":Whole_Disk_Flare_Prob:") :] if ":Whole_Disk_Flare_Prob:" in text else ""
    pm = re.search(_DAYPRE_PROB.format(c="M"), tail)
    px = re.search(_DAYPRE_PROB.format(c="X"), tail)
    if pm is None:
        return None
    try:
        first = date(int(h.group(1)), MONTHS[h.group(2).title()], int(h.group(3)))
    except ValueError:
        return None
    vals = lambda m: [int(v) / 100.0 for v in m.groups()] if m else [np.nan] * 3  # noqa: E731
    return {"issued": issued, "first_day": first, "m": vals(pm), "x": vals(px)}


def parse_flare_forecast(text: str) -> dict | None:
    """Either NOAA product (archive RSGA or live daypre): {issued, first_day, m, x} in 0..1, or None."""
    return parse_rsga(text) or parse_daypre(text)


# ----- downloading -----------------------------------------------------------------------------


@dataclass
class FetchStats:
    requested: int = 0
    cached: int = 0
    downloaded: int = 0
    missing: int = 0  # NOAA has no such file (404)
    failed: int = 0  # network trouble; a re-run retries these


def _path(cache: Path, kind: str, d: date) -> Path:
    return cache / kind / f"{d.year}" / f"{d:%Y%m%d}.txt"


def fetch_range(
    kind: str, start: date, end: date, cache_dir: str | Path, workers: int = 4,
    retries: int = 3, timeout_s: float = 30.0,
) -> tuple[dict[date, Path], FetchStats]:  # fmt: skip
    """Download ``kind`` ('srs' or 'rsga') for every day in [start, end]; return {date: file} and counts."""
    if kind not in URLS:
        raise ValueError(f"kind must be one of {sorted(URLS)}")
    cache = Path(cache_dir)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    stats = FetchStats(requested=len(days))
    out: dict[date, Path] = {}

    def one(d: date) -> tuple[date, str]:
        p = _path(cache, kind, d)
        if p.exists() and p.stat().st_size > 0:
            return d, "cached"
        miss = p.with_suffix(".missing")
        if miss.exists():
            return d, "missing"
        try:
            r = http_get(
                URLS[kind].format(y=d.year, m=d.month, d=d.day), timeout=timeout_s, retries=retries
            )
        except SourceUnavailable as e:
            if "HTTP 404" in str(e):
                p.parent.mkdir(parents=True, exist_ok=True)
                miss.write_text("404\n")
                return d, "missing"
            return d, "failed"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(p)
        return d, "downloaded"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for d, res in ex.map(one, days):
            setattr(stats, res, getattr(stats, res) + 1)
            if res in ("cached", "downloaded"):
                out[d] = _path(cache, kind, d)
            done += 1
            if done % 250 == 0 or done == len(days):
                log.info("[%s] %d/%d days (%d downloaded, %d cached, %d missing, %d failed)",
                         kind, done, len(days), stats.downloaded, stats.cached, stats.missing, stats.failed)  # fmt: skip
    if stats.failed > 0.2 * max(1, len(days)):
        raise SourceUnavailable(
            f"[{kind}] {stats.failed} of {len(days)} days could not be downloaded (network). Re-run the same "
            "command: finished days are cached. If this machine blocks the NGDC site, download the files "
            f"manually into {cache / kind} as <YYYY>/<YYYYMMDD>.txt."
        )
    return out, stats


def load_srs(files: dict[date, Path]) -> tuple[dict[date, pd.DataFrame], int]:
    """Parse cached SRS files -> ({file date: regions}, number of files that were not understood)."""
    out, bad = {}, 0
    for d, p in files.items():
        _, reg, ok = parse_srs(p.read_text(errors="replace"))
        if ok:
            out[d] = reg
        else:
            bad += 1
    if bad:
        log.warning(
            "SRS: %d of %d files had no recognisable sunspot section and were skipped",
            bad,
            len(files),
        )
    return out, bad


def load_rsga(files: dict[date, Path]) -> tuple[pd.DataFrame, int]:
    """Parse cached RSGA files -> DataFrame indexed by the first forecast day (UTC date) with
    columns noaa_m1, noaa_x1 (day 1) and noaa_m2/3, noaa_x2/3, plus the file date."""
    rows, bad = [], 0
    for d, p in files.items():
        r = parse_rsga(p.read_text(errors="replace"))
        if r is None or r["first_day"] != d + timedelta(days=1):
            bad += 1
            continue
        rows.append({"date": pd.Timestamp(r["first_day"], tz="UTC"), "issued": r["issued"],
                     **{f"noaa_m{i + 1}": r["m"][i] for i in range(3)},
                     **{f"noaa_x{i + 1}": r["x"][i] for i in range(3)}})  # fmt: skip
    if bad:
        log.warning(
            "RSGA: %d of %d files had no readable event probabilities and were skipped",
            bad,
            len(files),
        )
    if not rows:
        return pd.DataFrame(columns=["noaa_m1", "noaa_x1"]), bad
    df = pd.DataFrame(rows).drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df, bad

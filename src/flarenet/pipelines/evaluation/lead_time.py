"""Alert lead-time analysis for M/X flares."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.constants import M_CLASS_IDX
from flarenet.core.utils import get_logger

log = get_logger(__name__)

LEAD_COLUMNS = [
    "dataset",
    "event_id",
    "class",
    "onset_min",
    "first_alert_min",
    "lead_time_min",
    "status",
]


def compute_lead_times(
    flare_db: pd.DataFrame,
    forecast_times,
    forecast_probs,
    threshold: float,
    max_lead_min: float = 360.0,
    label: str = "",
) -> pd.DataFrame:
    """Match each M/X flare to the earliest alert that fired shortly before it.

    For every M/X event, only forecasts issued in ``[onset - max_lead_min, onset)`` are considered,
    because a forecast made at time t only claims to predict flares within ``max_lead_min`` after t.
    Using the whole earlier record would credit alerts hours or days before the flare as "lead time".

    Status per event
    ----------------
    HIT           : an alert fired before onset (within the horizon); lead = onset - first alert (> 0)
    LATE          : no early alert, but one fired during the event; lead <= 0
    MISS          : no alert before or during the event
    Events with no forecast issued inside the look-back window have no coverage
    (e.g. they fall in the train/val part of a block split) and are excluded, not counted as misses.
    """
    ft = np.asarray(forecast_times, dtype=float)
    fp = np.asarray(forecast_probs, dtype=float)
    mx = flare_db[flare_db["class_idx"] >= M_CLASS_IDX] if len(flare_db) > 0 else pd.DataFrame()
    if len(mx) == 0:
        log.info("[%s] No M/X events in flare DB, lead time not applicable.", label)
        return pd.DataFrame(columns=LEAD_COLUMNS)

    alert = fp >= threshold
    rows, n_uncovered = [], 0
    for _, ev in mx.iterrows():
        onset = float(ev["onset_min"])
        end = float(ev["end_min"]) if "end_min" in ev else onset + float(ev.get("duration_min", 60))

        window = (ft >= onset - max_lead_min) & (ft < onset)
        during = (ft >= onset) & (ft <= end)
        if not window.any() and not during.any():
            n_uncovered += 1
            continue

        if (alert & window).any():
            first = float(ft[alert & window][0])
            status = "HIT"
        elif (alert & during).any():
            first = float(ft[alert & during][0])
            status = "LATE"
        else:
            first, status = float("nan"), "MISS"
        rows.append(
            {
                "dataset": label,
                "event_id": int(ev["event_id"]),
                "class": ev["class"],
                "onset_min": onset,
                "first_alert_min": first,
                "lead_time_min": onset - first if status != "MISS" else float("nan"),
                "status": status,
            }
        )

    out = pd.DataFrame(rows, columns=LEAD_COLUMNS)
    if n_uncovered:
        log.info(
            "[%s] %d M/X events had no forecast coverage and were excluded", label, n_uncovered
        )
    hits = out[out.status == "HIT"]
    if len(out):
        msg = (
            f"[{label}] M/X events scored: {len(out)} | HIT={len(hits)} "
            f"LATE={(out.status == 'LATE').sum()} MISS={(out.status == 'MISS').sum()}"
        )
        if len(hits):
            msg += (
                f" | lead min/median/mean/max = {hits.lead_time_min.min():.0f}/"
                f"{hits.lead_time_min.median():.0f}/{hits.lead_time_min.mean():.0f}/"
                f"{hits.lead_time_min.max():.0f} min"
            )
        log.info(msg)
    return out


def summarise_lead_times(lead: pd.DataFrame) -> dict:
    """Compact JSON-friendly summary of a lead-time table."""
    if len(lead) == 0:
        return {
            "n_events": 0,
            "n_hits": 0,
            "n_late": 0,
            "n_miss": 0,
            "hit_rate": None,
            "mean_min": None,
            "median_min": None,
        }
    hits = lead[lead.status == "HIT"]
    return {
        "n_events": int(len(lead)),
        "n_hits": int(len(hits)),
        "n_late": int((lead.status == "LATE").sum()),
        "n_miss": int((lead.status == "MISS").sum()),
        "hit_rate": float(len(hits) / len(lead)),
        "mean_min": float(hits.lead_time_min.mean()) if len(hits) else None,
        "median_min": float(hits.lead_time_min.median()) if len(hits) else None,
    }

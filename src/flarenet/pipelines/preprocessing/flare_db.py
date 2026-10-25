"""One-row-per-event flare database built from a labelled lightcurve."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.time import Time

from flarenet.core.constants import CLASS_NAMES

DB_COLUMNS = [
    "event_id",
    "class",
    "class_idx",
    "onset_min",
    "peak_min",
    "end_min",
    "duration_min",
    "rise_min",
    "decay_min",
    "peak_sxr",
    "dataset",
]


def build_flare_db(
    df: pd.DataFrame, label: str = "synthetic", t0_unix: float | None = None
) -> pd.DataFrame:
    """Summarise every contiguous flare event (onset, peak, end, class, rise/decay times).

    Parameters
    ----------
    df : lightcurve with ``flare_id``, ``flare_class``, ``time_min`` and ``sxr_raw`` (or ``sxr``).
    label : dataset name stored in the ``dataset`` column.
    t0_unix : Unix time of ``time_min == 0``; when given, UTC columns are added.
    """
    ids = df["flare_id"].to_numpy()
    classes = df["flare_class"].to_numpy()
    sxr = (df["sxr_raw"] if "sxr_raw" in df.columns else df["sxr"]).to_numpy()
    minutes = df["time_min"].to_numpy()

    rows = []
    for event_id in np.unique(ids):
        if event_id < 0:
            continue
        mask = ids == event_id
        ev_cls, ev_t, ev_sxr = classes[mask], minutes[mask], sxr[mask]
        dominant = int(ev_cls.max())
        if dominant == 0:
            continue
        valid = ~np.isnan(ev_sxr)
        if not valid.any():
            continue
        peak_loc = np.where(valid)[0][np.nanargmax(ev_sxr[valid])]
        onset, peak, end = float(ev_t[0]), float(ev_t[peak_loc]), float(ev_t[-1])
        row = {
            "event_id": int(event_id),
            "class": CLASS_NAMES[dominant],
            "class_idx": dominant,
            "onset_min": round(onset, 2),
            "peak_min": round(peak, 2),
            "end_min": round(end, 2),
            "duration_min": round(end - onset, 2),
            "rise_min": round(peak - onset, 2),
            "decay_min": round(end - peak, 2),
            "peak_sxr": float(ev_sxr[peak_loc]),
            "dataset": label,
        }
        if t0_unix is not None:
            row["onset_utc"] = Time(t0_unix + onset * 60, format="unix").iso
            row["peak_utc"] = Time(t0_unix + peak * 60, format="unix").iso
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=DB_COLUMNS)
    return pd.DataFrame(rows).reset_index(drop=True)


def format_flare_db(fdb: pd.DataFrame, label: str = "") -> str:
    """Human-readable table of the flare database for logs."""
    if len(fdb) == 0:
        return f"[{label}] No flare events detected."
    counts = fdb["class"].value_counts().to_dict()
    cols = (
        ["event_id", "class", "onset_utc", "peak_utc", "duration_min", "peak_sxr"]
        if "onset_utc" in fdb.columns
        else ["event_id", "class", "onset_min", "peak_min", "duration_min", "peak_sxr"]
    )
    head = f"Flare database [{label}]: {len(fdb)} events, classes {counts}"
    return head + "\n" + fdb[cols].to_string(index=False)

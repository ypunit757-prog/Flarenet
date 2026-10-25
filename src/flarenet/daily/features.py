"""One row per UTC day: features known at the issue time, and whether an M/X flare starts the next day.

Row for target day T:  features use only information available at T-1 at ``issue_hour`` UTC (default 22:00,
when NOAA issues its own forecast): the X-ray flux before that moment, flares whose peak is already past, the
flares that occurred one solar rotation (27 days) earlier, and the Solar Region Summary issued at 00:30 UTC on
day T-1. The label is "an event of at least class M begins during UTC day T".

Column prefixes (used for ablations): f_ flux level, h_ flare history, rot_ rotation recurrence, r_ sunspot regions.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from flarenet.core.schema import LightcurveBundle

FLUX_WINDOWS_H = (3, 6, 12, 24, 48, 72, 168)
COUNT_WINDOWS_H = (24, 72, 168)
QUIET_LOG_FLUX = -9.0  # "no flare in the window" for peak-flux features (below A-class)
SINCE_CAP_H = 720.0
ROT_DAYS = 27
Z_ORD = {
    "A": 0,
    "B": 1,
    "H": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
}  # modified Zurich class, rough complexity order
P_ORD = {"x": 0, "r": 1, "s": 2, "a": 3, "h": 4, "k": 5}  # penumbra
C_ORD = {"x": 0, "o": 1, "i": 2, "c": 3}  # compactness


def _nan_if_empty(fn, a, default=np.nan):
    a = a[np.isfinite(a)]
    return float(fn(a)) if len(a) else default


def region_features(reg: pd.DataFrame | None, prev: pd.DataFrame | None) -> dict[str, float]:
    """Summaries of one Solar Region Summary (``prev`` = the previous day's, for growth)."""
    keys = ["r_n", "r_area_sum", "r_area_max", "r_nn_sum", "r_nn_max", "r_n_beta", "r_n_gamma", "r_n_delta",
            "r_n_complex", "r_area_complex", "r_comp_max", "r_pen_max", "r_cmp_max", "r_area_central",
            "r_n_east", "r_n_new", "r_area_growth", "r_nn_growth"]  # fmt: skip
    if reg is None:
        return {k: np.nan for k in keys}
    if len(reg) == 0:
        out = {k: 0.0 for k in keys}
        out["r_comp_max"] = out["r_pen_max"] = out["r_cmp_max"] = -1.0
    else:
        mag = reg["mag"].str.lower()
        cx = reg["z"].isin(["E", "F"]) | mag.str.contains("gamma") | mag.str.contains("delta")
        out = {
            "r_n": float(len(reg)),
            "r_area_sum": float(reg["area"].sum()),
            "r_area_max": float(reg["area"].max()),
            "r_nn_sum": float(reg["nn"].sum()),
            "r_nn_max": float(reg["nn"].max()),
            "r_n_beta": float(mag.str.contains("beta").sum()),
            "r_n_gamma": float(mag.str.contains("gamma").sum()),
            "r_n_delta": float(mag.str.contains("delta").sum()),
            "r_n_complex": float(cx.sum()),
            "r_area_complex": float(reg.loc[cx, "area"].sum()),
            "r_comp_max": float(reg["z"].map(Z_ORD).fillna(0).max()),
            "r_pen_max": float(reg["p"].map(P_ORD).fillna(0).max()),
            "r_cmp_max": float(reg["c"].map(C_ORD).fillna(0).max()),
            "r_area_central": float(reg.loc[reg["lon"].abs() <= 45, "area"].sum()),
            "r_n_east": float((reg["lon"] < 0).sum()),
        }
        if prev is None:
            out["r_n_new"] = out["r_area_growth"] = out["r_nn_growth"] = np.nan
        else:
            out["r_n_new"] = float((~reg["number"].isin(prev["number"])).sum())
            out["r_area_growth"] = out["r_area_sum"] - float(prev["area"].sum())
            out["r_nn_growth"] = out["r_nn_sum"] - float(prev["nn"].sum())
        return out
    if prev is None:
        out["r_n_new"] = out["r_area_growth"] = out["r_nn_growth"] = np.nan
    else:
        out["r_area_growth"] = -float(prev["area"].sum())
        out["r_nn_growth"] = -float(prev["nn"].sum())
    return out


def events_table(bundle: LightcurveBundle) -> pd.DataFrame:
    """Flare events with absolute UTC seconds for onset and peak, class index (2=C, 3=M, 4=X), log peak flux."""
    fdb = bundle.flare_db
    if fdb is None or len(fdb) == 0:
        return pd.DataFrame({"onset": [], "peak": [], "cls": [], "lpk": []}, dtype=float)
    return pd.DataFrame(
        {
            "onset": bundle.t0_unix + fdb["onset_min"].to_numpy(float) * 60.0,
            "peak": bundle.t0_unix + fdb["peak_min"].to_numpy(float) * 60.0,
            "cls": fdb["class_idx"].to_numpy(int),
            "lpk": np.log10(np.clip(fdb["peak_sxr"].to_numpy(float), 1e-11, None)),
        }
    ).sort_values("peak", ignore_index=True)


def day_features(
    t_issue: float, t_day: float, lf: np.ndarray, t0: float, ev: pd.DataFrame,
    regions: dict[date, pd.DataFrame] | None, srs_day: date, rot_ok: bool,
) -> dict[str, float]:  # fmt: skip
    """All features for one issue time. ``lf`` is log10 flux on the 1-minute grid starting at ``t0``."""
    i = int(round((t_issue - t0) / 60.0))
    f: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last = lf[max(0, i - 9) : i + 1]
        f["f_cur"] = _nan_if_empty(np.median, last)
        for h in FLUX_WINDOWS_H:
            f[f"f_max_{h}h"] = _nan_if_empty(np.max, lf[max(0, i - h * 60 + 1) : i + 1])
        w6, w24, w72 = (lf[max(0, i - h * 60 + 1) : i + 1] for h in (6, 24, 72))
        f["f_mean_6h"] = _nan_if_empty(np.mean, w6)
        f["f_mean_24h"] = _nan_if_empty(np.mean, w24)
        f["f_std_24h"] = _nan_if_empty(np.std, w24)
        f["f_q10_72h"] = _nan_if_empty(lambda a: np.quantile(a, 0.10), w72)
        f["f_q90_24h"] = _nan_if_empty(lambda a: np.quantile(a, 0.90), w24)
        f["f_linmean_24h"] = _nan_if_empty(lambda a: np.log10(np.mean(10.0**a)), w24)
        f["f_linmean_72h"] = _nan_if_empty(lambda a: np.log10(np.mean(10.0**a)), w72)
        recent, before = lf[max(0, i - 119) : i + 1], lf[max(0, i - 359) : max(0, i - 119)]
        f["f_trend_6h"] = _nan_if_empty(np.mean, recent) - _nan_if_empty(np.mean, before)
        f["f_cov_24h"] = float(np.isfinite(w24).mean()) if len(w24) else np.nan

    # flare history: only events whose peak has already happened
    pk = ev["peak"].to_numpy()
    known = ev[pk <= t_issue]
    for h in COUNT_WINDOWS_H:
        win = known[known["peak"] > t_issue - h * 3600.0]
        c = win["cls"].to_numpy()
        f[f"h_nC_{h}"] = float((c >= 2).sum())
        f[f"h_nM_{h}"] = float((c >= 3).sum())
        f[f"h_nX_{h}"] = float((c >= 4).sum())
        f[f"h_peak_{h}"] = float(win["lpk"].max()) if len(win) else QUIET_LOG_FLUX
    for name, k in (("C", 2), ("M", 3), ("X", 4)):
        sel = known[known["cls"] >= k]
        f[f"h_since_{name}"] = (
            min(SINCE_CAP_H, (t_issue - float(sel["peak"].iloc[-1])) / 3600.0)
            if len(sel)
            else SINCE_CAP_H
        )

    # one solar rotation earlier: events starting on the equivalent target day (+-1 day for the wider window)
    if rot_ok:
        d0 = t_day - ROT_DAYS * 86400.0  # start of the same UTC day 27 days earlier
        on = ev["onset"].to_numpy()
        cls = ev["cls"].to_numpy()
        lpk = ev["lpk"].to_numpy()
        same = (on >= d0) & (on < d0 + 86400.0)
        wide = (on >= d0 - 86400.0) & (on < d0 + 2 * 86400.0)
        f["rot_nC_27"] = float(((cls >= 2) & same).sum())
        f["rot_nM_27"] = float(((cls >= 3) & same).sum())
        f["rot_nM_27_wide"] = float(((cls >= 3) & wide).sum())
        f["rot_peak_27_wide"] = float(lpk[wide].max()) if wide.any() else QUIET_LOG_FLUX
    else:
        f.update(
            {
                "rot_nC_27": np.nan,
                "rot_nM_27": np.nan,
                "rot_nM_27_wide": np.nan,
                "rot_peak_27_wide": np.nan,
            }
        )

    if regions is not None:
        f.update(region_features(regions.get(srs_day), regions.get(srs_day - timedelta(days=1))))
    return f


def build_day_table(
    bundle: LightcurveBundle, regions: dict[date, pd.DataFrame] | None = None, issue_hour: int = 22,
    min_coverage: float = 0.6,
) -> pd.DataFrame:  # fmt: skip
    """Feature + label table for every UTC day of ``bundle`` that is fully covered. Index = target date (UTC)."""
    df = bundle.df
    t0 = bundle.t0_unix
    n = len(df)
    flux = (df["sxr_raw"] if "sxr_raw" in df.columns else df["sxr"]).to_numpy(float)
    lf = np.log10(np.clip(np.where(np.isfinite(flux) & (flux > 0), flux, np.nan), 1e-11, None))
    ev = events_table(bundle)
    t_end = t0 + n * 60.0
    first_day = int(
        np.ceil((t0 + 86400.0) / 86400.0)
    )  # first target day whose issue time is inside the data
    last_day = int(np.floor(t_end / 86400.0)) - 1  # last target day fully inside the data
    rows, idx = [], []
    for dnum in range(first_day, last_day + 1):
        t_day = dnum * 86400.0
        t_issue = t_day - 86400.0 + issue_hour * 3600.0
        if t_issue - t0 < 3600.0:
            continue
        i = int(round((t_issue - t0) / 60.0))
        cov = np.isfinite(lf[max(0, i - 1439) : i + 1]).mean()
        if cov < min_coverage:
            continue
        srs_day = date.fromordinal(date(1970, 1, 1).toordinal() + dnum - 1)
        f = day_features(
            t_issue,
            t_day,
            lf,
            t0,
            ev,
            regions,
            srs_day,
            rot_ok=(t_day - ROT_DAYS * 86400.0 - 86400.0 >= t0),
        )
        sel = (ev["onset"] >= t_day) & (ev["onset"] < t_day + 86400.0)
        f["y_m"] = float(((ev["cls"] >= 3) & sel).any())
        f["y_x"] = float(((ev["cls"] >= 4) & sel).any())
        # does the target day itself have usable data? (a flare cannot be seen in a data gap)
        j0, j1 = int(round((t_day - t0) / 60.0)), int(round((t_day + 86400.0 - t0) / 60.0))
        f["target_cov"] = float(np.isfinite(lf[j0:j1]).mean())
        rows.append(f)
        idx.append(pd.Timestamp(date.fromordinal(date(1970, 1, 1).toordinal() + dnum), tz="UTC"))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))
    out["source"] = bundle.name
    return out


def feature_columns(
    table: pd.DataFrame, groups: tuple[str, ...] = ("f_", "h_", "rot_", "r_")
) -> list[str]:
    cols = [c for c in table.columns if c.startswith(groups)]
    # a column that is NaN everywhere (e.g. no SRS downloaded) carries no information
    return [c for c in cols if table[c].notna().any()]

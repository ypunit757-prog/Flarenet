"""NOAA-style flare detection on a calibrated soft X-ray (0.1-0.8 nm, W/m^2) lightcurve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flarenet.core.goes import flux_to_class_idx
from flarenet.core.registry import LABELERS
from flarenet.pipelines.preprocessing.labelers.base import Labeler, LabelInput, events_to_arrays


def detect_goes_events(
    flux: np.ndarray,
    rise_min: int = 4,
    rise_factor: float = 1.4,
    end_fraction: float = 0.5,
    min_peak_flux: float = 1e-7,
    max_duration_min: int = 720,
    max_gap_min: int = 30,
) -> list[dict]:
    """Return events ``{start, peak, end, peak_flux}`` (minute indices, non-overlapping).

    Rules (the NOAA SWPC event definition, simplified):
      start : first of ``rise_min`` samples that increase monotonically, with the last at least
              ``rise_factor`` x the first;
      peak  : maximum before the flux decays;
      end   : flux decays to ``end_fraction`` of the way from the peak back to the start flux.
    Events whose peak is below ``min_peak_flux`` are discarded. NaN samples are skipped; a gap of
    more than ``max_gap_min`` consecutive NaNs ends the event.
    """
    f = np.asarray(flux, dtype=float)
    n = len(f)
    n_inc = max(rise_min - 1, 1)
    if n <= n_inc + 1:
        return []
    inc = (np.diff(f) > 0).astype(int)  # NaN comparisons are False
    run = np.convolve(inc, np.ones(n_inc, dtype=int), mode="valid")  # length n - n_inc
    first, last = f[: n - n_inc], f[n_inc:]
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero((run == n_inc) & (last >= rise_factor * first))

    events: list[dict] = []
    last_end = -1
    for i in cand:
        if i <= last_end:
            continue
        f0 = f[i]
        peak, pidx, end, nan_run = f0, int(i), None, 0
        k = int(i) + 1
        while k < n and k - i <= max_duration_min:
            v = f[k]
            if np.isnan(v):
                nan_run += 1
                if nan_run > max_gap_min:
                    end = k - nan_run
                    break
                k += 1
                continue
            nan_run = 0
            if v > peak:
                peak, pidx = v, k
            elif v <= f0 + end_fraction * (peak - f0):
                end = k
                break
            k += 1
        if end is None:
            end = min(k, n - 1)
        last_end = end
        if peak >= min_peak_flux:
            events.append(
                {"start": int(i), "peak": pidx, "end": int(end), "peak_flux": float(peak)}
            )
    return events


@LABELERS.register("goes_events")
class GoesEventLabeler(Labeler):
    """Calibrated classes: each event gets the GOES class of its peak flux over its whole span."""

    @dataclass
    class Params:
        rise_min: int = 4
        rise_factor: float = 1.4
        end_fraction: float = 0.5
        min_peak_flux: float = 1e-7
        max_duration_min: int = 720
        max_gap_min: int = 30
        flux_column: str = "sxr_raw"

    def label(self, inp: LabelInput) -> dict[str, np.ndarray]:
        p = self.params
        flux = inp.df[p.flux_column].to_numpy(dtype=float)
        evs = detect_goes_events(
            flux, p.rise_min, p.rise_factor, p.end_fraction, p.min_peak_flux,
            p.max_duration_min, p.max_gap_min,
        )  # fmt: skip
        spans = [(e["start"], e["end"], int(flux_to_class_idx(e["peak_flux"]))) for e in evs]
        return events_to_arrays(len(flux), spans)

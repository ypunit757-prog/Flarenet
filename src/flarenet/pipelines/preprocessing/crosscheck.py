"""Compare detected flare events with an official catalog (e.g. our GOES detector vs NASA DONKI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.core.goes import class_type_to_flux, flux_to_class_idx


def compare_event_tables(
    detected: pd.DataFrame, catalog: pd.DataFrame, t0_unix: float, tol_min: float = 15.0
) -> dict:
    """Match detected events (``peak_min`` since t0) to catalog flares by peak time.

    Returns counts, recall/precision of the detector w.r.t. the catalog, and the share of matched
    pairs whose class letter agrees. A catalog usually lists only C-class and above, so only
    detected events of class C or stronger are used for precision.
    """
    det_peak = (
        detected["peak_min"].to_numpy(dtype=float) * 60.0 + t0_unix
        if len(detected)
        else np.array([])
    )
    det_cls = detected["class_idx"].to_numpy() if len(detected) else np.array([], dtype=int)
    cat_peak = catalog["peak_unix"].to_numpy(dtype=float) if len(catalog) else np.array([])
    cat_cls = (
        flux_to_class_idx([class_type_to_flux(c) for c in catalog["class_type"]])
        if len(catalog)
        else np.array([], dtype=int)
    )
    tol = tol_min * 60.0
    used: set[int] = set()
    matched = agree = 0
    for cp, cc in zip(cat_peak, cat_cls, strict=True):
        if not np.isfinite(cp) or len(det_peak) == 0:
            continue
        d = np.abs(det_peak - cp)
        for j in np.argsort(d):
            if d[j] > tol:
                break
            if int(j) not in used:
                used.add(int(j))
                matched += 1
                agree += int(det_cls[j] == cc)
                break
    n_det_c = int((det_cls >= 2).sum())
    matched_c = sum(1 for j in used if det_cls[j] >= 2)
    return {
        "n_catalog": int(len(catalog)),
        "n_detected": int(len(detected)),
        "n_detected_C_plus": n_det_c,
        "matched": matched,
        "recall_vs_catalog": matched / len(catalog) if len(catalog) else None,
        "precision_C_plus": matched_c / n_det_c if n_det_c else None,
        "class_agreement": agree / matched if matched else None,
        "tolerance_min": tol_min,
    }

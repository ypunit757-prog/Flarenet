"""GOES flare-class helpers: flux <-> class letter/number, on the 0.1-0.8 nm W/m^2 scale."""

from __future__ import annotations

import re

import numpy as np

from flarenet.core.constants import CLASS_NAMES, CLASS_THRESHOLDS, GOES_LETTER_FLUX

_CLASS_RE = re.compile(r"^\s*([ABCMX])\s*([0-9]*\.?[0-9]*)\s*$", re.IGNORECASE)


def class_type_to_flux(class_type: str | None) -> float:
    """'M5.2' -> 5.2e-5 W/m^2, 'X' -> 1e-4, anything unparseable -> NaN."""
    if not class_type or not isinstance(class_type, str):
        return float("nan")
    m = _CLASS_RE.match(class_type)
    if not m:
        return float("nan")
    letter, num = m.group(1).upper(), m.group(2)
    return GOES_LETTER_FLUX[letter] * (float(num) if num else 1.0)


def flux_to_class_idx(flux) -> np.ndarray:
    """Flux (W/m^2) -> class index 0..4 (QUIET/A below 1e-7, then B, C, M, X)."""
    f = np.asarray(flux, dtype=float)
    idx = np.zeros(f.shape, dtype=int)
    with np.errstate(invalid="ignore"):
        for i, thr in enumerate(CLASS_THRESHOLDS, start=1):
            idx[f >= thr] = i
    return idx


def flux_to_class_type(flux: float) -> str:
    """5.2e-5 -> 'M5.2'; below B-class returns 'A<x>' (e.g. 'A3.0')."""
    if not np.isfinite(flux) or flux <= 0:
        return "?"
    letter = "A"
    base = GOES_LETTER_FLUX["A"]
    for name in ("B", "C", "M", "X"):
        if flux >= GOES_LETTER_FLUX[name]:
            letter, base = name, GOES_LETTER_FLUX[name]
    return f"{letter}{flux / base:.1f}"


def class_idx_to_name(idx: int) -> str:
    return CLASS_NAMES[int(idx)]

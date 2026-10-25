"""Shared constants for FlareNet."""

# Five nowcast classes, in increasing flare strength.
CLASS_NAMES = ["QUIET", "B", "C", "M", "X"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Peak-flux bands (W/m^2-like units used by the synthetic generator) per class.
CLASS_BANDS = {
    "QUIET": (1e-9, 1e-7),
    "B": (1e-7, 1e-6),
    "C": (1e-6, 1e-5),
    "M": (1e-5, 1e-4),
    "X": (1e-4, 1e-3),
}

N_CLASSES = len(CLASS_NAMES)
N_CHANNELS = 12  # engineered input channels, see features.py

M_CLASS_IDX = CLASS_TO_IDX["M"]

# Dark dashboard palette.
CLASS_COLORS = {
    "QUIET": "#888888",
    "B": "#4a90e2",
    "C": "#2ecc71",
    "M": "#e67e22",
    "X": "#e74c3c",
}
BG_DARK = "#0d1117"
BG_PANEL = "#161b22"
GRID_KW = {"alpha": 0.18, "color": "#555555"}

# ---- GOES / calibrated-flux definitions (W/m^2, 0.1-0.8 nm channel) ------------------------
# Lower flux bound of classes B, C, M, X. Anything below 1e-7 is QUIET (GOES A-class and below).
CLASS_THRESHOLDS = (1e-7, 1e-6, 1e-5, 1e-4)
# Flux that corresponds to "<letter>1.0", used to turn strings such as 'M5.2' into a flux.
GOES_LETTER_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

# Columns of the processed 1-minute lightcurve every pipeline hands to the next one.
LIGHTCURVE_COLUMNS = [
    "time_min",
    "sxr",
    "hxr",
    "sxr_raw",
    "hxr_raw",
    "flare_class",
    "flare_id",
    "active_regime",
]

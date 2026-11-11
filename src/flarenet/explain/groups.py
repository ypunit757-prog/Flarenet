"""Readable names for the model inputs.

Single features such as ``f_max_48h`` and ``h_nM_6`` mean little to a reader, and neighbouring ones (the 24-hour and 48-hour
maximum) carry nearly the same information, so credit would be split between them arbitrarily. Inputs are therefore explained
in groups of related features. Every feature the tree model can see belongs to exactly one group (a test checks this).
"""

from __future__ import annotations

# group id -> (label, features)
TREE_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "level_now": ("X-ray level right now",
                  ("f_cur", "f_max_30m", "f_max_60m", "f_max_120m", "f_rel_q10_72h", "f_rel_mean_24h")),
    "trend": ("Rising or falling flux", ("f_slope_30m", "f_slope_1h", "f_trend_6h", "f_since_max_6h")),
    "peak_hours": ("Peak flux over the last 3 to 12 hours", ("f_max_3h", "f_max_6h", "f_max_12h")),
    "activity_days": ("Background activity over the last 1 to 3 days",
                      ("f_max_24h", "f_max_48h", "f_max_72h", "f_mean_6h", "f_mean_24h", "f_std_24h",
                       "f_q10_72h", "f_q90_24h", "f_linmean_24h", "f_linmean_72h")),
    "flares_hours": ("Flares in the last 3 to 6 hours", ("h_nC_3", "h_nM_3", "h_nC_6", "h_nM_6")),
    "flares_day": ("Flares in the last day", ("h_nC_24", "h_nM_24", "h_nX_24", "h_peak_24")),
    "flares_3days": ("Flares in the last 3 days", ("h_nC_72", "h_nM_72", "h_nX_72", "h_peak_72")),
    "since_flare": ("Time since the last C, M and X flare", ("h_since_C", "h_since_M", "h_since_X")),
    "history_week": ("Seven days and one solar rotation back",
                     ("f_max_168h", "h_nC_168", "h_nM_168", "h_nX_168", "h_peak_168",
                      "rot_nM_slot", "rot_nC_slot", "rot_nM_wide")),
    "coverage": ("How complete the last 24 hours of data are", ("f_cov_24h",)),
}  # fmt: skip

# the network's 12 input channels (flarenet.pipelines.features.channels.CHANNEL_NAMES)
CHANNEL_LABELS: dict[str, str] = {
    "log_sxr": "Soft X-ray flux (long band)",
    "log_hxr": "Short-band X-ray flux",
    "d_log_sxr": "Rate of change of the soft flux",
    "d_log_hxr": "Rate of change of the short-band flux",
    "d2_log_sxr": "Acceleration of the soft flux",
    "d2_log_hxr": "Acceleration of the short-band flux",
    "hardness": "Spectral hardness (short band against long band)",
    "max_sxr_2h": "Highest flux in the last 2 hours",
    "max_sxr_6h": "Highest flux in the last 6 hours",
    "ewma_sxr_4h": "4-hour running average of the flux",
    "log_fluence_1h": "Energy released in the last hour",
    "recent_flare_rate_6h": "Flares detected in the last 6 hours",
}


def tree_groups(columns) -> dict[str, tuple[str, list[str]]]:
    """Groups restricted to the feature columns a fitted tree model uses (empty groups dropped). A feature that belongs
    to no known group goes into ``other`` so that nothing the model reads is left unexplained."""
    cols = list(columns)
    seen: set[str] = set()
    out: dict[str, tuple[str, list[str]]] = {}
    for gid, (label, feats) in TREE_GROUPS.items():
        mine = [c for c in cols if c in feats]
        if mine:
            out[gid] = (label, mine)
            seen.update(mine)
    rest = [c for c in cols if c not in seen]
    if rest:
        out["other"] = ("Other inputs", rest)
    return out


def channel_groups(channels) -> dict[str, tuple[str, list[int]]]:
    """One group per network input channel (the channels are already distinct measurements): id -> (label, [index])."""
    return {c: (CHANNEL_LABELS.get(c, c), [i]) for i, c in enumerate(channels)}

"""Percentile-relative classes for instruments without a flux calibration (Aditya-L1 SoLEXS)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flarenet.core.registry import LABELERS
from flarenet.pipelines.preprocessing.labelers.base import Labeler, LabelInput


@LABELERS.register("relative_percentile")
class RelativePercentileLabeler(Labeler):
    """Class = number of multipliers of the ``percentile`` count rate that the soft rate exceeds.

    This is the v5.1 real-data proxy: it is relative, not calibrated GOES, and it uses the whole
    series' percentile, so it is not causal. Contiguous minutes with class > 0 form one event.
    """

    @dataclass
    class Params:
        percentile: float = 10.0
        multipliers: list[float] = field(default_factory=lambda: [1.5, 3.0, 8.0, 20.0])  # B C M X

    def label(self, inp: LabelInput) -> dict[str, np.ndarray]:
        s = inp.df["sxr_raw"].to_numpy(dtype=float)
        base = np.nanpercentile(s, self.params.percentile)
        cls = np.zeros(len(s), dtype=int)
        with np.errstate(invalid="ignore"):
            for idx, mult in enumerate(self.params.multipliers, start=1):
                cls[s > base * mult] = idx
        active = cls > 0
        starts = active & ~np.concatenate([[False], active[:-1]])
        fid = np.where(active, np.cumsum(starts), -1).astype(int)  # ids from 1, as in v5.1
        return {"flare_class": cls, "flare_id": fid, "active_regime": (cls >= 2).astype(int)}

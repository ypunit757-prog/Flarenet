"""Labels taken from an official flare catalog (NASA DONKI, or any CSV) instead of the flux."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flarenet.core.goes import class_type_to_flux, flux_to_class_idx
from flarenet.core.registry import LABELERS
from flarenet.pipelines.preprocessing.labelers.base import Labeler, LabelInput, events_to_arrays
from flarenet.sources.catalogs import make_catalog


@LABELERS.register("catalog")
class CatalogLabeler(Labeler):
    """Mark every catalog flare from begin to end with its catalog class.

    Missing end times are filled with ``peak + (peak - begin)`` or ``default_duration_min``.
    """

    @dataclass
    class Params:
        catalog: str = "donki"
        catalog_params: dict[str, Any] = field(default_factory=dict)
        default_duration_min: float = 30.0

    def label(self, inp: LabelInput) -> dict[str, np.ndarray]:
        p = self.params
        n = len(inp.df)
        t0 = inp.t0_unix
        cat = make_catalog(p.catalog, p.catalog_params).fetch(t0, t0 + n * 60.0)
        spans = []
        for r in cat.sort_values("begin_unix").itertuples():
            begin = float(r.begin_unix)
            peak = float(r.peak_unix) if np.isfinite(r.peak_unix) else np.nan
            if np.isfinite(r.end_unix):
                end = float(r.end_unix)
            elif np.isfinite(peak):
                end = peak + (peak - begin)
            else:
                end = begin + p.default_duration_min * 60.0
            cls = int(flux_to_class_idx(class_type_to_flux(r.class_type)))
            spans.append((int(np.floor((begin - t0) / 60.0)), int(np.ceil((end - t0) / 60.0)), cls))
        return events_to_arrays(n, spans)

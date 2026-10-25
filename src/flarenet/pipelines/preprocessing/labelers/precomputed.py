"""Ground-truth labels delivered with the data (synthetic generator)."""

from __future__ import annotations

import numpy as np

from flarenet.core.registry import LABELERS
from flarenet.core.schema import SchemaError
from flarenet.pipelines.preprocessing.labelers.base import Labeler, LabelInput


@LABELERS.register("precomputed")
class PrecomputedLabeler(Labeler):
    def label(self, inp: LabelInput) -> dict[str, np.ndarray]:
        missing = [k for k in ("flare_class", "flare_id") if k not in inp.aligned]
        if missing:
            raise SchemaError(
                f"labeler 'precomputed' needs the source to provide {missing}; this source "
                "does not. Choose a labeler such as 'goes_events' or 'relative_percentile'."
            )
        n = len(inp.df)
        regime = inp.aligned.get("active_regime", np.zeros(n, dtype=int))
        return {
            "flare_class": inp.aligned["flare_class"].astype(int),
            "flare_id": inp.aligned["flare_id"].astype(int),
            "active_regime": regime.astype(int),
        }

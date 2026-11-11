"""JSON that every browser can read.

Python's ``json`` writes ``NaN`` and ``Infinity`` for missing numbers, which are not JSON: ``JSON.parse`` in a browser throws on
them, so one undefined precision (no alerts, so 0 / 0) would break a whole page. ``clean`` turns them into ``null`` and numpy
values into plain Python ones; ``dumps`` refuses to write anything that is still not valid JSON.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def clean(obj: Any) -> Any:
    """A copy of ``obj`` with numpy scalars/arrays converted and every NaN or infinity replaced by None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def dumps(obj: Any, **kw) -> str:
    """``json.dumps`` of ``clean(obj)`` with ``allow_nan=False``, so the output is always valid JSON."""
    return json.dumps(clean(obj), allow_nan=False, **kw)

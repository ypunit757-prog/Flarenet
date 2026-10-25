"""Labelers register themselves on import."""

from flarenet.pipelines.preprocessing.labelers import (  # noqa: F401
    catalog,
    goes_events,
    precomputed,
    relative,
)
from flarenet.pipelines.preprocessing.labelers.base import Labeler, LabelInput  # noqa: F401

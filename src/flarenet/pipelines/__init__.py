"""One sub-package per pipeline type, each exposing ``build_pipeline()``.

ingestion -> preprocessing -> features -> training -> evaluation -> reporting
                                      `-> inference (offline predict, live monitor)
"""

from flarenet.pipelines import (  # noqa: F401
    evaluation,
    features,
    inference,
    ingestion,
    preprocessing,
    reporting,
    training,
)

"""Workflows that compose the pipelines (train, evaluate, run, predict ...)."""

from flarenet.orchestration.runner import (
    add_aditya_source,
    run_all,
    run_evaluate,
    run_features,
    run_ingest,
    run_noaa_compare,
    run_predict,
    run_preprocess,
    run_train,
)

__all__ = [
    "add_aditya_source", "run_all", "run_evaluate", "run_features", "run_ingest",
    "run_noaa_compare", "run_predict", "run_preprocess", "run_train",
]  # fmt: skip

"""Feature pipeline: labelled lightcurve -> engineered channels -> windows -> block split."""

from flarenet.pipelines.features.pipeline import build_pipeline

__all__ = ["build_pipeline"]

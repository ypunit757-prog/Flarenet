"""Ingestion pipeline: fetch raw instrument data into the on-disk store."""

from flarenet.pipelines.ingestion.pipeline import build_pipeline

__all__ = ["build_pipeline"]

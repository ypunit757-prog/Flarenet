"""Importing this package registers every built-in data source and event catalog."""

from flarenet.sources import aditya_l1, catalogs, synthetic  # noqa: F401
from flarenet.sources.base import DataSource, SourceUnavailable, make_source  # noqa: F401
from flarenet.sources.nasa import donki, goes_file, goes_ncei, goes_swpc  # noqa: F401

__all__ = ["DataSource", "SourceUnavailable", "make_source"]

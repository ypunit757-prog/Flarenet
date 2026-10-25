"""Tiny plug-in registries. Adding a data source, labeler or model is one decorated class."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> class mapping with a ``@REGISTRY.register("name")`` decorator."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if key in self._items and self._items[key] is not obj:
                raise ValueError(f"{self.kind} {key!r} is already registered")
            self._items[key] = obj
            if isinstance(obj, type) and hasattr(obj, "kind") and vars(obj).get("kind", "") == "":
                obj.kind = (
                    key  # e.g. DataSource.kind, so reports and fingerprints name the source type
                )
            return obj

        return deco

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} {key!r}. Available: {', '.join(sorted(self._items)) or '(none)'}"
            ) from None

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items


# Populated by importing flarenet.sources / pipelines.preprocessing.labelers / models.
SOURCES: Registry = Registry("data source")
CATALOGS: Registry = Registry("event catalog")
LABELERS: Registry = Registry("labeler")
MODELS: Registry = Registry("model")

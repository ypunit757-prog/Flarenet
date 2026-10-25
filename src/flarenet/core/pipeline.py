"""Pipeline framework: a ``Pipeline`` is an ordered list of ``Stage`` objects sharing a context.

Every pipeline type in ``flarenet.pipelines.*`` is built from this, so they all log, time and
record themselves the same way and write a ``manifest_<pipeline>.json`` into the run directory.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flarenet import __version__
from flarenet.core.artifacts import DataStore
from flarenet.core.config import Config
from flarenet.core.utils import fingerprint, get_logger, save_json

log = get_logger(__name__)


class PipelineError(RuntimeError):
    """A stage failed; ``stage`` names which one."""

    def __init__(self, pipeline: str, stage: str, cause: BaseException):
        super().__init__(f"Pipeline {pipeline!r} failed in stage {stage!r}: {cause}")
        self.pipeline, self.stage, self.__cause__ = pipeline, stage, cause

    def __reduce__(self):  # lets the error cross a process boundary (multiprocessing, executors)
        return (PipelineError, (self.pipeline, self.stage, self.__cause__))


@dataclass
class PipelineContext:
    """Shared state for one pipeline execution."""

    cfg: Config
    store: DataStore
    run_dir: Path
    force: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Config, force: bool = False) -> PipelineContext:
        run_dir = cfg.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(cfg, DataStore(cfg.data_dir), run_dir, force)


class Stage(ABC):
    """One step. Read what you need from ``ctx.state`` / the store, write results back."""

    name: str = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None: ...


class Pipeline:
    def __init__(self, name: str, stages: list[Stage]):
        self.name = name
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        records = []
        status = "ok"
        t_all = time.time()
        log.info("=== pipeline '%s' (%d stages) ===", self.name, len(self.stages))
        try:
            for stage in self.stages:
                t0 = time.time()
                log.info("--- %s.%s", self.name, stage.name)
                try:
                    stage.run(ctx)
                except Exception as e:
                    records.append(
                        {"stage": stage.name, "status": "failed", "seconds": time.time() - t0}
                    )
                    status = "failed"
                    log.error("Stage %s failed:\n%s", stage.name, traceback.format_exc())
                    raise PipelineError(self.name, stage.name, e) from e
                records.append({"stage": stage.name, "status": "ok", "seconds": time.time() - t0})
        finally:
            save_json(
                {
                    "pipeline": self.name,
                    "status": status,
                    "flarenet_version": __version__,
                    "config_fingerprint": fingerprint(ctx.cfg.to_dict()),
                    "total_seconds": time.time() - t_all,
                    "stages": records,
                },
                ctx.run_dir / f"manifest_{self.name}.json",
            )
        log.info("=== pipeline '%s' done in %.1fs ===", self.name, time.time() - t_all)
        return ctx

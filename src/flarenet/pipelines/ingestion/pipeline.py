"""Ingestion pipeline: fetch every configured source and cache its raw samples.

Caches live in ``<data_dir>/raw/<source>/``; a source is re-fetched only when its fingerprint
(parameters, input-file sizes and mtimes, seed) changes, when ``--force`` is given, or when it is
a live feed (fingerprint ends in ``-live``).
"""

from __future__ import annotations

from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.utils import get_logger
from flarenet.sources.base import SourceUnavailable, make_source

log = get_logger(__name__)


class FetchStage(Stage):
    name = "fetch"

    def run(self, ctx: PipelineContext) -> None:
        names = ctx.state.get("source_names") or [s.name for s in ctx.cfg.sources]
        ctx.state["source_names"] = names
        fps: dict[str, str] = ctx.state.setdefault("raw_fingerprints", {})
        for name in names:
            spec = ctx.cfg.source(name)
            src = make_source(spec, ctx.cfg.seed)
            fp = src.fingerprint()
            d = ctx.store.raw_dir(name)
            live = fp.endswith("-live")
            if not ctx.force and not live and ctx.store.is_fresh(d, fp):
                log.info("[%s] raw data up to date (cache hit)", name)
            else:
                try:
                    raw = src.fetch()
                except SourceUnavailable as e:
                    raise SourceUnavailable(f"[{name}] ({spec.kind}) {e}") from e
                ctx.store.save_raw(raw, fp)
                log.info("[%s] fetched %d soft samples (%.1f days)", name, len(raw.channels["sxr"]),
                         raw.span_days)  # fmt: skip
            fps[name] = fp


def build_pipeline() -> Pipeline:
    return Pipeline("ingestion", [FetchStage()])

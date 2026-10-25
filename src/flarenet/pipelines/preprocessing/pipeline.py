"""Preprocessing pipeline: RawSeries -> 1-minute grid -> scaling -> labels -> LightcurveBundle.

Stages: ``resample`` -> ``scale`` -> ``label`` -> ``bundle`` (event table, validation, cache).
Bundles are cached in ``<data_dir>/processed/<source>/`` keyed by a fingerprint of the raw data and
every setting that affects the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flarenet.core.config import SourceSpec
from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.registry import LABELERS
from flarenet.core.schema import LightcurveBundle, validate_lightcurve
from flarenet.core.utils import fingerprint, get_logger
from flarenet.pipelines.preprocessing import labelers  # noqa: F401  (registers labelers)
from flarenet.pipelines.preprocessing.crosscheck import compare_event_tables
from flarenet.pipelines.preprocessing.flare_db import build_flare_db, format_flare_db
from flarenet.pipelines.preprocessing.labelers.base import LabelInput
from flarenet.pipelines.preprocessing.resample import resample_to_minutes
from flarenet.pipelines.preprocessing.scaling import apply_scaling
from flarenet.sources.base import make_source

log = get_logger(__name__)
PROCESSING_VERSION = "1"


@dataclass
class Work:
    """Intermediate state of one source while it moves through the stages."""

    spec: SourceSpec
    labeler_name: str
    scaling: str
    labeler_params: dict
    cross_check: dict | None
    fp: str
    raw: object | None = None
    df: pd.DataFrame | None = None
    t0: float = 0.0
    aligned: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    labels: dict | None = None
    cached: bool = False


def resolve_policy(spec: SourceSpec, cfg_seed: int = 42) -> tuple[str, str, dict, dict | None]:
    """Labeler / scaling for a source: explicit config wins, else the source kind's defaults."""
    src = make_source(spec, cfg_seed)
    lab = spec.labeler or src.default_labeler
    scaling = spec.scaling or src.default_scaling
    lp = dict(spec.labeler_params)
    cross = lp.pop("cross_check", None)
    return lab, scaling, lp, cross


def _processed_fp(raw_fp: str, labeler: str, scaling: str, lp: dict, cross: dict | None) -> str:
    return fingerprint(
        raw_fp, labeler, scaling, lp, cross, PROCESSING_VERSION
    )  # + scaling params below


class _Base(Stage):
    def works(self, ctx: PipelineContext) -> list[Work]:
        return [w for w in ctx.state["work"].values() if not w.cached]


class PlanStage(Stage):
    """Decide labeler / scaling per source and skip sources whose bundle is already fresh."""

    name = "plan"

    def run(self, ctx: PipelineContext) -> None:
        raw_fps = ctx.state["raw_fingerprints"]
        ctx.state["work"] = {}
        ctx.state.setdefault("bundles", {})
        for name in ctx.state["source_names"]:
            spec = ctx.cfg.source(name)
            lab, scaling, lp, cross = resolve_policy(spec, ctx.cfg.seed)
            fp = _processed_fp(raw_fps[name], lab, scaling,
                               {**lp, "_": spec.labeler_params,
                                **({"scaling_params": spec.scaling_params} if spec.scaling_params else {})},
                               cross)  # fmt: skip
            w = Work(spec, lab, scaling, lp, cross, fp)
            pdir = ctx.store.processed_dir(name)
            if (
                not ctx.force
                and not raw_fps[name].endswith("-live")
                and ctx.store.is_fresh(pdir, fp)
            ):
                ctx.state["bundles"][name] = ctx.store.load_bundle(name)
                w.cached = True
                log.info("[%s] processed lightcurve is up to date (cache hit)", name)
            ctx.state["work"][name] = w
            if not w.cached:
                w.raw = ctx.store.load_raw(name)


class ResampleStage(_Base):
    name = "resample"

    def run(self, ctx: PipelineContext) -> None:
        for w in self.works(ctx):
            w.df, w.t0, w.aligned, w.info = resample_to_minutes(w.raw)
            log.info("[%s] %d one-minute samples, soft coverage %.1f%%", w.spec.name, len(w.df),
                     100 * w.info["coverage_sxr"])  # fmt: skip


class ScaleStage(_Base):
    name = "scale"

    def run(self, ctx: PipelineContext) -> None:
        for w in self.works(ctx):
            w.df = apply_scaling(w.scaling, w.df, w.spec.scaling_params, w.t0)
            if w.df.attrs.get("calibration"):
                w.info["calibration"] = w.df.attrs["calibration"]
                log.info("[%s] cross-calibration: %s", w.spec.name, w.info["calibration"])


class LabelStage(_Base):
    name = "label"

    def run(self, ctx: PipelineContext) -> None:
        for w in self.works(ctx):
            labeler = LABELERS.get(w.labeler_name)(w.labeler_params)
            w.labels = labeler.label(LabelInput(w.df, w.t0, w.aligned))
            n_ev = len(np.unique(w.labels["flare_id"][w.labels["flare_id"] >= 0]))
            log.info("[%s] labeler=%s scaling=%s -> %d events", w.spec.name, w.labeler_name,
                     w.scaling, n_ev)  # fmt: skip


def build_bundle(w: Work, seed: int = 42) -> LightcurveBundle:
    """Assemble, validate and describe the labelled lightcurve of one source (no I/O)."""
    df = w.df.copy()
    for k in ("flare_class", "flare_id", "active_regime"):
        df[k] = w.labels[k]
    df = df[
        ["time_min", "sxr", "hxr", "sxr_raw", "hxr_raw", "flare_class", "flare_id", "active_regime"]
    ]
    validate_lightcurve(df, w.spec.name)
    fdb = build_flare_db(df, label=w.spec.name, t0_unix=w.t0)
    src = make_source(w.spec, seed)
    meta = {
        **src.describe(),
        "labeler": w.labeler_name,
        "scaling": w.scaling,
        "n_events": int(len(fdb)),
        "event_classes": fdb["class"].value_counts().to_dict() if len(fdb) else {},
        **w.info,
    }
    if w.cross_check:
        meta["cross_check"] = _cross_check(w, fdb)
    return LightcurveBundle(w.spec.name, w.spec.kind, df, w.t0, meta, fdb)


def _cross_check(w: Work, fdb: pd.DataFrame) -> dict:
    from flarenet.sources.catalogs import make_catalog

    cc = w.cross_check
    n = len(w.df)
    cat = make_catalog(cc.get("catalog", "donki"), cc.get("catalog_params", {})).fetch(
        w.t0, w.t0 + n * 60.0
    )
    res = compare_event_tables(fdb, cat, w.t0, float(cc.get("tolerance_min", 15.0)))
    log.info("[%s] cross-check vs %s: %s", w.spec.name, cc.get("catalog", "donki"), res)
    return res


class BundleStage(_Base):
    name = "bundle"

    def run(self, ctx: PipelineContext) -> None:
        for w in self.works(ctx):
            bundle = build_bundle(w, ctx.cfg.seed)
            ctx.store.save_bundle(bundle, w.fp)
            ctx.state["bundles"][w.spec.name] = bundle
            log.info(format_flare_db(bundle.flare_db, w.spec.name).splitlines()[0])


def process_raw_series(spec: SourceSpec, raw, seed: int = 42) -> LightcurveBundle:
    """Run resample -> scale -> label -> bundle for one RawSeries in memory (no cache).
    Used by ``flarenet predict`` on ad-hoc data and by the live monitor."""
    lab, scaling, lp, cross = resolve_policy(spec, seed)
    w = Work(spec, lab, scaling, lp, cross, fp="")
    w.raw = raw
    w.df, w.t0, w.aligned, w.info = resample_to_minutes(raw)
    w.df = apply_scaling(scaling, w.df, spec.scaling_params, w.t0)
    if w.df.attrs.get("calibration"):
        w.info["calibration"] = w.df.attrs["calibration"]
    w.labels = LABELERS.get(lab)(lp).label(LabelInput(w.df, w.t0, w.aligned))
    return build_bundle(w, seed)


def build_pipeline() -> Pipeline:
    return Pipeline(
        "preprocessing", [PlanStage(), ResampleStage(), ScaleStage(), LabelStage(), BundleStage()]
    )

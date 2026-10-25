"""Compose the pipelines into the user-facing workflows.

Every workflow is ``ingestion -> preprocessing -> features`` (all cached on disk) followed by
the pipeline(s) it is about, so each step can also be run on its own and re-runs are cheap.
"""

from __future__ import annotations

from pathlib import Path

from flarenet.core.config import Config, SourceSpec
from flarenet.core.pipeline import PipelineContext
from flarenet.core.utils import get_logger, save_json, set_seed
from flarenet.pipelines import (
    evaluation,
    features,
    inference,
    ingestion,
    preprocessing,
    reporting,
    training,
)

log = get_logger(__name__)


def _context(cfg: Config, force: bool, names: list[str] | None) -> PipelineContext:
    set_seed(cfg.seed)
    ctx = PipelineContext.from_config(cfg, force)
    ctx.state["source_names"] = names or [s.name for s in cfg.sources]
    save_json(cfg.to_dict(), ctx.run_dir / "config_used.json")
    return ctx


def _prepare(ctx: PipelineContext, upto: str) -> None:
    ingestion.build_pipeline().run(ctx)
    if upto == "ingest":
        return
    preprocessing.build_pipeline().run(ctx)
    if upto == "preprocess":
        return
    features.build_pipeline().run(ctx)


def run_ingest(cfg: Config, force: bool = False, names: list[str] | None = None) -> PipelineContext:
    ctx = _context(cfg, force, names)
    _prepare(ctx, "ingest")
    return ctx


def run_preprocess(
    cfg: Config, force: bool = False, names: list[str] | None = None
) -> PipelineContext:
    ctx = _context(cfg, force, names)
    _prepare(ctx, "preprocess")
    return ctx


def run_features(
    cfg: Config, force: bool = False, names: list[str] | None = None
) -> PipelineContext:
    ctx = _context(cfg, force, names)
    _prepare(ctx, "features")
    return ctx


def run_train(cfg: Config, force: bool = False) -> PipelineContext:
    """Train on the ``role: train`` sources and write ``<run_dir>/checkpoint.pt``."""
    names = [s.name for s in cfg.train_sources()]
    if not names:
        raise ValueError("No source with role 'train' configured.")
    ctx = _context(cfg, force, names)
    _prepare(ctx, "features")
    training.build_pipeline().run(ctx)
    return ctx


def run_evaluate(
    cfg: Config, checkpoint: str | Path | None = None, force: bool = False,
    ctx: PipelineContext | None = None,
) -> PipelineContext:  # fmt: skip
    """Score a checkpoint on test splits + eval sources, then write report, tables, figures."""
    if ctx is None:
        ctx = _context(cfg, force, None)
        _prepare(ctx, "features")
    if checkpoint:
        ctx.state["checkpoint"] = str(checkpoint)
    evaluation.build_pipeline().run(ctx)
    reporting.build_pipeline().run(ctx)
    return ctx


def run_all(cfg: Config, force: bool = False) -> PipelineContext:
    """Whole experiment in one go: data -> features -> train -> evaluate -> report."""
    ctx = _context(cfg, force, None)
    _prepare(ctx, "features")
    training.build_pipeline().run(ctx)
    return run_evaluate(cfg, ctx=ctx)


def run_predict(
    cfg: Config, checkpoint: str | Path, source: str, out: str | Path | None = None, force: bool = False
) -> PipelineContext:  # fmt: skip
    """Forecast over one configured source (any role, any instrument) with a checkpoint."""
    cfg.source(source)  # raises a helpful KeyError if unknown
    ctx = _context(cfg, force, [source])
    ctx.state["checkpoint"] = str(checkpoint)
    if out:
        ctx.state["predictions_path"] = str(out)
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    inference.build_pipeline().run(ctx)
    return ctx


def add_aditya_source(
    cfg: Config, solexs: str, hel1os: list[str], role: str = "eval"
) -> SourceSpec:
    """Convenience for the ``--solexs/--hel1os`` flags: append an ``aditya_l1`` source."""
    spec = SourceSpec(
        name="aditya_l1", kind="aditya_l1", role=role,
        params={"solexs_path": solexs, "hel1os_dirs": list(hel1os)},
    )  # fmt: skip
    cfg.sources = [s for s in cfg.sources if s.name != spec.name] + [spec]
    return spec


def run_noaa_compare(
    cfg: Config, checkpoint: str | Path, source: str | None = None,
    noaa_csv: str | None = None, allow_in_sample: bool = False, force: bool = False,
) -> dict:  # fmt: skip
    """24 h daily M/X forecasts at 00:00 UTC on one source, scored like NOAA's (and against
    NOAA's own probabilities if ``noaa_csv`` is given). Needs a checkpoint trained with
    ``data.horizon_min: 1440``; the source should be one the model never trained on."""
    import pandas as pd

    from flarenet.core.utils import resolve_device
    from flarenet.pipelines.evaluation import noaa_protocol as npx
    from flarenet.pipelines.evaluation.scoring import infer
    from flarenet.pipelines.features.windows import build_windows
    from flarenet.pipelines.inference.predict import feature_kwargs
    from flarenet.pipelines.training.checkpoint import load_checkpoint

    evals = [s.name for s in cfg.sources if s.role == "eval"]
    source = source or (evals[0] if evals else cfg.sources[0].name)
    spec = cfg.source(source)
    if spec.role == "train" and not allow_in_sample:
        raise ValueError(
            f"Source {source!r} has role 'train': the model has seen its days, so the scores would be "
            "in-sample. Give the data a role 'eval' (a later period) or pass --allow-in-sample."
        )
    ctx = _context(cfg, force, [source])
    ingestion.build_pipeline().run(ctx)
    preprocessing.build_pipeline().run(ctx)
    bundle = ctx.state["bundles"][source]

    device = resolve_device(cfg.training.device)
    lm = load_checkpoint(checkpoint, device)
    d = lm.config["data"]
    if int(d["horizon_min"]) != 1440:
        raise ValueError(
            f"The checkpoint forecasts {d['horizon_min']} min ahead, not 24 h. Train a daily model with "
            "--set data.horizon_min=1440 (see configs/noaa_daily.yaml)."
        )
    starts = npx.midnight_starts(bundle.t0_unix, len(bundle.df), int(d["window_min"]))
    ws = build_windows(
        bundle.df, int(d["window_min"]), 1440, 1440, channel_names=lm.channels,
        max_missing_frac=d.get("max_missing_frac", 1.0), starts=starts, **feature_kwargs(lm.config),
    )  # fmt: skip
    _, _, prob = infer(lm.model, ws.X, lm.mu, lm.sg, device, now_bias=lm.now_bias)
    y = ws.y_fc.astype(int)
    days = pd.to_datetime(bundle.unix_of(ws.t_end), unit="s", utc=True).floor("D")

    noaa = None
    if noaa_csv:
        ser = npx.read_noaa_csv(noaa_csv)
        common = days.isin(ser.index)
        if common.sum() < 30:
            raise ValueError(
                f"Only {int(common.sum())} days overlap with the NOAA file; need >= 30."
            )
        y, prob, days = y[common], prob[common], days[common]
        noaa = ser.loc[days].to_numpy(dtype=float)
    res = npx.compare(y, prob, lm.threshold, noaa)
    res["source"], res["in_sample"] = source, spec.role == "train"
    caveats = [
        "Reference NOAA numbers come from a different period and catalogue; only the "
        "same-days row (requires --noaa-csv) is a like-for-like comparison.",
        "Labels: an M/X flare starts within 24 h of 00:00 UTC, from this source's own flare "
        f"detection ({spec.labeler or 'default labeler'}); NOAA verifies against its event list.",
        "The model sees only the 90 minutes before 00:00 UTC, NOAA forecasters see much more "
        "(active regions, magnetograms). Skill here comes from X-ray history alone.",
    ]
    if res["in_sample"]:
        caveats.insert(
            0, "IN-SAMPLE: this source was used for training; do not quote these scores."
        )
    if res["model_theta_0.5"]["positive_days"] < 20:
        caveats.append(
            f"Only {res['model_theta_0.5']['positive_days']} M/X days: intervals are wide."
        )
    out_dir = ctx.run_dir
    save_json(res, out_dir / "noaa_compare.json")
    (out_dir / "noaa_compare.md").write_text(npx.format_markdown(res, source, caveats))
    log.info("Wrote %s", out_dir / "noaa_compare.md")
    res["_markdown"] = npx.format_markdown(res, source, caveats)
    return res

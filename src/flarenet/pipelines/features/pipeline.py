"""Feature pipeline: bundles -> windows (cached) -> leakage-safe split.

Sources with ``role: train`` are block-split into train/val/test (the test part is kept per
source); sources with ``role: eval`` are never trained on and are scored on all their windows.
"""

from __future__ import annotations

import numpy as np

from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.schema import SchemaError, WindowSet
from flarenet.core.utils import fingerprint, get_logger
from flarenet.pipelines.features.split import block_split
from flarenet.pipelines.features.windows import build_windows

log = get_logger(__name__)


def window_stride(cfg, role: str) -> int:
    """Stride used to cut windows: ``train_stride_min`` for training sources, else ``stride_min``."""
    d = cfg.data
    return d.train_stride_min if role == "train" and d.train_stride_min else d.stride_min


def window_fingerprint(bundle_fp: str, cfg, role: str = "eval") -> str:
    d = cfg.data
    stride = window_stride(cfg, role)
    # stride_min is the same as before for eval sources, so existing caches stay valid
    parts = [bundle_fp, d.window_min, stride, d.horizon_min, d.max_missing_frac,
             cfg.features.channels, "v1"]  # fmt: skip
    f = cfg.features
    if f.relative:  # appended only when on, so caches built without it stay valid
        parts += ["relative", f.baseline_min, f.baseline_quantile]
    return fingerprint(*parts)


def on_eval_grid(ws: WindowSet, window_min: int, stride_min: int) -> np.ndarray:
    """Mask of windows whose start minute lies on the ``stride_min`` grid (start = t_end - window + 1)."""
    start = np.rint(ws.t_end - (window_min - 1)).astype(np.int64)
    return start % stride_min == 0


def _near(t: np.ndarray, other: np.ndarray, gap: float) -> np.ndarray:
    """True where ``t`` is closer than ``gap`` to some value of ``other``."""
    if len(other) == 0 or len(t) == 0:
        return np.zeros(len(t), dtype=bool)
    o = np.sort(other)
    i = np.searchsorted(o, t)
    left = np.abs(t - o[np.clip(i - 1, 0, len(o) - 1)])
    right = np.abs(t - o[np.clip(i, 0, len(o) - 1)])
    return np.minimum(left, right) < gap


def purge_overlaps(
    parts: dict[str, WindowSet], window_min: int, horizon_min: int
) -> dict[str, int]:
    """Remove train windows that overlap val/test and val windows that overlap test (in place).

    A window uses minutes ``[t - window + 1, t]`` as input and ``(t, t + horizon]`` for its forecast
    label, so two windows whose end times are closer than ``window - 1 + horizon`` minutes share
    information. Test windows are never touched, so test sets stay identical with or without purging.
    Returns how many windows were dropped per split.
    """
    gap = window_min - 1 + horizon_min
    t_val, t_test = parts["val"].t_end, parts["test"].t_end
    drop_train = _near(parts["train"].t_end, np.concatenate([t_val, t_test]), gap)
    drop_val = _near(t_val, t_test, gap)
    dropped = {"train": int(drop_train.sum()), "val": int(drop_val.sum())}
    parts["train"] = parts["train"].subset(~drop_train)
    parts["val"] = parts["val"].subset(~drop_val)
    return dropped


class WindowStage(Stage):
    name = "windows"

    def run(self, ctx: PipelineContext) -> None:
        cfg, d = ctx.cfg, ctx.cfg.data
        ctx.state.setdefault("windows", {})
        for name in ctx.state["source_names"]:
            bundle = ctx.state["bundles"].get(name) or ctx.store.load_bundle(name)
            ctx.state["bundles"][name] = bundle
            bfp = ctx.store.read_meta(ctx.store.processed_dir(name))["fingerprint"]
            role = cfg.source(name).role
            stride = window_stride(cfg, role)
            fp = window_fingerprint(bfp, cfg, role)
            fdir = ctx.store.features_dir(name)
            if not ctx.force and not bfp.endswith("-live") and ctx.store.is_fresh(fdir, fp):
                ws = ctx.store.load_windows(name)
                log.info("[%s] %d windows (cache hit)", name, len(ws))
            else:
                ws = build_windows(bundle.df, d.window_min, stride, d.horizon_min,
                                   channel_names=cfg.features.channels,
                                   max_missing_frac=d.max_missing_frac,
                                   relative=cfg.features.relative,
                                   baseline_min=cfg.features.baseline_min,
                                   baseline_quantile=cfg.features.baseline_quantile)  # fmt: skip
                ctx.store.save_windows(name, ws, fp)
                log.info("[%s] built %d windows (forecast-positive %.3f)", name, len(ws),
                         float(ws.y_fc.mean()))  # fmt: skip
            ctx.state["windows"][name] = ws.tagged(name)


class SplitStage(Stage):
    name = "split"

    def run(self, ctx: PipelineContext) -> None:
        cfg, d = ctx.cfg, ctx.cfg.data
        names = ctx.state["source_names"]
        train, val, test, evals = [], [], {}, {}
        weights: dict[str, float] = {}
        for i, name in enumerate(names):
            spec = cfg.source(name)
            ws = ctx.state["windows"][name]
            if spec.role == "eval":
                evals[name] = ws
                continue
            parts = block_split(ws, d.block_days, d.train_frac, d.val_frac, cfg.seed + i)
            if d.train_stride_min and d.train_stride_min != d.stride_min:
                # finer windows are for training only: val / test stay on the stride_min grid
                for k in ("val", "test"):
                    parts[k] = parts[k].subset(on_eval_grid(parts[k], d.window_min, d.stride_min))
            if d.purge:
                dropped = purge_overlaps(parts, d.window_min, d.horizon_min)
                log.info("[%s] purge: dropped %d train and %d val windows overlapping another split",
                         name, dropped["train"], dropped["val"])  # fmt: skip
            train.append(parts["train"])
            val.append(parts["val"])
            test[name] = parts["test"]
            weights[name] = spec.weight
            log.info("[%s] split: train=%d val=%d test=%d windows", name, len(parts["train"]),
                     len(parts["val"]), len(parts["test"]))  # fmt: skip
        out: dict = {"test": test, "eval": evals, "source_weights": weights}
        if train:
            out["train"], out["val"] = WindowSet.concat(train), WindowSet.concat(val)
            if len(out["train"]) == 0 or len(out["val"]) == 0:
                raise SchemaError(
                    "Train or validation split is empty. The series is too short for "
                    f"data.block_days={d.block_days}: lower block_days or fetch more data."
                )
        ctx.state["splits"] = out


def build_pipeline() -> Pipeline:
    return Pipeline("features", [WindowStage(), SplitStage()])

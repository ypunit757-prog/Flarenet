"""Forecast with a trained checkpoint on any configured source (offline files or a live feed)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flarenet.core.config import Config
from flarenet.core.constants import CLASS_NAMES
from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.utils import get_logger, resolve_device
from flarenet.pipelines.evaluation.scoring import infer
from flarenet.pipelines.features.windows import build_windows
from flarenet.pipelines.training.checkpoint import load_checkpoint

log = get_logger(__name__)
PREDICTION_COLUMNS = ["time_min", "time_utc", "p_mx_6h", "alert", "nowcast_class"]


def feature_kwargs(config: dict) -> dict:
    """``build_windows`` options a checkpoint was trained with (older checkpoints: defaults)."""
    f = config.get("features", {}) or {}
    return {
        "relative": bool(f.get("relative", False)),
        "baseline_min": int(f.get("baseline_min", 1440)),
        "baseline_quantile": float(f.get("baseline_quantile", 0.1)),
    }


def forecast_bundle(loaded, bundle, device, for_latest_only: bool = False) -> pd.DataFrame:
    """Windows ending anywhere in the series (including the last hours) -> forecast table."""
    d = loaded.config["data"]
    ws = build_windows(
        bundle.df, d["window_min"], d["stride_min"], d["horizon_min"], for_inference=True,
        channel_names=loaded.channels, max_missing_frac=d.get("max_missing_frac", 1.0),
        **feature_kwargs(loaded.config),
    )  # fmt: skip
    if for_latest_only:
        ws = ws.subset(slice(-1, None))
    pred_cls, _, prob = infer(
        loaded.model, ws.X, loaded.mu, loaded.sg, device, now_bias=loaded.now_bias
    )
    unix = bundle.unix_of(ws.t_end)
    return pd.DataFrame(
        {
            "time_min": ws.t_end,
            "time_utc": pd.to_datetime(unix, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S"),
            "p_mx_6h": prob,
            "alert": (prob >= loaded.threshold).astype(int),
            "nowcast_class": [CLASS_NAMES[i] for i in pred_cls],
        }
    )[PREDICTION_COLUMNS]


class ForecastStage(Stage):
    name = "forecast"

    def run(self, ctx: PipelineContext) -> None:
        from flarenet.pipelines.reporting.dashboard import render_forecast_plot

        cfg: Config = ctx.cfg
        device = resolve_device(cfg.training.device)
        loaded = load_checkpoint(ctx.state["checkpoint"], device)
        name = ctx.state["source_names"][0]
        bundle = ctx.state["bundles"][name]
        out = forecast_bundle(loaded, bundle, device)
        path = Path(ctx.state.get("predictions_path") or ctx.run_dir / f"predictions_{name}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        render_forecast_plot(out, bundle.df, loaded.threshold, str(path.with_suffix(".png")),
                             title=f"{name}: soft X-ray")  # fmt: skip
        ctx.state["predictions"] = out
        ctx.state["bundles"][name] = bundle
        log.info("Wrote %d forecasts to %s | alerts: %d (threshold %.2f)", len(out), path,
                 int(out.alert.sum()), loaded.threshold)  # fmt: skip


def build_pipeline() -> Pipeline:
    return Pipeline("inference", [ForecastStage()])

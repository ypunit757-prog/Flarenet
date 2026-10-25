"""Training pipeline: windows -> fitted model -> ``checkpoint.pt`` + ``history.json``."""

from __future__ import annotations

from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.utils import get_logger, save_json
from flarenet.pipelines.training.trainer import train_model

log = get_logger(__name__)


class FitStage(Stage):
    name = "fit"

    def run(self, ctx: PipelineContext) -> None:
        sp = ctx.state["splits"]
        if "train" not in sp:
            raise ValueError("No source with role 'train' is configured, nothing to train on.")
        ctx.state["training"] = train_model(
            ctx.cfg, sp["train"], sp["val"], ctx.run_dir, sp["source_weights"]
        )


class SaveStage(Stage):
    name = "save"

    def run(self, ctx: PipelineContext) -> None:
        save_json(ctx.state["training"]["history"], ctx.run_dir / "history.json")
        save_json(ctx.cfg.to_dict(), ctx.run_dir / "config_used.json")


def build_pipeline() -> Pipeline:
    return Pipeline("training", [FitStage(), SaveStage()])

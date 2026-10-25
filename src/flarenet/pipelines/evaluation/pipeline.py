"""Evaluation pipeline: score a checkpoint on held-out splits and on external (eval) sources.

* ``role: train`` sources are scored on their held-out test blocks.
* ``role: eval`` sources are scored on all their windows (never trained on).
For each dataset: nowcast accuracy / macro-F1, forecast AUC / TSS / HSS / POD / FAR at the saved
alert threshold, per-class detection, M/X lead times, block-bootstrap intervals and calibration.
"""

from __future__ import annotations

from pathlib import Path

from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.schema import WindowSet
from flarenet.core.utils import get_logger, resolve_device
from flarenet.pipelines.evaluation.baselines import fit_baselines, score_baselines
from flarenet.pipelines.evaluation.scoring import evaluate_dataset
from flarenet.pipelines.evaluation.uncertainty import block_bootstrap_ci, calibration_summary
from flarenet.pipelines.training.checkpoint import load_checkpoint

log = get_logger(__name__)


def align_channels(ws: WindowSet, channels: list[str]) -> WindowSet:
    """Reorder / subset a WindowSet's channels to match a checkpoint's."""
    if ws.channels == channels:
        return ws
    missing = [c for c in channels if c not in ws.channels]
    if missing:
        raise ValueError(
            f"Checkpoint expects channels {channels} but the windows lack {missing}. "
            "Use the config the model was trained with (runs/<name>/config_used.json)."
        )
    out = ws.subset(slice(None))
    out.X = ws.X[:, [ws.channels.index(c) for c in channels], :]
    out.channels = list(channels)
    return out


class LoadModelStage(Stage):
    name = "load_model"

    def run(self, ctx: PipelineContext) -> None:
        path = Path(ctx.state.get("checkpoint") or ctx.run_dir / "checkpoint.pt")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path} (run `flarenet train` first)")
        loaded = load_checkpoint(path, resolve_device(ctx.cfg.training.device))
        trained = loaded.config.get("features") or {}
        f = ctx.cfg.features
        mine = (f.relative, f.baseline_min, f.baseline_quantile)
        theirs = (bool(trained.get("relative", False)), trained.get("baseline_min", 1440),
                  trained.get("baseline_quantile", 0.1))  # fmt: skip
        if mine[0] != theirs[0] or (mine[0] and mine[1:] != theirs[1:]):
            raise ValueError(
                "The checkpoint was trained with features.relative/baseline_min/baseline_quantile = "
                f"{theirs} but this config builds windows with {mine}; evaluating would feed the "
                "model differently scaled inputs. Use the same features settings (or --set them)."
            )
        ctx.state["loaded"] = loaded
        log.info("Loaded %s (alert threshold %.2f, channels %d)", path,
                 ctx.state["loaded"].threshold, len(ctx.state["loaded"].channels))  # fmt: skip


class ScoreStage(Stage):
    name = "score"

    def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.cfg
        lm = ctx.state["loaded"]
        device = resolve_device(cfg.training.device)
        sp = ctx.state["splits"]
        todo = [(n, ws, "test") for n, ws in sp["test"].items()]
        todo += [(n, ws, "external") for n, ws in sp["eval"].items()]
        results = {}
        bparams = None
        if "train" in sp and "val" in sp:
            bparams = fit_baselines(sp["train"], sp["val"], ctx.state["bundles"])
        for name, ws, kind in todo:
            if len(ws) == 0:
                log.warning("[%s] no windows to evaluate, skipped", name)
                continue
            ws = align_channels(ws, lm.channels)
            label = f"{name} ({'held-out test' if kind == 'test' else 'external'})"
            fdb = ctx.state["bundles"][name].flare_db
            res = evaluate_dataset(label, lm.model, ws.X, ws.y_now, ws.y_fc, ws.t_end, lm.mu, lm.sg,
                                   lm.threshold, fdb, device, cfg.data.horizon_min, lm.now_bias)  # fmt: skip
            res.kind = kind
            ev = cfg.evaluation
            res.ci = block_bootstrap_ci(res.forecast_true, res.forecast_prob, res.times, lm.threshold,
                                        ev.bootstrap_n, ev.bootstrap_block_hours, cfg.seed)  # fmt: skip
            res.calibration = calibration_summary(
                res.forecast_true, res.forecast_prob, ev.calibration_bins
            )
            if bparams is not None:
                res.baselines = score_baselines(ws, ctx.state["bundles"], bparams,
                                                res.calibration.get("brier"))  # fmt: skip
            results[name] = res
        if not results:
            raise ValueError("Nothing to evaluate: every source produced zero test windows.")
        ctx.state["eval_results"] = results
        ctx.state["threshold"] = lm.threshold


def build_pipeline() -> Pipeline:
    return Pipeline("evaluation", [LoadModelStage(), ScoreStage()])

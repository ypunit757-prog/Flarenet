"""Reporting pipeline: evaluation results -> JSON / Markdown report, CSV tables and figures."""

from __future__ import annotations

import json

import numpy as np

from flarenet import __version__
from flarenet.core.pipeline import Pipeline, PipelineContext, Stage
from flarenet.core.utils import fingerprint, get_logger, save_json
from flarenet.models import count_parameters
from flarenet.pipelines.reporting.calibration_plot import render_reliability
from flarenet.pipelines.reporting.dashboard import render_dashboard

log = get_logger(__name__)


def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(ci: dict, k: str) -> str:
    lo_hi = ci.get(k) if ci else None
    if not lo_hi or not all(np.isfinite(v) for v in lo_hi):
        return ""
    return f" [{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]"


def build_caveats(ctx: PipelineContext) -> list[str]:
    """Plain-language warnings derived from the data actually used. Quote these with the numbers."""
    out = []
    for name, res in ctx.state["eval_results"].items():
        b = ctx.state["bundles"][name]
        spec = ctx.cfg.source(name)
        if b.kind == "synthetic":
            out.append(
                f"[{name}] synthetic data: far more flare-active than the real Sun; use only as a pipeline sanity check, not as evidence of skill."
            )
        label_kind = b.meta.get("label_kind", "")
        if "relative" in label_kind or "proxy" in label_kind:
            out.append(
                f"[{name}] labels are a relative proxy ({label_kind}), not calibrated GOES classes, so scores are not comparable with GOES-based results."
            )
        n_mx = int((b.flare_db["class_idx"] >= 3).sum()) if len(b.flare_db) else 0
        if n_mx < 10:
            out.append(
                f"[{name}] only {n_mx} M/X events in the whole series: skill scores have very wide uncertainty."
            )
        if b.meta.get("hxr_substituted"):
            out.append(
                f"[{name}] no hard-channel data: the hard channel was replaced by the soft one, so hardness/derivative-of-hard channels carry no extra information."
            )
        if spec.role == "train" and res.kind == "test":
            out.append(
                f"[{name}] scored on held-out 3-day blocks of the same series used for training; this is not a test on a different period or instrument."
            )
    loaded = ctx.state.get("loaded")
    if loaded is None or "recent_flare_rate_6h" in loaded.channels:
        out.append(
            "Feature 'recent_flare_rate_6h' is computed from flare labels; run with features.channels excluding it for a purely flux-based model."
        )
    return out


class MetricsStage(Stage):
    name = "metrics"

    def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.cfg
        lm = ctx.state["loaded"]
        sources = {}
        for name in ctx.state["source_names"]:
            b = ctx.state["bundles"][name]
            spec = cfg.source(name)
            sources[name] = {
                "kind": spec.kind,
                "role": spec.role,
                "labeler": b.meta.get("labeler"),
                "scaling": b.meta.get("scaling"),
                "label_kind": b.meta.get("label_kind"),
                "units": b.meta.get("units"),
                "minutes": int(b.n_minutes),
                "span_days": round(b.n_minutes / 1440, 2),
                "n_events": b.meta.get("n_events"),
                "event_classes": b.meta.get("event_classes"),
                "cross_check": b.meta.get("cross_check"),
            }
        report = {
            "flarenet_version": __version__,
            "run_name": cfg.run_name,
            "config_fingerprint": fingerprint(cfg.to_dict()),
            "model": {
                "parameters": count_parameters(lm.model),
                "channels": lm.channels,
                "trained_on": lm.trained_on,
                **{k: v for k, v in cfg.model.__dict__.items()},
            },
            "window": {
                "window_min": cfg.data.window_min,
                "stride_min": cfg.data.stride_min,
                "horizon_min": cfg.data.horizon_min,
            },  # fmt: skip
            "alert_threshold": lm.threshold,
            "sources": sources,
            "results": {
                n: r.summary() | {"dataset": r.name} for n, r in ctx.state["eval_results"].items()
            },
            "caveats": build_caveats(ctx),
        }
        ctx.state["report"] = report
        save_json(report, ctx.run_dir / "metrics_report.json")
        for name, res in ctx.state["eval_results"].items():
            ctx.state["bundles"][name].flare_db.to_csv(
                ctx.run_dir / f"flare_db_{name}.csv", index=False
            )
            res.lead.to_csv(ctx.run_dir / f"lead_time_{name}.csv", index=False)


class MarkdownStage(Stage):
    name = "markdown"

    def run(self, ctx: PipelineContext) -> None:
        r = ctx.state["report"]
        lines = [f"# FlareNet run `{r['run_name']}`", "",
                 f"Alert threshold {r['alert_threshold']:.2f} | window {r['window']['window_min']} min | "
                 f"horizon {r['window']['horizon_min']} min | model {r['model']['name']} "
                 f"({r['model']['parameters']:,} parameters) | trained on {', '.join(r['model']['trained_on']) or 'n/a'}",
                 "", "| dataset | windows | M/X-positive | forecast AUC | TSS | HSS | POD | FAR | nowcast acc | macro-F1 | lead median (min) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for res in r["results"].values():
            ci = res["confidence_95"]
            lead = res["lead_time"]
            lines.append(
                f"| {res['dataset']} | {res['n_windows']} | {_fmt(res['forecast_positive_frac'])} | "
                f"{_fmt(res['AUC'])}{_ci(ci, 'AUC')} | {_fmt(res['TSS'])}{_ci(ci, 'TSS')} | "
                f"{_fmt(res['HSS'])}{_ci(ci, 'HSS')} | {_fmt(res['POD'])} | {_fmt(res['FAR'])} | "
                f"{_fmt(res['accuracy'])} | {_fmt(res['macro_F1'])} | {_fmt(lead.get('median_min'), 0)} |"
            )
        base_rows = []
        for res in r["results"].values():
            b = res.get("baselines") or {}
            if not b:
                continue
            fp, nr = b.get("flux_persistence", {}), b.get("nowcast_flux_rule", {})
            bss = b.get("model_brier_skill_vs_climatology")
            base_rows.append(
                f"| {res['dataset']} | {_fmt(res['TSS'])} | {_fmt(fp.get('TSS'))} | {_fmt(res['AUC'])} | "
                f"{_fmt(fp.get('AUC'))} | {_fmt(bss)} | {_fmt(res['accuracy'])} | "
                f"{_fmt(nr.get('accuracy'))} | {_fmt(res['macro_F1'])} | {_fmt(nr.get('macro_F1'))} |"
            )
        if base_rows:
            lines += [
                "", "## Against simple reference forecasters (same windows)", "",
                "| dataset | model TSS | 6 h flux-rule TSS | model AUC | flux-rule AUC | Brier skill vs climatology "
                "| model nowcast acc | flux-rule acc | model macro-F1 | flux-rule macro-F1 |",
                "|---|---|---|---|---|---|---|---|---|---|",
                *base_rows,
                "",
                "The flux rule needs GOES-calibrated flux: for sources with proxy labels (e.g. Aditya-L1 "
                "relative classes) its nowcast columns are not meaningful.",
            ]  # fmt: skip
        lines += [
            "",
            "## Data sources",
            "",
            "| source | kind | role | labels | days | events |",
            "|---|---|---|---|---|---|",
        ]
        for n, s in r["sources"].items():
            lines.append(
                f"| {n} | {s['kind']} | {s['role']} | {s['labeler']} | {s['span_days']} | {s['n_events']} |"
            )
        lines += ["", "## Read before quoting these numbers", ""] + [f"- {c}" for c in r["caveats"]]
        (ctx.run_dir / "report.md").write_text("\n".join(lines) + "\n")


class FigureStage(Stage):
    name = "figures"

    def run(self, ctx: PipelineContext) -> None:
        results = ctx.state["eval_results"]
        render_reliability(results, str(ctx.run_dir / "reliability.png"))
        hist_path = ctx.run_dir / "history.json"
        history = ctx.state.get("training", {}).get("history") or (
            json.loads(hist_path.read_text()) if hist_path.exists() else None
        )
        if not history or not history.get("train_loss"):
            log.info("No training history found: skipping the 12-panel dashboard")
            return
        tests = [n for n, r in results.items() if r.kind == "test"]
        exts = [n for n, r in results.items() if r.kind == "external"]
        first = (tests or exts)[0]
        second = exts[0] if tests and exts else None
        b = ctx.state["bundles"]
        render_dashboard(
            results[first], results[second] if second else None,
            b[first].df, b[first].flare_db,
            b[second].df if second else None, b[second].flare_db if second else None,
            history, ctx.state["threshold"], str(ctx.run_dir / "visual_dashboard.png"),
        )  # fmt: skip


def build_pipeline() -> Pipeline:
    return Pipeline("reporting", [MetricsStage(), MarkdownStage(), FigureStage()])

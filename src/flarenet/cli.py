"""Command line: ``flarenet <command>``. Run ``flarenet --help`` for the list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flarenet import __version__
from flarenet.core.config import load_config
from flarenet.core.utils import setup_logging


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c", "--config", default=None, help="YAML config (default: built-in synthetic run)"
    )
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set training.epochs=5 --set sources.0.params.n_days=30")  # fmt: skip
    p.add_argument("--run-name", default=None, help="run directory name under paths.runs_dir")
    p.add_argument("--out", default=None, help="run directory (overrides paths.runs_dir/run_name)")
    p.add_argument("--data-dir", default=None, help="cache for raw / processed / feature data")
    p.add_argument("--force", action="store_true", help="ignore caches and recompute")
    p.add_argument("--log-level", default="INFO")


def _aditya(p: argparse.ArgumentParser) -> None:
    p.add_argument("--solexs", default=None, help="add an Aditya-L1 eval source: SoLEXS SDD2 *.lc")
    p.add_argument(
        "--hel1os", nargs="+", default=None, help="... and its HEL1OS level-1 orbit dirs"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="flarenet", description="Solar flare nowcasting and forecasting."
    )
    ap.add_argument("--version", action="version", version=f"flarenet {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, helptext in [
        ("ingest", "pipeline 1: fetch raw data of every configured source into the cache"),
        ("preprocess", "pipelines 1-2: + resample, scale and label into 1-minute lightcurves"),
        ("features", "pipelines 1-3: + engineered channels, windows and the block split"),
    ]:
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument(
            "--source", action="append", default=None, help="only this source (repeatable)"
        )

    p = sub.add_parser("train", help="pipelines 1-4: train on the role=train sources")
    _common(p)
    p = sub.add_parser("evaluate", help="pipelines 1-3 + evaluation + reporting for a checkpoint")
    _common(p)
    _aditya(p)
    p.add_argument("--checkpoint", default=None, help="default: <run dir>/checkpoint.pt")
    p = sub.add_parser("run", help="everything: train, evaluate, report")
    _common(p)
    _aditya(p)
    p = sub.add_parser("predict", help="forecast over one source with a checkpoint")
    _common(p)
    _aditya(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--source",
        default=None,
        help="source name from the config (default: the first eval source)",
    )
    p.add_argument("--predictions", default=None, help="output CSV path")
    p = sub.add_parser(
        "noaa-compare",
        help="daily 24 h M/X forecast scored like NOAA's, optionally vs a NOAA probability CSV",
    )
    _common(p)
    p.add_argument("--checkpoint", required=True, help="trained with data.horizon_min=1440")
    p.add_argument(
        "--source", default=None, help="eval source to score (default: first eval source)"
    )
    p.add_argument(
        "--noaa-csv", default=None, help="CSV with date and prob_m (NOAA 24 h M-class probability)"
    )
    p.add_argument(
        "--allow-in-sample", action="store_true", help="score a role=train source (in-sample!)"
    )
    p = sub.add_parser(
        "daily",
        help="day-level M/X forecaster with sunspot-region features, scored against NOAA's own forecasts",
    )
    _common(p)
    p = sub.add_parser(
        "noaa-text",
        help="download NOAA's sunspot-region summaries and published forecasts for a date range and check they parse",
    )
    p.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    p.add_argument("--dir", default="data/downloads/noaa_text")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-level", default="INFO")
    p = sub.add_parser("monitor", help="live forecasting loop on the NOAA/NASA GOES X-ray feed")
    _common(p)
    p.add_argument("--checkpoint", required=True)

    p = sub.add_parser(
        "device", help="show available accelerators (CPU / Apple MPS / CUDA) and benchmark them"
    )
    p.add_argument("--steps", type=int, default=30, help="timed training steps per device")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cnn-channels", type=int, default=32, help="model width to benchmark")
    p.add_argument("--log-level", default="WARNING")
    p = sub.add_parser(
        "experiment",
        help="multi-seed A/B comparison of config variants (mean +- std, paired differences)",
    )
    p.add_argument(
        "-e", "--experiment", required=True, help="experiment YAML (seeds, baseline, variants)"
    )
    p.add_argument("-c", "--config", default=None, help="base config for every variant")
    p.add_argument("--out", default="runs/experiment")
    p.add_argument("--jobs", type=int, default=1, help="parallel runs (use 1 on a GPU)")
    p.add_argument(
        "--only", default=None, help="comma-separated variant names (baseline is always run)"
    )
    p.add_argument("--seeds", default=None, help="comma-separated seeds, overriding the file")
    p.add_argument("--device", default=None, help="force training.device for every run")
    p.add_argument("--log-level", default="INFO")
    p = sub.add_parser("sources", help="list registered data sources, catalogs, labelers, scalers")
    p.add_argument("--log-level", default="WARNING")
    p = sub.add_parser("donki", help="download the NASA DONKI flare list for a date range to CSV")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--api-key", default="DEMO_KEY")
    p.add_argument("--out", default="donki_flares.csv")
    p.add_argument("--log-level", default="INFO")
    return ap


def _config(args):
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "run_name", None):
        cfg.run_name = args.run_name
    if getattr(args, "out", None) and args.command != "donki":
        out = Path(args.out)
        cfg.paths.runs_dir, cfg.run_name = str(out.parent), out.name
    if getattr(args, "data_dir", None):
        cfg.paths.data_dir = args.data_dir
    if getattr(args, "solexs", None):
        if not args.hel1os:
            raise SystemExit("--solexs needs --hel1os as well")
        from flarenet.orchestration import add_aditya_source

        add_aditya_source(cfg, args.solexs, args.hel1os)
    cfg.validate()
    return cfg


def _list_registry() -> None:
    import flarenet.pipelines  # noqa: F401  (registers labelers)
    import flarenet.sources  # noqa: F401
    from flarenet.core.registry import CATALOGS, LABELERS, MODELS, SOURCES
    from flarenet.pipelines.preprocessing.scaling import SCALERS

    for title, reg in [("data sources", SOURCES), ("event catalogs", CATALOGS),
                       ("labelers", LABELERS), ("scaling policies", SCALERS), ("models", MODELS)]:  # fmt: skip
        print(f"{title}: {', '.join(reg.keys())}")
    print()
    for k in SOURCES.keys():
        c = SOURCES.get(k)
        print(f"  {k:16s} default labeler={c.default_labeler:20s} scaling={c.default_scaling}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "device":
        from flarenet.core.config import ModelConfig
        from flarenet.core.device import benchmark, describe_devices, format_report

        info = describe_devices()
        mcfg = ModelConfig(cnn_channels=args.cnn_channels)
        bench = benchmark(steps=args.steps, batch_size=args.batch_size, mcfg=mcfg)
        print(format_report(info, bench))
        return 0
    if args.command == "experiment":
        from flarenet.experiment import load_experiment, run_experiment

        spec = load_experiment(args.experiment)
        df = run_experiment(
            spec, args.out, args.config, args.jobs,
            args.only.split(",") if args.only else None,
            [int(x) for x in args.seeds.split(",")] if args.seeds else None,
            args.device,
        )  # fmt: skip
        print((Path(args.out) / "summary.md").read_text())
        return 0 if len(df) else 1
    if args.command == "sources":
        _list_registry()
        return 0
    if args.command == "noaa-text":
        from datetime import date as _date

        from flarenet.daily.noaa_text import fetch_range, load_rsga, load_srs

        d0, d1 = _date.fromisoformat(args.start), _date.fromisoformat(args.end)
        for kind in ("srs", "rsga"):
            files, st = fetch_range(kind, d0, d1, args.dir, args.workers)
            if kind == "srs":
                days, bad = load_srs(files)
                n_reg = sum(len(v) for v in days.values())
                print(
                    f"srs : {len(files)} files, {len(days)} parsed ({bad} unreadable), {st.missing} not published, {n_reg} region rows"
                )
            else:
                tab, bad = load_rsga(files)
                print(
                    f"rsga: {len(files)} files, {len(tab)} with readable forecasts ({bad} unreadable), {st.missing} not published"
                )
        return 0
    if args.command == "donki":
        import pandas as pd

        from flarenet.sources.catalogs import make_catalog

        t0 = pd.Timestamp(args.start, tz="UTC").timestamp()
        t1 = pd.Timestamp(args.end, tz="UTC").timestamp() + 86399
        df = make_catalog("donki", {"api_key": args.api_key}).fetch(t0, t1)
        df.to_csv(args.out, index=False)
        print(f"{len(df)} flares -> {args.out}")
        return 0

    cfg = _config(args)
    from flarenet import orchestration as orch

    if args.command == "ingest":
        orch.run_ingest(cfg, args.force, args.source)
    elif args.command == "preprocess":
        orch.run_preprocess(cfg, args.force, args.source)
    elif args.command == "features":
        orch.run_features(cfg, args.force, args.source)
    elif args.command == "train":
        orch.run_train(cfg, args.force)
    elif args.command == "evaluate":
        orch.run_evaluate(cfg, args.checkpoint, args.force)
    elif args.command == "run":
        orch.run_all(cfg, args.force)
    elif args.command == "predict":
        src = args.source or (cfg.eval_sources() or cfg.sources)[0].name
        orch.run_predict(cfg, args.checkpoint, src, args.predictions, args.force)
    elif args.command == "noaa-compare":
        res = orch.run_noaa_compare(cfg, args.checkpoint, args.source, args.noaa_csv,
                                    args.allow_in_sample, args.force)  # fmt: skip
        print(res["_markdown"])
    elif args.command == "daily":
        from flarenet.daily.pipeline import run_daily

        print(run_daily(cfg, args.force)["_markdown"])
    elif args.command == "monitor":
        from flarenet.pipelines.inference.monitor import run_monitor

        run_monitor(cfg, args.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())

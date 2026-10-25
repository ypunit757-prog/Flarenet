"""End-to-end runs of the pipelines on tiny data (a few seconds each, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import (
    goes_like,
    json_dump,
    swpc_records,
    tiny_cfg,
    write_fake_aditya,
    write_goes_days,
)

from flarenet import orchestration as orch
from flarenet.cli import main
from flarenet.core.config import SourceSpec
from flarenet.pipelines.inference.monitor import one_tick, run_monitor
from flarenet.pipelines.training.checkpoint import load_checkpoint

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("syn")
    cfg = tiny_cfg(tmp)
    ctx = orch.run_all(cfg)
    return cfg, ctx


def test_synthetic_run_writes_every_artifact(synthetic_run):
    cfg, _ = synthetic_run
    run = cfg.run_dir
    for name in ["checkpoint.pt", "history.json", "metrics_report.json", "report.md", "visual_dashboard.png",
                 "reliability.png", "flare_db_synthetic.csv", "config_used.json"] + [
                 f"manifest_{p}.json" for p in ("ingestion", "preprocessing", "features", "training", "evaluation", "reporting")]:  # fmt: skip
        assert (run / name).exists(), name
    rep = json.loads((run / "metrics_report.json").read_text())
    assert list(rep["sources"]) == ["synthetic"] and "synthetic" in rep["results"]
    assert any("synthetic data" in c for c in rep["caveats"])  # honesty about what the score means
    assert json.loads((run / "manifest_training.json").read_text())["status"] == "ok"


def test_second_run_hits_the_cache(synthetic_run, caplog):
    cfg, _ = synthetic_run
    with caplog.at_level("INFO"):
        orch.run_features(cfg)
    assert "cache hit" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO"):
        orch.run_features(cfg, force=True)
    assert "cache hit" not in caplog.text


def test_checkpoint_roundtrip_and_stage_failure_is_named(synthetic_run, tmp_path):
    cfg, _ = synthetic_run
    loaded = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    assert 0 < loaded.threshold < 1 and len(loaded.channels) == 12
    bad = tiny_cfg(
        tmp_path,
        [
            SourceSpec(
                name="x", kind="nasa_goes_file", params={"paths": [str(tmp_path / "none.nc")]}
            )
        ],
    )
    from flarenet.core.pipeline import PipelineError

    with pytest.raises(PipelineError, match="ingestion"):
        orch.run_ingest(bad)
    assert json.loads((bad.run_dir / "manifest_ingestion.json").read_text())["status"] == "failed"


def test_goes_train_then_aditya_eval_then_predict_and_monitor(tmp_path):
    """The headline use case: train on NASA/NOAA GOES-format files, test on another instrument."""
    t, soft, hard, _ = goes_like(30, seed=5)
    write_goes_days(tmp_path / "goes", t, soft, hard)
    solexs, hel = write_fake_aditya(tmp_path / "al1")
    cfg = tiny_cfg(tmp_path, [
        SourceSpec(name="goes16", kind="nasa_goes_file", role="train", params={"paths": [str(tmp_path / "goes" / "*.nc")]}),
        SourceSpec(name="aditya", kind="aditya_l1", role="eval", params={"solexs_path": solexs, "hel1os_dirs": hel}),
    ])  # fmt: skip
    orch.run_all(cfg)
    rep = json.loads((cfg.run_dir / "metrics_report.json").read_text())
    assert set(rep["results"]) >= {"goes16", "aditya"}
    caveats = " ".join(rep["caveats"])
    assert (
        "relative proxy" in caveats or "proxy" in caveats
    )  # Aditya labels are not GOES-calibrated
    assert rep["model"]["trained_on"] == ["goes16"]

    ckpt = cfg.run_dir / "checkpoint.pt"
    ctx = orch.run_predict(cfg, ckpt, "aditya")
    preds = pd.read_csv(cfg.run_dir / "predictions_aditya.csv")
    assert list(preds.columns) == ["time_min", "time_utc", "p_mx_6h", "alert", "nowcast_class"]
    assert preds.p_mx_6h.between(0, 1).all() and preds.time_min.max() > 840 - 360
    assert (cfg.run_dir / "predictions_aditya.png").exists() and "predictions" in ctx.state

    # live monitor on a saved SWPC feed (no network): forecast the newest window, append to a log
    t2, s2, h2, _ = goes_like(2, seed=9)
    feed = json_dump(swpc_records(t2, s2, h2), tmp_path / "feed.json")
    spec = SourceSpec(
        name="live", kind="nasa_goes_swpc", role="eval", params={"json_path": str(feed)}
    )
    loaded = load_checkpoint(ckpt, "cpu")
    row = one_tick(cfg, loaded, "cpu", spec)
    assert 0 <= row["p_mx_6h"] <= 1 and row["nowcast_class"] in {"QUIET", "B", "C", "M", "X"}
    cfg.monitor.max_iterations, cfg.monitor.interval_s = 2, 0
    rows = run_monitor(cfg, ckpt, spec, sleep=lambda s: None)
    assert len(rows) == 2 and len(pd.read_csv(cfg.run_dir / cfg.monitor.alert_log)) == 2


def test_monitor_skips_ticks_with_too_little_history(synthetic_run, tmp_path):
    cfg, _ = synthetic_run
    t, s, h, _ = goes_like(1, seed=2)
    feed = json_dump(swpc_records(t[:100], s[:100], h[:100]), tmp_path / "short.json")
    spec = SourceSpec(
        name="live", kind="nasa_goes_swpc", role="eval", params={"json_path": str(feed)}
    )
    loaded = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    with pytest.raises(ValueError, match="min_context_min"):
        one_tick(cfg, loaded, "cpu", spec)


def test_feature_ablation_changes_model_input_and_still_evaluates(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg.features.channels = [c for c in __import__("flarenet.pipelines.features.channels", fromlist=["x"]).CHANNEL_NAMES
                             if c != "recent_flare_rate_6h"]  # fmt: skip
    orch.run_all(cfg)
    loaded = load_checkpoint(cfg.run_dir / "checkpoint.pt", "cpu")
    assert len(loaded.channels) == 11 and "recent_flare_rate_6h" not in loaded.channels
    rep = json.loads((cfg.run_dir / "metrics_report.json").read_text())
    assert not any(
        "recent_flare_rate_6h" in c and "computed from flare labels" in c for c in rep["caveats"]
    )


def test_finetune_from_checkpoint(synthetic_run, tmp_path):
    cfg0, _ = synthetic_run
    cfg = tiny_cfg(tmp_path)
    cfg.training.finetune_from = str(cfg0.run_dir / "checkpoint.pt")
    cfg.training.freeze_encoder = True
    orch.run_train(cfg)
    assert (cfg.run_dir / "checkpoint.pt").exists()


def test_cli_ingest_sources_and_run(tmp_path, capsys):
    assert main(["sources"]) == 0
    out = capsys.readouterr().out
    for kind in (
        "nasa_goes_ncei",
        "nasa_goes_swpc",
        "nasa_goes_file",
        "aditya_l1",
        "synthetic",
        "donki",
    ):
        assert kind in out
    base = ["--data-dir", str(tmp_path / "d"), "--out", str(tmp_path / "runs" / "r"), "--log-level", "WARNING",
            "--set", "sources.0.params.n_days=30", "--set", "data.block_days=2"]  # fmt: skip
    assert main(["ingest", *base]) == 0
    assert (tmp_path / "d" / "raw" / "synthetic" / "meta.json").exists()
    assert main(["features", *base]) == 0
    assert (tmp_path / "d" / "features" / "synthetic" / "windows.npz").exists()
    rc = main(["run", *base, "--set", "training.epochs=1", "--set", "training.device=cpu",
               "--set", "evaluation.bootstrap_n=0"])  # fmt: skip
    assert rc == 0 and (tmp_path / "runs" / "r" / "report.md").exists()
    assert np.isfinite(
        json.loads((tmp_path / "runs" / "r" / "metrics_report.json").read_text())["alert_threshold"]
    )

import sys

import torch

from flarenet.cli import main
from flarenet.core.device import available_devices, benchmark, describe_devices, format_report
from flarenet.core.utils import resolve_device


def test_describe_and_available_include_cpu():
    info = describe_devices()
    assert "cpu" in info["available"] and info["available"] == available_devices()
    assert {"torch", "mps_built", "mps_available", "cuda_available", "hints"} <= set(info)


def test_benchmark_cpu_runs_and_matches_itself():
    res = benchmark(["cpu"], steps=2, batch_size=16)
    assert res["cpu"]["ms_per_step"] > 0 and res["cpu"]["max_abs_diff"] == 0.0


def test_format_report_recommends_the_faster_device():
    bench = {"cpu": {"ms_per_step": 100.0, "windows_per_s": 2560, "max_abs_diff": 0.0},
             "mps": {"ms_per_step": 25.0, "windows_per_s": 10240, "max_abs_diff": 1e-6}}  # fmt: skip
    text = format_report(
        {"torch": "x", "platform": "p", "python": "3", "available": ["cpu", "mps"], "hints": []},
        bench,
    )
    assert "mps is 4.0x faster" in text
    bench["mps"]["ms_per_step"] = 400.0
    assert "CPU is faster" in format_report(
        {"torch": "x", "platform": "p", "python": "3", "available": ["cpu", "mps"], "hints": []},
        bench,
    )


def test_resolve_device_falls_back_to_cpu_when_mps_missing(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("mps").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_auto_prefers_mps_when_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_mps_fallback_env_is_set_on_macos():
    import os

    if sys.platform == "darwin":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_cli_device(capsys):
    assert main(["device", "--steps", "1", "--batch-size", "8"]) == 0
    assert "available devices" in capsys.readouterr().out


def test_checkpoint_loads_for_a_non_cpu_device(tmp_path):
    """Regression: load_checkpoint(path, 'mps') crashed turning the normaliser into NumPy because
    every tensor was mapped onto the GPU. The 'meta' device fails the same way without a GPU."""
    import numpy as np

    from flarenet.core.config import Config
    from flarenet.models.flarenet import build_model
    from flarenet.pipelines.training.checkpoint import load_checkpoint, save_checkpoint

    cfg = Config()
    model = build_model(cfg.model, in_channels=12)
    mu, sg = np.zeros((1, 12, 1), np.float32), np.ones((1, 12, 1), np.float32)
    import inspect

    params = list(inspect.signature(save_checkpoint).parameters)
    kwargs = {"path": tmp_path / "c.pt", "model": model, "mu": mu, "sg": sg, "threshold": 0.5,
              "cfg": cfg, "channels": [f"c{i}" for i in range(12)], "trained_on": ["x"]}  # fmt: skip
    save_checkpoint(**{k: v for k, v in kwargs.items() if k in params})
    loaded = load_checkpoint(tmp_path / "c.pt", "meta")
    assert isinstance(loaded.mu, np.ndarray) and loaded.mu.shape == (1, 12, 1)
    assert next(loaded.model.parameters()).device.type == "meta"
    cpu = load_checkpoint(tmp_path / "c.pt", "cpu")
    assert next(cpu.model.parameters()).device.type == "cpu"

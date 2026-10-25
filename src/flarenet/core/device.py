"""Which accelerator is available, and is it actually faster? (``flarenet device``)

Apple-silicon Macs expose the GPU to PyTorch as the ``mps`` device; NVIDIA GPUs as ``cuda``.
The training config picks one with ``training.device`` (``auto`` = cuda, then mps, then cpu).
"""

from __future__ import annotations

import platform
import time

import torch
import torch.nn.functional as F

from flarenet.core.config import ModelConfig
from flarenet.core.constants import N_CHANNELS


def available_devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append("mps")
    return out


def describe_devices() -> dict:
    mps = getattr(torch.backends, "mps", None)
    info = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(mps is not None and mps.is_built()),
        "mps_available": bool(mps is not None and mps.is_available()),
        "available": available_devices(),
    }
    hints = []
    if platform.system() == "Darwin":
        if platform.machine() != "arm64":
            hints.append(
                "Python is running as x86_64 (Rosetta). Install an arm64 Python (e.g. from python.org or Homebrew) to get MPS."
            )
        elif not info["mps_built"]:
            hints.append(
                "This PyTorch build has no MPS support. Reinstall: pip install --upgrade torch"
            )
        elif not info["mps_available"]:
            hints.append("MPS is built but unavailable: needs macOS 12.3+ on Apple silicon.")
    info["hints"] = hints
    return info


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _build(mcfg: ModelConfig, in_channels: int):
    from flarenet.models.flarenet import build_model

    return build_model(mcfg, in_channels=in_channels)


def _make_step(model, opt, xb, nb, fb):
    """One training step (forward, loss, backward, optimiser) as a zero-argument callable."""

    def step():
        model.train()
        now_logits, fc_logits, _ = model(xb)
        loss = F.cross_entropy(now_logits, nb) + F.binary_cross_entropy_with_logits(fc_logits, fb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


def benchmark(
    devices: list[str] | None = None,
    steps: int = 30,
    batch_size: int = 256,
    window: int = 90,
    mcfg: ModelConfig | None = None,
    seed: int = 0,
) -> dict[str, dict]:
    """Time full training steps (forward + backward + AdamW) of FlareNet on each device.

    Returns ``{device: {"ms_per_step": ..., "windows_per_s": ...}}`` plus, for non-CPU devices,
    ``max_abs_diff``: the largest difference between that device's outputs and the CPU's for
    identical weights and inputs (should be ~1e-4 or less; large values mean a numerics problem).
    """
    mcfg = mcfg or ModelConfig()
    devices = devices or available_devices()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, N_CHANNELS, window, generator=g)
    y_now = torch.randint(0, 5, (batch_size,), generator=g)
    y_fc = torch.randint(0, 2, (batch_size,), generator=g).float()

    torch.manual_seed(seed)
    reference = _build(mcfg, N_CHANNELS)
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    reference.eval()
    with torch.no_grad():
        ref_now, ref_fc, _ = reference(x[:32])

    results: dict[str, dict] = {}
    for name in devices:
        dev = torch.device(name)
        model = _build(mcfg, N_CHANNELS)
        model.load_state_dict(state)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        xb, nb, fb = x.to(dev), y_now.to(dev), y_fc.to(dev)

        step = _make_step(model, opt, xb, nb, fb)

        for _ in range(3):  # warm-up: kernel compilation / allocator
            step()
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        _sync(dev)
        dt = (time.perf_counter() - t0) / steps
        row = {"ms_per_step": dt * 1e3, "windows_per_s": batch_size / dt}

        model.load_state_dict(state)  # parity check on the untrained weights
        model.eval()
        with torch.no_grad():
            now, fc, _ = model(x[:32].to(dev))
        row["max_abs_diff"] = float(
            max((now.cpu() - ref_now).abs().max(), (fc.cpu() - ref_fc).abs().max())
        )
        results[name] = row
    return results


def format_report(info: dict, bench: dict[str, dict] | None) -> str:
    lines = [f"torch {info['torch']} | {info['platform']} | python {info['python']}",
             f"available devices: {', '.join(info['available'])}"]  # fmt: skip
    lines += [f"NOTE: {h}" for h in info["hints"]]
    if bench:
        lines.append("")
        lines.append(f"{'device':8s} {'ms/step':>9s} {'windows/s':>11s} {'max |diff| vs cpu':>19s}")
        for name, r in bench.items():
            lines.append(
                f"{name:8s} {r['ms_per_step']:9.1f} {r['windows_per_s']:11.0f} {r['max_abs_diff']:19.2e}"
            )
        if len(bench) > 1:
            best = min(bench, key=lambda k: bench[k]["ms_per_step"])
            if best == "cpu":
                lines.append(
                    "\nCPU is faster for this model size. Set `--set training.device=cpu`, or try a larger batch size / model so the GPU has more work per step."
                )
            else:
                speed = bench["cpu"]["ms_per_step"] / bench[best]["ms_per_step"]
                lines.append(
                    f"\n{best} is {speed:.1f}x faster than cpu. Use `--set training.device={best}` (or leave device: auto)."
                )
    return "\n".join(lines)

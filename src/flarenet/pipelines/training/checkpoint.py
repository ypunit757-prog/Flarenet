"""Checkpoint = weights + normaliser + alert threshold + the config and channel list they need."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flarenet import __version__
from flarenet.core.config import Config
from flarenet.models import EnsembleModel, build_model


@dataclass
class LoadedModel:
    model: torch.nn.Module
    mu: np.ndarray
    sg: np.ndarray
    threshold: float
    config: dict
    channels: list[str]
    trained_on: list[str]
    # Per-class logit offsets added to the nowcast logits (tuned on validation), or None.
    now_bias: np.ndarray | None = None


def save_checkpoint(
    path, model, mu, sg, threshold, cfg: Config, channels, trained_on, now_bias=None
) -> None:
    payload = {
        "mu": torch.from_numpy(np.asarray(mu)),
        "sg": torch.from_numpy(np.asarray(sg)),
        "threshold": float(threshold),
        "config": cfg.to_dict(),
        "channels": list(channels),
        "trained_on": list(trained_on),
        "flarenet_version": __version__,
    }
    if isinstance(model, EnsembleModel):
        payload["members"] = [m.state_dict() for m in model.members]
    else:
        payload["state_dict"] = model.state_dict()
    if now_bias is not None:
        payload["now_bias"] = torch.as_tensor(np.asarray(now_bias, dtype=np.float32))
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device) -> LoadedModel:
    from flarenet.core.config import ModelConfig, build_dataclass

    # Load on the CPU first: mu/sg become NumPy arrays (impossible for tensors living on MPS/CUDA),
    # and only the model is moved to the requested device afterwards.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    mcfg = build_dataclass(ModelConfig, ckpt["config"]["model"])
    n_ch = len(ckpt["channels"])
    if "members" in ckpt:
        members = []
        for sd in ckpt["members"]:
            m = build_model(mcfg, in_channels=n_ch)
            m.load_state_dict(sd)
            members.append(m)
        model = EnsembleModel(members)
    else:
        model = build_model(mcfg, in_channels=n_ch)
        model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    bias = ckpt.get("now_bias")
    return LoadedModel(
        model, ckpt["mu"].numpy(), ckpt["sg"].numpy(), float(ckpt["threshold"]),
        ckpt["config"], list(ckpt["channels"]), list(ckpt.get("trained_on", [])),
        None if bias is None else bias.numpy(),
    )  # fmt: skip

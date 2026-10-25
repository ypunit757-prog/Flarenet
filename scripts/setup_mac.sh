#!/usr/bin/env bash
# One-shot setup for an Apple-silicon Mac using uv. Run from the project folder:  bash scripts/setup_mac.sh
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ "$(uname -m)" = "arm64" ] || { echo "Not arm64 (Rosetta?) - MPS needs a native Apple-silicon Python"; exit 1; }
uv sync --python 3.12 --extra netcdf      # creates .venv, installs everything from uv.lock (MPS-enabled torch)
uv run flarenet device                     # shows whether the GPU (mps) is available and how fast it is
echo "Done. Train on the GPU with:  uv run flarenet run -c configs/mac_mps.yaml"

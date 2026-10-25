# Running on a Mac GPU (Apple MPS)

FlareNet uses PyTorch's `mps` device (the Apple-silicon GPU). With the default `training.device: auto`
it is chosen automatically (order: CUDA, then MPS, then CPU) and the training log shows
`device=mps`.

## 0. Fastest: uv (recommended)

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
cd flarenet
uv sync --python 3.12 --extra netcdf      # creates .venv and installs every dependency from uv.lock
uv run flarenet device                    # confirms the GPU: "available devices: cpu, mps"
uv run flarenet run -c configs/mac_mps.yaml
```
`bash scripts/setup_mac.sh` does the first three steps. Python 3.12 is pinned because every dependency
has Apple-silicon wheels for it. Use `uv run <command>` (or `source .venv/bin/activate`) for everything.

## 1. (alternative, pip) Set up (Apple-silicon Mac, macOS 12.3 or newer)

```bash
python3 -c "import platform; print(platform.machine())"     # must print arm64, not x86_64
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,netcdf]"        # the standard macOS torch wheel includes MPS
```

## 2. Check that MPS works, and whether it is faster

```bash
flarenet device
```

It prints the available devices, then times real training steps on each (CPU vs MPS), checks that
the GPU gives the same outputs as the CPU for identical weights (`max |diff|` should be about 1e-4
or smaller), and tells you which to use. Example shape of the output:

```
available devices: cpu, mps
device     ms/step   windows/s   max |diff| vs cpu
cpu           ...
mps           ...
```

## 3. Train on the GPU

```bash
flarenet run                                   # auto picks mps
flarenet run --set training.device=mps         # force it
flarenet run --set training.device=cpu         # force CPU
```

## Things to know

* **Small model, small win.** FlareNet has about 122k parameters and uses LSTMs. At the default batch
  size the GPU often gains little, and CPU can even win; the GPU pays off with more work per step:
  `--set training.batch_size=512` (or 1024) and/or `--set model.cnn_channels=64`. Run
  `flarenet device --batch-size 512 --cnn-channels 64` to measure that setting before a long run.
* **Unsupported ops fall back to CPU.** On macOS FlareNet sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
  automatically, so a missing MPS kernel slows that op down instead of crashing the run.
* **Float32 only.** MPS has no float64; all model tensors in this project are float32.
* **Results differ slightly between devices** (floating-point order of operations), so a CPU run
  and an MPS run with the same seed will not match bit for bit. Compare metrics, not weights.
* **Checkpoints are portable.** A checkpoint trained on MPS loads on CPU/CUDA and the reverse.
* **Out of memory / "MPS backend out of memory":** lower `training.batch_size`.
* **DataLoader workers:** leave at the default; macOS starts workers with `spawn`, which is slower
  than it helps for data this small.

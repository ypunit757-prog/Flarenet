"""Typed configuration: dataclasses loaded from YAML with ``--set a.b.c=value`` overrides.

The ``sources`` list is what makes the project multi-instrument: each entry names a registered
data-source ``kind`` (synthetic, aditya_l1, nasa_goes_swpc, nasa_goes_ncei, nasa_goes_file, ...),
a ``role`` (``train`` = split into train/val/test by time blocks, ``eval`` = never trained on,
scored on the whole series) and kind-specific ``params``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

ROLES = ("train", "eval")


@dataclass
class SourceSpec:
    name: str = "synthetic"
    kind: str = "synthetic"
    role: str = "train"  # train | eval
    params: dict[str, Any] = field(default_factory=dict)
    labeler: str | None = None  # None -> the source kind's default labeler
    labeler_params: dict[str, Any] = field(default_factory=dict)
    scaling: str | None = (
        None  # native | background_normalised | goes_calibrated; None -> source default
    )
    scaling_params: dict[str, Any] = field(default_factory=dict)  # options of the scaling policy
    weight: float = 1.0  # relative sampling weight when several train sources are mixed


@dataclass
class PathsConfig:
    data_dir: str = "data"  # raw / processed / features caches
    runs_dir: str = "runs"  # one sub-directory per run


@dataclass
class DataConfig:
    window_min: int = 90
    stride_min: int = 60
    horizon_min: int = 360
    block_days: int = 3
    train_frac: float = 0.70
    val_frac: float = 0.15
    max_missing_frac: float = 1.0  # drop windows with more than this share of missing samples
    # Stride for the TRAIN windows of role=train sources (None = stride_min). A finer stride gives
    # more (overlapping) training windows; val/test keep the stride_min grid so scores stay comparable.
    train_stride_min: int | None = None
    # Drop train/val windows whose input or forecast span overlaps a window of another split.
    purge: bool = False


@dataclass
class FeatureConfig:
    # Subset of features.channels.CHANNEL_NAMES to feed the model; None = all 12.
    channels: list[str] | None = None
    # Instrument-independent levels: log-flux channels are measured relative to a trailing quiet
    # baseline (the ``baseline_quantile`` of the last ``baseline_min`` minutes) instead of in
    # absolute units, so a different detector gain / background does not move the inputs.
    relative: bool = False
    baseline_min: int = 1440
    baseline_quantile: float = 0.1


@dataclass
class ModelConfig:
    name: str = "flarenet"
    cnn_channels: int = 32
    nowcast_hidden: int = 32
    forecast_hidden: int = 40
    dropout: float = 0.35
    # Nowcast head: feed the newest minutes' raw (normalised) features straight to the classifier,
    # and read both LSTM directions at the end they have actually seen the whole window.
    nowcast_skip: bool = False
    nowcast_pool: str = "last"  # last | final_states


@dataclass
class TrainConfig:
    epochs: int = 35
    patience: int = 8
    batch_size: int = 256
    lr: float = 5e-4
    min_lr: float = 5e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.08
    focal_gamma: float = 2.0
    forecast_loss_weight: float = 2.5
    positive_oversample: float = 3.0
    grad_clip: float = 5.0
    device: str = "auto"
    # Fine-tuning: start from a checkpoint (e.g. trained on synthetic data) instead of scratch.
    finetune_from: str | None = None
    freeze_encoder: bool = False
    refit_normaliser: bool = True  # False keeps the checkpoint's normaliser when fine-tuning
    # --- improvements (defaults reproduce the original behaviour) ---
    ema_decay: float = 0.0  # >0: evaluate / keep an exponential moving average of the weights
    select_metric: str = "tss"  # early stopping / checkpoint choice: tss | auc | tss_auc
    forecast_balance: str = "fixed"  # fixed = positive_oversample + capped pos_weight; auto = 50/50
    nowcast_weight_power: float = 0.5  # class weight = (inverse frequency) ** power
    nowcast_weight_cap: float = 6.0
    nowcast_bias_tuning: bool = False  # tune per-class logit offsets on val for macro-F1
    threshold_smooth: int = 0  # smooth val TSS over +-N threshold steps before taking the max
    ensemble_size: int = 1  # members trained with different seeds, probabilities averaged
    # --- instrument-robustness augmentation (train windows only; 0 = off) ---
    aug_level_jitter: float = 0.0  # sigma (log10 units) of a random per-window flux-level shift
    aug_hxr_dropout: float = 0.0  # probability a window's hard channel is replaced by the soft one
    aug_noise_max: float = 0.0  # extra Gaussian noise, sigma drawn uniformly from [0, this]


@dataclass
class EvalConfig:
    bootstrap_n: int = 200  # 0 disables confidence intervals
    bootstrap_block_hours: int = 72  # resample whole blocks so overlapping windows stay together
    calibration_bins: int = 10


@dataclass
class MonitorConfig:
    """Near-real-time forecasting loop (``flarenet monitor``)."""

    source_kind: str = "nasa_goes_swpc"
    product: str = "xrays-1-day"  # NOAA SWPC GOES X-ray JSON product
    satellite: str = "primary"  # primary | secondary
    interval_s: int = 60
    max_iterations: int | None = None  # None = run until interrupted
    alert_log: str = "alerts.csv"
    min_context_min: int = 450  # refuse to forecast with less history than this


@dataclass
class DailyConfig:
    """Settings of the day-level forecaster (``flarenet daily``), scored against NOAA's own forecasts."""

    target: str = "M"  # M | X: the flare class that must start during the target UTC day
    issue_hour_utc: int = (
        22  # NOAA issues its forecast at 22:00 UTC; features use only data before this
    )
    use_regions: bool = True  # add NOAA sunspot-region (SRS) features
    compare_noaa: bool = (
        True  # download NOAA's published forecasts (RSGA) and score on the same days
    )
    text_dir: str | None = (
        None  # cache for NOAA text files (default <data_dir>/downloads/noaa_text)
    )
    workers: int = 4
    mode: str = "walk_forward"  # walk_forward: every test block is predicted by a model trained on earlier days
    # train_eval: train on role=train sources, test on role=eval sources
    first_test: str | None = (
        None  # first test date (ISO); default: min_train_days after the data starts
    )
    min_train_days: int = 730
    step_days: int = 90  # length of each walk-forward test block
    gap_days: int = 3  # days left out between training and test (flare days cluster in time)
    blend_gbm: float = 0.5
    calibration: str = "platt"  # platt | isotonic | none
    oof_blocks: int = 5
    ablate: bool = True  # also score flux-only / +history / +rotation / +regions feature sets
    min_coverage: float = 0.6  # skip days whose flux data cover less than this share
    bootstrap_n: int = 1000
    bootstrap_block_days: int = 14


@dataclass
class Config:
    seed: int = 42
    run_name: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    daily: DailyConfig = field(default_factory=DailyConfig)
    sources: list[SourceSpec] = field(default_factory=lambda: [SourceSpec(params={"n_days": 200})])

    # ----- convenience -------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.paths.runs_dir) / self.run_name

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source(self, name: str) -> SourceSpec:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(f"No source named {name!r}. Configured: {[s.name for s in self.sources]}")

    def train_sources(self) -> list[SourceSpec]:
        return [s for s in self.sources if s.role == "train"]

    def eval_sources(self) -> list[SourceSpec]:
        return [s for s in self.sources if s.role == "eval"]

    def validate(self) -> None:
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"Source names must be unique, got {names}")
        for s in self.sources:
            if s.role not in ROLES:
                raise ValueError(f"Source {s.name!r}: role must be one of {ROLES}, got {s.role!r}")
            if not s.name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Source name {s.name!r} must be alphanumeric, '-' or '_'")
        d = self.data
        if not 0 < d.train_frac < 1 or not 0 < d.val_frac < 1 or d.train_frac + d.val_frac >= 1:
            raise ValueError("data.train_frac and data.val_frac must leave room for a test split")
        if d.window_min < 8 or d.stride_min < 1 or d.horizon_min < 1:
            raise ValueError("data.window_min must be >= 8, stride_min and horizon_min >= 1")
        if d.train_stride_min is not None and not 1 <= d.train_stride_min <= d.stride_min:
            raise ValueError("data.train_stride_min must be between 1 and data.stride_min")
        if self.features.baseline_min < 60 or not 0.0 < self.features.baseline_quantile < 0.5:
            raise ValueError(
                "features.baseline_min must be >= 60 and baseline_quantile in (0, 0.5)"
            )
        ta = self.training
        if ta.aug_level_jitter < 0 or ta.aug_noise_max < 0 or not 0.0 <= ta.aug_hxr_dropout <= 1.0:
            raise ValueError("training.aug_* must be >= 0 and aug_hxr_dropout in [0, 1]")
        if self.features.relative and ta.aug_level_jitter > 0:
            raise ValueError(
                "training.aug_level_jitter has no meaning with features.relative=true "
                "(relative levels are already independent of the detector gain)"
            )
        if self.model.nowcast_pool not in ("last", "final_states"):
            raise ValueError("model.nowcast_pool must be 'last' or 'final_states'")
        t = self.training
        for key, allowed in (("select_metric", ("tss", "auc", "tss_auc")),
                             ("forecast_balance", ("fixed", "auto"))):  # fmt: skip
            if getattr(t, key) not in allowed:
                raise ValueError(
                    f"training.{key} must be one of {allowed}, got {getattr(t, key)!r}"
                )
        if not 0.0 <= t.ema_decay < 1.0:
            raise ValueError("training.ema_decay must be in [0, 1)")
        if t.ensemble_size < 1 or t.threshold_smooth < 0:
            raise ValueError("training.ensemble_size must be >= 1 and threshold_smooth >= 0")
        dl = self.daily
        for key, allowed in (("target", ("M", "X")), ("mode", ("walk_forward", "train_eval")),
                             ("calibration", ("platt", "isotonic", "none"))):  # fmt: skip
            if getattr(dl, key) not in allowed:
                raise ValueError(f"daily.{key} must be one of {allowed}, got {getattr(dl, key)!r}")
        if not 0 <= dl.issue_hour_utc <= 23 or not 0.0 <= dl.blend_gbm <= 1.0:
            raise ValueError("daily.issue_hour_utc must be 0..23 and daily.blend_gbm in [0, 1]")
        if dl.step_days < 7 or dl.min_train_days < 60 or dl.oof_blocks < 2 or dl.gap_days < 0:
            raise ValueError(
                "daily.step_days >= 7, min_train_days >= 60, oof_blocks >= 2, gap_days >= 0"
            )


# ----- coercion / override machinery -------------------------------------------------------


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value: ints and floats (incl. 1e-3) first, then YAML (bool, null, lists)."""
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def coerce(current: Any, value: Any, key: str) -> Any:
    """Match the type of the existing default. YAML 1.1 reads '1e-3' as a string, and '5' for a
    float field as an int, so numeric fields are converted explicitly."""
    if isinstance(current, bool) or value is None:
        return value
    if isinstance(current, float) and isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Config key {key!r} expects a number, got {value!r}") from e
    if isinstance(current, int) and isinstance(value, (float, str)):
        try:
            as_float = float(value)
        except ValueError as e:
            raise ValueError(f"Config key {key!r} expects an integer, got {value!r}") from e
        if as_float != int(as_float):
            raise ValueError(f"Config key {key!r} expects an integer, got {value!r}")
        return int(as_float)
    return value


def build_dataclass(cls: type, values: dict[str, Any] | None, where: str = "") -> Any:
    """Instantiate dataclass ``cls`` from a dict, rejecting unknown keys and coercing numbers."""
    obj = cls()
    _merge(obj, values or {}, where)
    return obj


def _merge(obj: Any, values: dict[str, Any], path: str = "") -> None:
    valid = {f.name for f in fields(obj)}
    for key, val in values.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if key == "sources" and isinstance(obj, Config):
            setattr(obj, key, [_source_from_dict(v, i) for i, v in enumerate(val or [])])
        elif is_dataclass(current) and isinstance(val, dict):
            _merge(current, val, f"{path}{key}.")
        else:
            setattr(obj, key, coerce(current, val, f"{path}{key}"))


def _plain(v: Any) -> Any:
    """YAML turns an unquoted 2024-05-01 into a date object; keep dates as ISO strings so
    configs stay JSON-serialisable (run manifests, checkpoints) and sources get text."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _source_from_dict(d: dict[str, Any], idx: int) -> SourceSpec:
    if not isinstance(d, dict):
        raise ValueError(f"sources[{idx}] must be a mapping, got {type(d).__name__}")
    spec = SourceSpec()
    _merge(spec, _plain(d), f"sources[{idx}].")
    if "name" not in d:
        spec.name = spec.kind
    return spec


def _step(cur: Any, part: str, key: str) -> Any:
    if is_dataclass(cur):
        if not hasattr(cur, part):
            raise KeyError(f"Unknown config section: {part} (in {key})")
        return getattr(cur, part)
    if isinstance(cur, list):
        return cur[int(part)]
    if isinstance(cur, dict):
        return cur.setdefault(part, {})
    raise KeyError(f"Cannot descend into {type(cur).__name__} at {part!r} (in {key})")


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply 'a.b=value' strings, e.g. ['training.epochs=5', 'sources.0.params.n_days=30']."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        target: Any = cfg
        for p in parts[:-1]:
            target = _step(target, p, key)
        last, value = parts[-1], _plain(_parse_scalar(raw))  # 2021-01-01 -> ISO text, not a date
        if is_dataclass(target):
            if not hasattr(target, last):
                raise KeyError(f"Unknown config key: {key}")
            setattr(target, last, coerce(getattr(target, last), value, key))
        elif isinstance(target, dict):
            target[last] = coerce(target.get(last), value, key)
        elif isinstance(target, list):
            target[int(last)] = value
        else:
            raise KeyError(f"Cannot set {key}")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load defaults, then a YAML file (if given), then CLI overrides; validate."""
    cfg = Config()
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _merge(cfg, raw)
    apply_overrides(cfg, overrides)
    cfg.validate()
    return cfg

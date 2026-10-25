from pathlib import Path

import numpy as np
import pytest

from flarenet.core.config import Config, SourceSpec, apply_overrides, load_config
from flarenet.sources.synthetic import generate_synthetic

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# default.yaml switches the validated improvements on; the code defaults and fast.yaml keep the
# original, cheap behaviour. These are the ONLY intended differences between the two.
DEFAULT_YAML_IMPROVEMENTS = {
    ("data", "train_stride_min"): 30,
    ("training", "ensemble_size"): 3,
    ("training", "nowcast_bias_tuning"): True,
}


def _flat(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, (*prefix, k))
        else:
            yield (*prefix, k), v


def test_default_yaml_differs_from_code_defaults_only_by_the_improvements():
    yaml_cfg = dict(_flat(load_config(CONFIGS / "default.yaml").to_dict()))
    code_cfg = dict(_flat(Config().to_dict()))
    diff = {k: yaml_cfg[k] for k in yaml_cfg if yaml_cfg[k] != code_cfg.get(k)}
    assert diff == DEFAULT_YAML_IMPROVEMENTS


def test_fast_yaml_is_the_original_behaviour():
    fast = load_config(CONFIGS / "fast.yaml").to_dict()
    assert fast.pop("run_name") == "fast"
    default = Config().to_dict()
    default.pop("run_name")
    assert fast == default


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads_and_validates(path):
    cfg = load_config(path)
    assert cfg.sources and cfg.train_sources() + cfg.eval_sources() == cfg.sources


def test_overrides_parse_types_and_descend_into_sources():
    cfg = apply_overrides(
        Config(),
        ["training.epochs=3", "training.lr=1e-3", "seed=9", "sources.0.params.n_days=12",
         "features.channels=[log_sxr,hardness]", "training.finetune_from=null"],
    )  # fmt: skip
    assert cfg.training.epochs == 3 and cfg.training.lr == pytest.approx(1e-3) and cfg.seed == 9
    assert cfg.sources[0].params["n_days"] == 12 and cfg.features.channels == [
        "log_sxr",
        "hardness",
    ]
    assert cfg.training.finetune_from is None


def test_unknown_keys_raise():
    for bad in ("training.nope=1", "nosection.x=1"):
        with pytest.raises(KeyError):
            apply_overrides(Config(), [bad])
    with pytest.raises(ValueError):
        apply_overrides(Config(), ["missing_equals"])


def test_validation_errors():
    cfg = Config()
    cfg.sources = [SourceSpec(name="a"), SourceSpec(name="a")]
    with pytest.raises(ValueError, match="unique"):
        cfg.validate()
    cfg.sources = [SourceSpec(name="a", role="bogus")]
    with pytest.raises(ValueError, match="role"):
        cfg.validate()
    cfg = Config()
    cfg.data.train_frac = 0.95
    with pytest.raises(ValueError, match="test split"):
        cfg.validate()


def test_numeric_coercion_from_yaml_strings_and_ints(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("training:\n  lr: 1e-3\n  min_lr: 1\n  epochs: 4.0\n")
    cfg = load_config(f)
    assert cfg.training.lr == pytest.approx(1e-3) and isinstance(cfg.training.min_lr, float)
    assert cfg.training.epochs == 4 and isinstance(cfg.training.epochs, int)
    with pytest.raises(ValueError):
        apply_overrides(Config(), ["training.epochs=2.5"])


def test_source_name_defaults_to_kind(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("sources:\n  - kind: nasa_goes_swpc\n    role: eval\n")
    assert load_config(f).sources[0].name == "nasa_goes_swpc"


def test_synthetic_is_deterministic_and_has_gaps_and_flares():
    a, ea = generate_synthetic(10, seed=1)
    b, eb = generate_synthetic(10, seed=1)
    assert np.array_equal(a.sxr.to_numpy(), b.sxr.to_numpy(), equal_nan=True)
    assert a.sxr.isna().any() and len(ea) == len(eb) > 0
    assert (
        set(ea["class"]) <= {"B", "C", "M", "X"}
        and len(a) == 10 * 1440
        and a.flare_class.max() >= 1
    )


def test_set_override_with_a_date_stays_json_serialisable():
    import json

    from flarenet.core.config import Config, apply_overrides

    cfg = Config()
    cfg.sources[0].kind = "nasa_goes_ncei"
    apply_overrides(cfg, ["sources.0.params.start=2021-01-01", "sources.0.params.end=2023-12-31"])
    assert cfg.sources[0].params["start"] == "2021-01-01"
    json.dumps(cfg.to_dict())

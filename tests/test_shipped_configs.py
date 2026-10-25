"""Every YAML in configs/ must load, validate and name only registered kinds/labelers."""

from pathlib import Path

import pytest

import flarenet.pipelines  # noqa: F401
import flarenet.sources  # noqa: F401
from flarenet.core.config import load_config
from flarenet.core.registry import SOURCES
from flarenet.core.utils import save_json

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads_validates_and_serialises(path, tmp_path):
    cfg = load_config(path)
    cfg.validate()
    for s in cfg.sources:
        assert s.kind in SOURCES, f"{path.name}: unknown kind {s.kind}"
        make_params = SOURCES.get(s.kind)(s.name, s.params)  # rejects unknown params
        assert make_params.params is not None
        for v in s.params.values():
            assert not hasattr(v, "isoformat"), "dates must be ISO strings"
    save_json(cfg.to_dict(), tmp_path / "c.json")  # YAML dates would break this


def test_ncei_example_dates_are_strings():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "nasa_goes_ncei.yaml")
    assert cfg.sources[0].params["start"] == "2024-03-01"

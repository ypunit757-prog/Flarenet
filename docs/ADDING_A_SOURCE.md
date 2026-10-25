# Adding a data source

A source fetches raw samples; everything else (1-minute grid, labels, features, training, reports)
is shared. Implement `fetch()`, register the class, reference it from a config.

```python
# src/flarenet/sources/my_mission.py
from dataclasses import dataclass
from pathlib import Path

from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.sources.base import DataSource, SourceUnavailable


@SOURCES.register("my_mission")
class MyMissionSource(DataSource):
    @dataclass
    class Params:  # validated; unknown keys in the config raise an error
        path: str | None = None

    default_labeler = "goes_events"  # calibrated W/m2 flux -> NOAA-style events
    # default_labeler = "relative_percentile"  + default_scaling = "background_normalised"
    # for count rates with no calibration (what aditya_l1 does)
    label_kind = "GOES-calibrated classes ..."
    units = "W/m2"
    soft_name, hard_name = "soft channel", "hard channel"

    def input_files(self):  # optional: file size/mtime join the cache fingerprint
        return [Path(self.params.path)] if self.params.path else []

    def fetch(self) -> RawSeries:
        if not self.params.path:
            raise SourceUnavailable(f"[{self.name}] params.path is required")
        t_unix, soft, hard = ...  # arrays, NaN where missing
        return RawSeries(
            self.name,
            self.kind,
            {"sxr": Channel(t_unix, soft), "hxr": Channel(t_unix, hard)},
            self.describe(),
        )
```

Import the module in `src/flarenet/sources/__init__.py`, then:

```yaml
sources:
  - name: my_run
    kind: my_mission
    role: eval
    params: {path: /data/file.fits}
```

Checks: `flarenet sources` lists it; `flarenet ingest -c my.yaml` fetches it; add a test that writes a
small fixture file in the real layout (see `tests/test_sources_nasa.py`).

Notes

* Fluxes in W/m2 for the 0.1-0.8 nm channel keep the model's input scale and the GOES class
  thresholds valid. Other units need `background_normalised` scaling or your own scaler.
* No hard channel? Omit `hxr`; the preprocessing stage substitutes the soft channel and the report
  says so. Consider an ablation without the hard-channel features.
* Remote data: raise `SourceUnavailable` with a message that tells the user what to do. Use
  `sources/nasa/http.py` (`http_get`, `download_file`, `list_directory`) for retries and atomic downloads.

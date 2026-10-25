import numpy as np
import pytest
from conftest import make_lightcurve

from flarenet.core.constants import CLASS_TO_IDX, N_CHANNELS
from flarenet.pipelines.features.channels import (
    CHANNEL_NAMES,
    fill_gaps,
    first_minute_flags,
    forward_any,
    select_channels,
)
from flarenet.pipelines.features.windows import build_windows


def test_fill_gaps_forward_fills_and_handles_leading_nan():
    a = np.array([np.nan, np.nan, 2.0, np.nan, 4.0, np.nan])
    out = fill_gaps(a)
    assert not np.isnan(out).any()
    assert out[3] == 2.0 and out[5] == 4.0
    assert out[0] == pytest.approx(3.0)  # nanmean of [2, 4]


def test_forward_any_looks_strictly_ahead():
    flags = np.zeros(10, dtype=bool)
    flags[5] = True
    out = forward_any(flags, 3)
    assert out[2] and out[4]
    assert not out[5] and not out[1]


def test_first_minute_flags_marks_true_escalation_not_event_start():
    """v5.1 fix: event starts as C, escalates to M later; the flag must land on the M minute."""
    fid = np.array([-1, 1, 1, 1, 1, -1, 2, 2, -1])
    cls = np.array([0, 2, 2, 3, 4, 0, 2, 2, 0])
    flags = first_minute_flags(fid, cls, CLASS_TO_IDX["M"])
    assert flags.tolist() == [False, False, False, True, False, False, False, False, False]


def _escalating_lightcurve(n=1000):
    df = make_lightcurve(n)
    df.loc[590:620, "flare_id"] = 0  # C at 590, escalates to M at 600
    df.loc[590:620, "flare_class"] = 2
    df.loc[600:620, "flare_class"] = 3
    return df


def test_build_windows_shapes_and_labels():
    ws = build_windows(_escalating_lightcurve(), window_min=90, stride_min=60, horizon_min=360)
    assert ws.X.shape[1:] == (N_CHANNELS, 90)
    assert ws.X.dtype == np.float32 and ws.y_now.dtype == np.int64
    assert len(ws) == len(ws.y_fc) == len(ws.t_end) == len(ws.missing_frac)
    assert ws.channels == CHANNEL_NAMES
    assert ws.y_fc[ws.t_end == 569][0] == 1.0  # sees the M onset at 600 within 360 min
    assert ws.y_fc[ws.t_end == 629][0] == 0.0  # already past it


def test_build_windows_for_inference_keeps_tail():
    df = make_lightcurve(800)
    train = build_windows(df, 90, 60, 360)
    inf = build_windows(df, 90, 60, 360, for_inference=True)
    assert len(inf) > len(train) and inf.t_end.max() > 800 - 360


def test_build_windows_rejects_short_series():
    with pytest.raises(ValueError):
        build_windows(make_lightcurve(300), 90, 60, 360)


def test_channel_subset_and_unknown_channel():
    names = ["log_sxr", "hardness"]
    ws = build_windows(make_lightcurve(700), channel_names=names)
    assert ws.X.shape[1] == 2 and ws.channels == names
    with pytest.raises(KeyError):
        select_channels([], ["nope"])
    with pytest.raises(ValueError):
        select_channels([], [])


def test_missing_fraction_and_filter():
    df = make_lightcurve(1000)
    df.loc[0:44, "sxr"] = np.nan  # half of the first 90-minute window
    ws = build_windows(df, 90, 60, 360)
    assert ws.missing_frac[0] == pytest.approx(0.5) and (ws.missing_frac[1:] == 0).all()
    kept = build_windows(df, 90, 60, 360, max_missing_frac=0.2)
    assert len(kept) == len(ws) - 1
    with pytest.raises(ValueError):
        build_windows(df.assign(sxr=np.nan), 90, 60, 360, max_missing_frac=0.1)

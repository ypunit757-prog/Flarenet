import numpy as np
import pandas as pd

from flarenet.pipelines.evaluation.lead_time import compute_lead_times, summarise_lead_times


def _db(onset=1000.0, end=1030.0, idx=3, eid=1):
    return pd.DataFrame(
        [
            {
                "event_id": eid,
                "class": "M",
                "class_idx": idx,
                "onset_min": onset,
                "end_min": end,
                "duration_min": end - onset,
            }
        ]
    )


TIMES = np.arange(0, 1100, 60, dtype=float)


def test_hit_lead_time_inside_horizon():
    probs = np.zeros_like(TIMES)
    probs[TIMES == 780] = 0.9  # alert 220 min before onset
    out = compute_lead_times(_db(), TIMES, probs, 0.5, max_lead_min=360)
    assert out.status.tolist() == ["HIT"]
    assert out.lead_time_min.iloc[0] == 220.0


def test_alert_before_horizon_is_not_credited():
    """An alert 16 h earlier must not count as lead time (it would be a false alarm for this event)."""
    probs = np.zeros_like(TIMES)
    probs[TIMES == 60] = 0.9
    out = compute_lead_times(_db(), TIMES, probs, 0.5, max_lead_min=360)
    assert out.status.tolist() == ["MISS"]


def test_late_alert_has_nonpositive_lead():
    probs = np.zeros_like(TIMES)
    probs[TIMES == 1020] = 0.9
    out = compute_lead_times(_db(), TIMES, probs, 0.5, max_lead_min=360)
    assert out.status.tolist() == ["LATE"]
    assert out.lead_time_min.iloc[0] <= 0


def test_events_without_forecast_coverage_are_excluded():
    probs = np.zeros_like(TIMES)
    out = compute_lead_times(_db(onset=50000.0, end=50030.0), TIMES, probs, 0.5)
    assert len(out) == 0


def test_only_m_and_x_events_are_scored():
    out = compute_lead_times(_db(idx=2), TIMES, np.ones_like(TIMES), 0.5)
    assert len(out) == 0


def test_summary():
    probs = np.zeros_like(TIMES)
    probs[TIMES == 780] = 0.9
    out = compute_lead_times(_db(), TIMES, probs, 0.5)
    s = summarise_lead_times(out)
    assert (
        s["n_events"] == 1
        and s["n_hits"] == 1
        and s["hit_rate"] == 1.0
        and s["median_min"] == 220.0
    )
    assert summarise_lead_times(pd.DataFrame())["n_events"] == 0

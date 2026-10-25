"""Synthetic SoLEXS/HEL1OS-like lightcurve generator with an active-region regime process.

Ground-truth flare labels come with the data (``RawSeries.aligned``), so the synthetic source
bypasses the label-from-signal step. Use it as a pipeline sanity check, not as evidence of skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flarenet.core.constants import CLASS_BANDS, CLASS_TO_IDX
from flarenet.core.registry import SOURCES
from flarenet.core.schema import Channel, RawSeries
from flarenet.sources.base import DataSource


def generate_synthetic(n_days: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ``n_days`` of 1-minute soft/hard X-ray lightcurves with labelled flares.

    Flares arrive as a regime-modulated process with memory: each flare raises a decaying
    "activity" level that makes stronger flares more likely shortly afterwards. Each flare
    has a Gaussian rise and exponential decay in soft X-rays and a bursty hard X-ray profile.
    Random data gaps are inserted to mimic telemetry dropouts.

    Returns
    -------
    df  : minute-resolution lightcurve with flare labels
    evs : table of generated events (class, onset, activity at onset)
    """
    rng = np.random.default_rng(seed)
    n = int(n_days * 24 * 60)
    t = np.arange(n, dtype=float)
    sxr = 3e-9 * (1 + 0.3 * np.sin(2 * np.pi * t / (24 * 60 * 27))) * rng.lognormal(0, 0.15, n)
    hxr = 5e-10 * rng.lognormal(0, 0.25, n)
    flare_class = np.zeros(n, dtype=int)
    flare_id = -np.ones(n, dtype=int)

    # Active-region regime: Poisson-like switches, ~36 h durations.
    regime = np.zeros(n, dtype=int)
    tc = 0.0
    while tc < n:
        tc += rng.exponential(1440 / 0.8)
        if tc >= n:
            break
        i0 = int(tc)
        i1 = min(n, int(tc + rng.exponential(36 * 60)))
        regime[i0:i1] = 1

    decay_rate = np.log(2) / 360.0
    activity_inc = {"B": 0.0, "C": 0.08, "M": 0.45, "X": 0.85}
    skew = {"B": 0.0, "C": 1.0, "M": 6.0, "X": 8.5}
    base_rate = {"B": 5.0, "C": 2.0, "M": 0.30, "X": 0.025}
    classes = list(base_rate.keys())
    base_w = np.array(list(base_rate.values()), float)
    skew_v = np.array([skew[c] for c in classes])

    n_cand = int(sum(base_rate.values()) * n_days * 4)
    cand = np.sort(rng.uniform(0, n, n_cand))
    regime_at = regime[np.clip(cand.astype(int), 0, n - 1)]
    keep_prob = (3.0 * regime_at + 1.0 * (1 - regime_at)) / 4.0
    onsets = cand[rng.random(n_cand) < keep_prob]

    events = []
    event_id = 0
    last_t = -1e9
    last_activity = 0.0
    for cur in onsets:
        dt = cur - last_t
        activity = last_activity * np.exp(-decay_rate * dt)
        regime_boost = 3.0 if regime[min(n - 1, int(cur))] else 0.0
        w = base_w * np.exp(skew_v * (activity + regime_boost * 0.1))
        w /= w.sum()
        cls = rng.choice(classes, p=w)
        lo, hi = CLASS_BANDS[cls]
        peak_flux = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
        t_rise = rng.uniform(2, 8)
        t_decay = rng.uniform(8, 40)
        lo_i = int(max(0, cur - 5))
        hi_i = int(min(n, cur + 8 * t_decay))
        if hi_i <= lo_i:
            continue
        tt = t[lo_i:hi_i]
        dtt = tt - cur
        soft = peak_flux * np.where(
            dtt < 0, np.exp(-(dtt**2) / (2 * t_rise**2)), np.exp(-dtt / t_decay)
        )

        hard_peak = rng.uniform(0.05, 0.6) * peak_flux * rng.uniform(2, 8)
        hard = np.zeros_like(tt)
        for _ in range(int(rng.integers(1, 5))):
            shift = rng.uniform(-2, 4)
            width = t_rise * rng.uniform(0.4, 1)
            hard += rng.uniform(0.3, 1) * np.exp(-((tt - cur - shift) ** 2) / (2 * width**2))
        if hard.max() > 0:
            hard = hard / hard.max() * hard_peak

        sxr[lo_i:hi_i] += soft
        hxr[lo_i:hi_i] += hard
        cid = CLASS_TO_IDX[cls]
        active = soft > 0.15 * peak_flux
        flare_class[lo_i:hi_i] = np.maximum(flare_class[lo_i:hi_i], np.where(active, cid, 0))
        flare_id[lo_i:hi_i] = np.where(active, event_id, flare_id[lo_i:hi_i])
        events.append({"class": cls, "onset_min": float(cur), "activity_at_onset": float(activity)})
        event_id += 1
        last_activity = min(1.0, activity + activity_inc[cls])
        last_t = cur

    # Slow background enhancement following activity, then noise and clipping.
    n_memory = np.log(1000.0) / decay_rate
    act_trace = np.zeros(n)
    for ev in events:
        amp = min(1.0, ev["activity_at_onset"] + activity_inc[ev["class"]])
        i0 = int(ev["onset_min"])
        i1 = min(n, int(ev["onset_min"] + n_memory))
        if i1 <= i0:
            continue
        act_trace[i0:i1] = np.maximum(
            act_trace[i0:i1], amp * np.exp(-decay_rate * np.arange(i1 - i0))
        )
    sxr = np.clip(sxr * (1 + 1.8 * act_trace) + rng.normal(0, 2e-10, n), 1e-10, None)
    hxr = np.clip(hxr * (1 + 0.9 * act_trace) + rng.normal(0, 5e-11, n), 1e-11, None)

    # Telemetry gaps.
    for _ in range(int(n_days / 7)):
        g0 = int(rng.integers(0, n - 60))
        glen = int(rng.integers(5, 60))
        sxr[g0 : g0 + glen] = np.nan
        hxr[g0 : g0 + glen] = np.nan

    df = pd.DataFrame(
        {
            "time_min": t,
            "sxr": sxr,
            "hxr": hxr,
            "flare_class": flare_class,
            "flare_id": flare_id,
            "active_regime": regime,
        }
    )
    df["sxr_raw"] = sxr
    return df, pd.DataFrame(events)


@SOURCES.register("synthetic")
class SyntheticSource(DataSource):
    @dataclass
    class Params:
        n_days: int = 200
        seed: int | None = None  # None -> the run seed
        t0_unix: float = 1_577_836_800.0  # 2020-01-01T00:00:00Z, arbitrary but minute-aligned

    default_labeler = "precomputed"
    default_scaling = "native"
    label_kind = "synthetic ground truth"
    units = "arb. (W/m2-like)"
    uses_seed = True
    soft_name = "synthetic soft X-ray"
    hard_name = "synthetic hard X-ray"

    def fetch(self) -> RawSeries:
        p = self.params
        seed = self.seed if p.seed is None else p.seed
        df, events = generate_synthetic(p.n_days, seed)
        t_unix = p.t0_unix + df["time_min"].to_numpy() * 60.0
        return RawSeries(
            source=self.name,
            kind=self.kind,
            channels={
                "sxr": Channel(t_unix, df["sxr"].to_numpy()),
                "hxr": Channel(t_unix, df["hxr"].to_numpy()),
            },
            meta={**self.describe(), "n_generated_events": int(len(events)), "seed": seed},
            aligned={
                "flare_class": df["flare_class"].to_numpy(),
                "flare_id": df["flare_id"].to_numpy(),
                "active_regime": df["active_regime"].to_numpy(),
            },
        )


@SOURCES.register("synthetic_instrument")
class SyntheticInstrumentSource(SyntheticSource):
    """The synthetic Sun seen through a *different detector*: known ground-truth flares, but the
    soft channel has another gain, response curve, background and noise, and the hard channel may
    be missing. Used to measure (and train for) instrument shift without needing real data from a
    second spacecraft. Results on it are simulated, never a statement about a real instrument.
    """

    @dataclass
    class Params:
        n_days: int = 200
        seed: int | None = None
        t0_unix: float = 1_577_836_800.0
        gain: float = 1.0  # multiplies the soft flux
        power: float = 1.0  # response nonlinearity: flux ** power (relative to the quiet level)
        background_frac: float = 0.0  # additive background, in units of the median soft flux
        noise_frac: float = 0.0  # sigma of multiplicative log-normal noise per sample
        hxr_mode: str = "keep"  # keep | none | scaled (hard channel gain-scaled with own noise)
        hxr_gain: float = 1.0

    def fetch(self) -> RawSeries:
        p = self.params
        if p.hxr_mode not in ("keep", "none", "scaled"):
            raise ValueError("hxr_mode must be keep, none or scaled")
        raw = super().fetch()
        rng = np.random.default_rng((self.seed if p.seed is None else p.seed) + 7919)
        t, s = raw.channels["sxr"].t_unix, raw.channels["sxr"].values.astype(float)
        quiet = float(np.nanmedian(s))
        s2 = quiet * (np.clip(s / quiet, 1e-9, None) ** p.power) + p.background_frac * quiet
        s2 = (
            p.gain * s2 * np.exp(rng.normal(0.0, p.noise_frac, s2.shape))
            if p.noise_frac
            else p.gain * s2
        )
        raw.channels["sxr"] = Channel(t, s2)
        if p.hxr_mode == "none":
            raw.channels.pop("hxr", None)
        elif p.hxr_mode == "scaled":
            h = raw.channels["hxr"].values.astype(float)
            h2 = p.hxr_gain * h * np.exp(rng.normal(0.0, max(p.noise_frac, 0.05), h.shape))
            raw.channels["hxr"] = Channel(raw.channels["hxr"].t_unix, h2)
        raw.meta.update(
            {
                "instrument_shift": {
                    k: getattr(p, k)
                    for k in (
                        "gain",
                        "power",
                        "background_frac",
                        "noise_frac",
                        "hxr_mode",
                        "hxr_gain",
                    )
                }
            }  # fmt: skip
        )
        return raw

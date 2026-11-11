"""Decide which alert events a new forecast produces, without repeating or inventing any.

Input is the ``forecast.json`` payload (see scripts/export_web.py), the state left by the previous run and the event history.
Output is the new state, the new history and the events that should be announced. Rules:

* No alert from unusable input: a demo payload, a payload whose newest data is older than ``max_age_h``, missing or non-finite
  probabilities. The state is left as it was, so a gap in the data never produces a flurry of alerts when data returns.
* One alert per episode: while the probability stays above the threshold nothing more is sent; a new episode within
  ``cooldown_min`` of the last announcement is recorded but not announced.
* An observed flare is announced once, by its peak time, and only while it is recent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

HISTORY_LIMIT = 500
LEVELS = ("sensitive", "balanced", "strict")
CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


@dataclass
class AlertConfig:
    level: str = (
        "balanced"  # sensitive | balanced | strict (the model's three validated thresholds)
    )
    threshold: float | None = None  # overrides the level when set (0 < threshold < 1)
    cooldown_min: float = (
        180.0  # a new episode this soon after the last announcement is logged, not announced
    )
    max_age_h: float = (
        3.0  # data older than this is "stale": never alert from it (same limit as the web page)
    )
    observed_min_class: str = "M"
    observed_recent_h: float = (
        3.0  # an observed flare is announced only if it peaked within this many hours
    )

    def validate(self) -> AlertConfig:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")
        if self.threshold is not None and not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        if self.observed_min_class not in ("C", "M", "X"):
            raise ValueError("observed_min_class must be C, M or X")
        return self


def _iso(u: float) -> str:
    return datetime.fromtimestamp(float(u), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def check_payload(payload: dict, now_unix: float, max_age_h: float = 3.0) -> tuple[bool, str]:
    """``(True, "")`` when the payload may be used for an alert, else ``(False, reason)``."""
    if not isinstance(payload, dict):
        return False, "the forecast payload is not an object"
    if payload.get("demo"):
        return False, "the payload is the simulated demo, not live data"
    try:
        t, p = payload["forecast"]["t"], payload["forecast"]["p"]
        end = _parse(payload["period"]["end"])
    except (KeyError, TypeError, ValueError):
        return False, "the payload lacks forecast times, probabilities or the end of the data"
    if len(t) < 2 or len(t) != len(p):
        return False, "fewer than 2 forecast points, or times and probabilities differ in length"
    last = p[-1]
    if (
        last is None
        or not isinstance(last, int | float)
        or not math.isfinite(last)
        or not 0.0 <= last <= 1.0
    ):
        return False, "the newest probability is missing or not a number between 0 and 1"
    age_h = (now_unix - end) / 3600.0
    if age_h > max_age_h:
        return False, f"the newest data is {age_h:.1f} h old (limit {max_age_h:g} h)"
    if age_h < -0.25:
        return False, "the newest data is in the future (clock or timestamp error)"
    return True, ""


def threshold_for(payload: dict, cfg: AlertConfig) -> tuple[float, str]:
    """The threshold in force and its name: an explicit value, else the model's level, else its default threshold."""
    if cfg.threshold is not None:
        return float(cfg.threshold), "custom"
    model = payload.get("model", {})
    levels = model.get("alert_levels") or {}
    if cfg.level in levels and levels[cfg.level]:
        return float(levels[cfg.level]), cfg.level
    return float(model["threshold"]), "default"


def episodes(t, p, threshold: float, horizon_min: float = 360.0, flares=(), now_unix: float | None = None,
             merge_gap_min: float = 60.0) -> list[dict]:  # fmt: skip
    """Runs of forecast times at or above ``threshold`` (runs less than ``merge_gap_min`` apart are one episode), each with
    its outcome: ``hit`` if an M/X flare peaked between its start and ``horizon_min`` after its end, ``false alarm`` if that
    window is over and none did, ``pending`` while the window is still open."""
    t = [float(x) for x in t]
    out: list[dict] = []
    cur = None
    for ti, pi in zip(t, p, strict=True):
        hot = pi is not None and math.isfinite(pi) and pi >= threshold
        if not hot:
            continue
        if cur is not None and ti - cur["end_u"] <= merge_gap_min * 60.0:
            cur["end_u"] = ti
            if pi > cur["peak_p"]:
                cur["peak_p"], cur["peak_u"] = float(pi), ti
        else:
            cur = {"start_u": ti, "end_u": ti, "peak_p": float(pi), "peak_u": ti}
            out.append(cur)
    peaks = []
    for f in flares:
        try:
            peaks.append((_parse(f["peak"]), f.get("class", "")))
        except (KeyError, ValueError, TypeError):
            continue
    for e in out:
        close = e["end_u"] + horizon_min * 60.0
        hits = [c for u, c in peaks if e["start_u"] <= u <= close]
        if hits:
            e["outcome"], e["flare"] = "hit", hits[0]
        elif now_unix is not None and now_unix < close:
            e["outcome"] = "pending"
        else:
            e["outcome"] = "false alarm"
        e.update(start=_iso(e["start_u"]), end=_iso(e["end_u"]), peak_time=_iso(e["peak_u"]))
        e["hours"] = round((e["end_u"] - e["start_u"]) / 3600.0, 2)
    return out


@dataclass
class Result:
    events: list[dict]
    notify: list[dict]
    state: dict
    history: list[dict]
    skipped: str | None = None


def process(payload: dict, state: dict | None, history: list[dict] | None, cfg: AlertConfig | None = None,
            now: datetime | None = None) -> Result:  # fmt: skip
    """Apply one new forecast. ``state`` and ``history`` are the values this function returned last time (or None/[])."""
    cfg = (cfg or AlertConfig()).validate()
    now_u = (now or datetime.now(timezone.utc)).timestamp()
    state = dict(state or {})
    history = list(history or [])
    ok, why = check_payload(payload, now_u, cfg.max_age_h)
    state["last_checked"] = _iso(now_u)
    if not ok:
        state["last_skipped"] = why
        return Result([], [], state, history, why)
    state.pop("last_skipped", None)

    thr, level = threshold_for(payload, cfg)
    p_now = float(payload["forecast"]["p"][-1])
    t_now = float(payload["forecast"]["t"][-1])
    in_alert = p_now >= thr
    known = {e["id"] for e in history}
    events: list[dict] = []
    notify: list[dict] = []
    base = {
        "threshold": round(thr, 3),
        "level": level,
        "calibrated": bool(payload.get("model", {}).get("calibrated")),
    }

    was = bool(state.get("in_alert"))
    if in_alert and not was:
        ev = {"id": f"risk-{_iso(t_now)}", "kind": "predicted_risk", "type": "started", "time": _iso(t_now),
              "p": round(p_now, 3), **base}  # fmt: skip
        last = state.get("last_announced")
        cool = last is not None and (now_u - _parse(last)) < cfg.cooldown_min * 60.0
        if cool:
            ev.update(
                notified=False, reason=f"within {cfg.cooldown_min:g} min of the previous alert"
            )
        else:
            ev["notified"] = True
            state["last_announced"] = _iso(now_u)
            notify.append(ev)
        state["episode_id"] = ev["id"]
        events.append(ev)
    elif was and not in_alert:
        events.append({"id": f"risk-end-{_iso(t_now)}", "kind": "predicted_risk", "type": "ended", "time": _iso(t_now),
                       "p": round(p_now, 3), "episode": state.get("episode_id"), **base, "notified": False})  # fmt: skip
        state["episode_id"] = None
    state["in_alert"] = in_alert

    floor = CLASS_ORDER[cfg.observed_min_class]
    for f in payload.get("flares", []):
        cls = str(f.get("class", ""))
        try:
            peak_u = _parse(f["peak"])
        except (KeyError, ValueError, TypeError):
            continue
        fid = f"obs-{f['peak']}"
        if (
            not cls
            or CLASS_ORDER.get(cls[0].upper(), -1) < floor
            or fid in known
            or any(e["id"] == fid for e in events)
        ):
            continue
        recent = 0 <= now_u - peak_u <= cfg.observed_recent_h * 3600.0
        ev = {"id": fid, "kind": "observed_flare", "type": "observed", "time": f["peak"], "class": cls,
              "forecast": f.get("forecast"), "lead_min": f.get("lead_min"), "notified": recent}  # fmt: skip
        if not recent:
            ev["reason"] = "peaked before the notification window; recorded only"
        else:
            notify.append(ev)
        events.append(ev)

    history.extend(events)
    history = history[-HISTORY_LIMIT:]
    return Result(events, notify, state, history)

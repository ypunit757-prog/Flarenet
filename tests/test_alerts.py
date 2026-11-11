"""Alert rules: never from unusable data, never twice for one episode, predicted risk never mistaken for an observed flare."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import process_alerts  # noqa: E402

from flarenet.alerts.engine import (
    HISTORY_LIMIT,
    AlertConfig,
    check_payload,
    episodes,
    process,
    threshold_for,
)  # noqa: E402
from flarenet.alerts.notify import (
    OBSERVED_NOTE,
    RISK_NOTE,
    build_message,
    send_email,
    smtp_settings,
)  # noqa: E402

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def payload(
    p_last=0.1, now=NOW, age_min=5, demo=False, flares=(), levels=None, calibrated=True, n=14
):
    end = now - timedelta(minutes=age_min)
    ts = [int((end - timedelta(minutes=30 * (n - 1 - i))).timestamp()) for i in range(n)]
    p = [0.05] * (n - 1) + [p_last]
    return {
        "demo": demo, "generated": iso(now), "period": {"end": iso(end)}, "model": {
            "threshold": 0.3, "calibrated": calibrated,
            "alert_levels": levels if levels is not None else {"sensitive": 0.15, "balanced": 0.3, "strict": 0.5},
        },
        "forecast": {"t": ts, "p": p, "alert": [int(x is not None and x >= 0.3) for x in p]}, "flares": list(flares),
    }  # fmt: skip


def flare(peak, cls="M2.1", forecast="WARNED", lead=95):
    return {
        "peak": iso(peak),
        "class": cls,
        "forecast": forecast,
        "lead_min": lead,
        "max_prob": 0.4,
    }


# ---------------------------------------------------------------- unusable input


@pytest.mark.parametrize("bad, why", [
    ({"demo": True}, "demo"),
    ({"age_min": 5 * 60}, "old"),
    ({"age_min": -120}, "future"),
    ({"p_last": float("nan")}, "number"),
    ({"p_last": None}, "number"),
    ({"p_last": 1.5}, "number"),
    ({"n": 1}, "fewer"),
])  # fmt: skip
def test_unusable_payloads_are_refused_with_a_reason(bad, why):
    ok, reason = check_payload(payload(**bad), NOW.timestamp())
    assert not ok and why in reason
    assert (
        not check_payload({"forecast": 1}, NOW.timestamp())[0]
        and not check_payload("x", NOW.timestamp())[0]
    )
    assert check_payload(payload(), NOW.timestamp()) == (True, "")


def test_nothing_changes_when_the_data_is_unusable():
    state = {"in_alert": True, "episode_id": "risk-x", "last_announced": "2026-11-10T09:00:00Z"}
    res = process(payload(p_last=0.9, demo=True), state, [], now=NOW)
    assert res.events == [] and res.notify == [] and res.skipped and res.state["in_alert"] is True
    stale = process(payload(p_last=0.9, age_min=400), state, [], now=NOW)
    assert stale.events == [] and "old" in stale.skipped


# ---------------------------------------------------------------- thresholds


def test_threshold_follows_a_custom_value_then_the_level_then_the_default():
    pl = payload()
    assert threshold_for(pl, AlertConfig(threshold=0.42)) == (0.42, "custom")
    assert threshold_for(pl, AlertConfig(level="strict")) == (0.5, "strict")
    assert threshold_for(payload(levels={}), AlertConfig(level="strict")) == (0.3, "default")
    with pytest.raises(ValueError):
        AlertConfig(level="loud").validate()
    with pytest.raises(ValueError):
        AlertConfig(threshold=1.2).validate()


# ---------------------------------------------------------------- one alert per episode


def test_an_alert_is_announced_once_per_episode():
    r1 = process(payload(p_last=0.6), {}, [], now=NOW)
    assert [e["type"] for e in r1.events] == ["started"] and len(r1.notify) == 1
    ev = r1.events[0]
    assert (
        ev["kind"] == "predicted_risk"
        and ev["notified"]
        and ev["level"] == "balanced"
        and ev["threshold"] == 0.3
    )

    r2 = process(
        payload(p_last=0.7, now=NOW + timedelta(minutes=30)),
        r1.state,
        r1.history,
        now=NOW + timedelta(minutes=30),
    )
    assert r2.events == [] and r2.notify == []  # still the same episode: no repeat

    r3 = process(
        payload(p_last=0.1, now=NOW + timedelta(minutes=60)),
        r2.state,
        r2.history,
        now=NOW + timedelta(minutes=60),
    )
    assert (
        [e["type"] for e in r3.events] == ["ended"]
        and r3.notify == []
        and r3.state["in_alert"] is False
    )


def test_a_new_episode_inside_the_cooldown_is_logged_but_not_announced():
    r1 = process(payload(p_last=0.6), {}, [], now=NOW)
    t1 = NOW + timedelta(minutes=60)
    r2 = process(payload(p_last=0.1, now=t1), r1.state, r1.history, now=t1)
    t2 = NOW + timedelta(minutes=90)
    r3 = process(
        payload(p_last=0.6, now=t2), r2.state, r2.history, now=t2
    )  # 90 min after the first announcement
    started = [e for e in r3.events if e["type"] == "started"][0]
    assert started["notified"] is False and "180" in started["reason"] and r3.notify == []
    t3 = NOW + timedelta(hours=5)
    r4 = process(
        payload(p_last=0.1, now=t3 - timedelta(minutes=30)),
        r3.state,
        r3.history,
        now=t3 - timedelta(minutes=30),
    )
    r5 = process(
        payload(p_last=0.6, now=t3), r4.state, r4.history, now=t3
    )  # beyond the cooldown: announced again
    assert len(r5.notify) == 1 and r5.notify[0]["notified"]


def test_a_data_gap_does_not_end_or_restart_an_episode():
    r1 = process(payload(p_last=0.6), {}, [], now=NOW)
    gap = NOW + timedelta(hours=8)
    r2 = process(payload(p_last=0.6, now=gap, age_min=500), r1.state, r1.history, now=gap)
    assert r2.events == [] and r2.state["in_alert"] is True  # unchanged, not "ended"
    back = NOW + timedelta(hours=9)
    r3 = process(payload(p_last=0.6, now=back), r2.state, r2.history, now=back)
    assert r3.events == [] and r3.notify == []  # data returned still high: no second announcement


# ---------------------------------------------------------------- observed flares are separate events


def test_an_observed_flare_is_announced_once_and_never_as_a_forecast():
    f = flare(NOW - timedelta(minutes=40))
    r1 = process(payload(flares=[f]), {}, [], now=NOW)
    obs = [e for e in r1.events if e["kind"] == "observed_flare"]
    assert (
        len(obs) == 1
        and obs[0]["class"] == "M2.1"
        and obs[0]["notified"]
        and obs[0]["forecast"] == "WARNED"
    )
    assert not [e for e in r1.events if e["kind"] == "predicted_risk"]
    r2 = process(payload(flares=[f]), r1.state, r1.history, now=NOW + timedelta(minutes=30))
    assert r2.events == [] and r2.notify == []  # the same flare is not announced again


def test_old_and_weak_flares_are_recorded_or_ignored_not_announced():
    old = flare(NOW - timedelta(hours=10), "M1.0")
    weak = flare(NOW - timedelta(minutes=30), "C5.0")
    r = process(payload(flares=[old, weak]), {}, [], now=NOW)
    assert (
        [e["class"] for e in r.events] == ["M1.0"]
        and r.notify == []
        and "recorded only" in r.events[0]["reason"]
    )
    r2 = process(payload(flares=[weak]), {}, [], AlertConfig(observed_min_class="C"), now=NOW)
    assert len(r2.notify) == 1


def test_history_is_capped():
    hist = [{"id": f"x{i}", "kind": "observed_flare"} for i in range(HISTORY_LIMIT + 50)]
    r = process(payload(p_last=0.6), {}, hist, now=NOW)
    assert len(r.history) == HISTORY_LIMIT and r.history[-1]["type"] == "started"


# ---------------------------------------------------------------- episodes with outcomes


def test_episodes_merge_flicker_and_judge_the_outcome():
    t0 = int(NOW.timestamp()) - 20 * 3600
    t = [t0 + 1800 * i for i in range(30)]
    p = [0.1] * 30
    for i in (2, 3, 5):  # one episode: the gap between 3 and 5 is only 30 min
        p[i] = 0.5
    p[4] = 0.2
    for i in (12, 13):
        p[i] = 0.6
    p[29] = 0.7  # 5.5 h ago: its 6 h window is still open
    flares = [
        flare(datetime.fromtimestamp(t[4] + 3600, tz=timezone.utc))
    ]  # within the first episode's window
    ep = episodes(t, p, 0.3, 360, flares, now_unix=NOW.timestamp())
    assert [e["outcome"] for e in ep] == ["hit", "false alarm", "pending"]
    assert ep[0]["peak_p"] == 0.5 and ep[0]["hours"] == 1.5 and ep[0]["flare"] == "M2.1"
    assert episodes(t, [0.0] * 30, 0.3) == []


# ---------------------------------------------------------------- messages


def test_a_predicted_risk_message_says_it_is_a_forecast_and_not_an_event():
    ev = process(payload(p_last=0.62), {}, [], now=NOW).notify[0]
    pl = payload(p_last=0.62)
    pl["explain"] = {
        "groups": [
            {"label": "Flares in the last 3 to 6 hours", "effect": 0.9, "detail": "2 M-class"},
            {"label": "Trend", "effect": -0.2},
        ]
    }
    subject, body = build_message(ev, pl, "https://flarenet.example")
    assert "risk" in subject.lower() and "observed" not in subject.lower()
    assert RISK_NOTE in body and "calibrated probability" in body and "62%" in body
    assert (
        "Flares in the last 3 to 6 hours (2 M-class)" in body and "Trend" not in body
    )  # only what raised the risk
    assert "https://flarenet.example" in body
    uncal = process(payload(p_last=0.62, calibrated=False), {}, [], now=NOW).notify[0]
    assert "not a calibrated probability" in build_message(uncal)[1]


def test_an_observed_message_is_not_an_official_record_and_reports_the_forecast_honestly():
    for fc, text in (
        ("WARNED", "lead time 95"),
        ("MISSED", "did not warn"),
        ("NO COVERAGE", "no forecast covering"),
    ):
        ev = process(
            payload(flares=[flare(NOW - timedelta(minutes=20), forecast=fc)]), {}, [], now=NOW
        ).notify[0]
        subject, body = build_message(ev)
        assert (
            "observed" in subject
            and OBSERVED_NOTE in body
            and text in body
            and "predicted" not in body.lower()
        )


# ---------------------------------------------------------------- email


def test_email_is_off_without_settings_and_parsed_from_the_environment():
    assert (
        smtp_settings({}) is None and smtp_settings({"SMTP_HOST": "smtp.x"}) is None
    )  # no recipient
    s = smtp_settings(
        {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USER": "me@x.org",
            "SMTP_PASSWORD": "pw",
            "ALERT_EMAIL_TO": "a@x.org; b@y.org",
        }
    )
    assert (
        s["port"] == 587
        and s["security"] == "starttls"
        and s["to"] == ["a@x.org", "b@y.org"]
        and s["sender"] == "me@x.org"
    )
    assert (
        smtp_settings({"SMTP_HOST": "h", "SMTP_PORT": "465", "ALERT_EMAIL_TO": "a@x.org"})[
            "security"
        ]
        == "ssl"
    )


class _FakeSMTP:
    sent: list = []
    log: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.log.append("starttls")

    def login(self, u, p):
        self.log.append(("login", u))

    def send_message(self, msg):
        self.sent.append(msg)


def test_send_email_uses_starttls_login_and_one_message_for_all_recipients():
    _FakeSMTP.sent, _FakeSMTP.log = [], []
    s = smtp_settings(
        {
            "SMTP_HOST": "h",
            "SMTP_USER": "u",
            "SMTP_PASSWORD": "p",
            "ALERT_EMAIL_TO": "a@x.org,b@x.org",
        }
    )
    send_email("subject", "body", s, smtp_factory=_FakeSMTP)
    assert _FakeSMTP.log == ["starttls", ("login", "u")]
    (m,) = _FakeSMTP.sent
    assert m["Subject"] == "subject" and m["To"] == "a@x.org, b@x.org" and "body" in m.get_content()


# ---------------------------------------------------------------- the script


def test_script_writes_history_and_does_not_repeat_on_the_next_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    now = datetime.now(timezone.utc)
    fc = tmp_path / "forecast.json"
    fc.write_text(json.dumps(payload(p_last=0.7, now=now)))
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    assert process_alerts.main(["--forecast", str(fc), "--out-dir", str(out1)]) == 0
    notify = json.loads((out1 / "notify.json").read_text())
    assert len(notify) == 1 and "risk" in notify[0]["subject"].lower()
    hist = json.loads((out1 / "alerts.json").read_text())
    assert hist["events"][0]["kind"] == "predicted_risk" and hist["updated"].endswith("Z")
    process_alerts.main(
        [
            "--forecast",
            str(fc),
            "--state",
            str(out1 / "alert_state.json"),
            "--history",
            str(out1 / "alerts.json"),
            "--out-dir",
            str(out2),
        ]
    )
    assert json.loads((out2 / "notify.json").read_text()) == []  # same episode: nothing new
    assert len(json.loads((out2 / "alerts.json").read_text())["events"]) == 1
    assert (
        process_alerts.main(["--send-only", str(out1 / "notify.json")]) == 0
    )  # no SMTP settings: nothing sent, no failure
    assert "email not configured" in capsys.readouterr().out


def test_script_survives_a_missing_or_broken_forecast(tmp_path, capsys):
    assert (
        process_alerts.main(
            ["--forecast", str(tmp_path / "none.json"), "--out-dir", str(tmp_path / "o")]
        )
        == 0
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert process_alerts.main(["--forecast", str(bad), "--out-dir", str(tmp_path / "o")]) == 0

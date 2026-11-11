import csv
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import main, settings  # noqa: E402

KEY = "k" * 20


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in (
        "FORECAST_URL",
        "FORECAST_FILE",
        "ALERTS_URL",
        "ALERTS_FILE",
        "FORWARD_LOG_URL",
        "FORWARD_LOG_FILE",
        "ADMIN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("FLARENET_BACKGROUND", "0")
    monkeypatch.setenv(
        "ALERTS_FILE", str(tmp_path / "no-alerts.json")
    )  # not the repo's (absent) file; keeps tests independent
    monkeypatch.setenv("FORWARD_LOG_FILE", str(tmp_path / "no-log.csv"))
    main._cache.update(body=None, fetched=0.0, origin=None)
    for c in main._aux.values():
        c.update(doc=None, fetched=0.0, origin="none", digest=None)
    main._store_state.update(path=None, store=None, error=None)
    yield
    if main._store_state["store"] is not None:
        main._store_state["store"].close()


EVENTS = [
    {
        "id": "risk-2026-11-01T10:00:00Z",
        "kind": "predicted_risk",
        "type": "started",
        "time": "2026-11-01T10:00:00Z",
        "p": 0.61,
        "threshold": 0.3,
        "level": "balanced",
        "notified": True,
    },
    {
        "id": "obs-2026-11-01T12:10:00Z",
        "kind": "observed_flare",
        "type": "observed",
        "time": "2026-11-01T12:10:00Z",
        "class": "M2.1",
        "forecast": "WARNED",
        "lead_min": 95,
        "notified": True,
    },
    {
        "id": "risk-end-2026-11-01T14:00:00Z",
        "kind": "predicted_risk",
        "type": "ended",
        "time": "2026-11-01T14:00:00Z",
        "p": 0.1,
        "notified": False,
    },
    {
        "id": "risk-2026-11-05T03:00:00Z",
        "kind": "predicted_risk",
        "type": "started",
        "time": "2026-11-05T03:00:00Z",
        "p": 0.4,
        "threshold": 0.3,
        "level": "balanced",
        "notified": False,
    },
]


def write_alerts(tmp_path, monkeypatch, events=EVENTS):
    p = tmp_path / "alerts.json"
    p.write_text(json.dumps({"updated": "2026-11-05T03:05:00Z", "events": events}))
    monkeypatch.setenv("ALERTS_FILE", str(p))


def test_serves_bundled_demo():
    c = TestClient(main.app)
    r = c.get("/api/forecast")
    assert r.status_code == 200 and r.headers["x-forecast-origin"] == "file"
    d = r.json()
    assert d["demo"] is True and len(d["series"]["t"]) > 100
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["demo"] is True and h["database"] == "ok"


def test_missing_file_is_a_json_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FORECAST_FILE", str(tmp_path / "none.json"))
    r = TestClient(main.app).get("/api/forecast")
    assert (
        r.status_code == 503
        and r.json()["error"]["code"] == "unavailable"
        and r.json()["error"]["request_id"]
    )


@pytest.mark.parametrize("content", ['{"hello": 1}', "{not json", "[1, 2]"])
def test_malformed_payload_is_a_json_502_not_a_crash(monkeypatch, tmp_path, content):
    p = tmp_path / "bad.json"
    p.write_text(content)
    monkeypatch.setenv("FORECAST_FILE", str(p))
    r = TestClient(main.app).get("/api/forecast")
    assert r.status_code == 502 and r.json()["error"]["code"] == "bad_payload"


def test_url_then_stale_when_it_goes_down(monkeypatch):
    good = (Path(main.ROOT) / "data" / "forecast.json").read_bytes()
    state = {"up": True}

    def fake(url):
        if not state["up"]:
            raise OSError("down")
        return good

    monkeypatch.setattr(main, "_from_url", fake)
    monkeypatch.setenv("FORECAST_URL", "https://example.test/f.json")
    monkeypatch.setenv("FORECAST_CACHE_S", "0")
    c = TestClient(main.app)
    assert c.get("/api/forecast").headers["x-forecast-origin"] == "url"
    state["up"] = False
    r = c.get("/api/forecast")
    assert r.status_code == 200 and r.headers["x-forecast-origin"] == "url-stale"


def test_cors_allows_configured_origin():
    c = TestClient(main.app)
    r = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------------------------------------------------------------- logging, request ids, unexpected errors


def test_every_response_carries_a_request_id_and_a_hostile_one_is_replaced():
    c = TestClient(main.app)
    assert c.get("/", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    evil = c.get("/", headers={"X-Request-ID": "x\nFAKE LOG LINE"}).headers["x-request-id"]
    assert "\n" not in evil and evil != "x\nFAKE LOG LINE" and len(evil) == 12


def test_an_unexpected_error_is_a_generic_json_500_without_internals(monkeypatch, caplog):
    def boom():
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main, "load_payload", boom)
    c = TestClient(main.app, raise_server_exceptions=False)
    with caplog.at_level("ERROR", logger="flarenet.api"):
        r = c.get("/api/health")
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal"
    assert "secret internal detail" not in r.text and r.json()["error"]["request_id"]
    assert (
        "secret internal detail" in caplog.text
    )  # the detail is in the server log, with the traceback


def test_requests_are_logged(caplog):
    with caplog.at_level("INFO", logger="flarenet.api"):
        TestClient(main.app).get("/api/forecast")
    assert any("/api/forecast -> 200" in m for m in caplog.messages)


# ---------------------------------------------------------------- alerts


def test_alerts_come_newest_first_and_are_filtered_and_paged(monkeypatch, tmp_path):
    write_alerts(tmp_path, monkeypatch)
    c = TestClient(main.app)
    d = c.get("/api/alerts").json()
    assert d["total"] == 4 and d["origin"] == "file" and d["updated"] == "2026-11-05T03:05:00Z"
    assert [e["time"] for e in d["events"]] == sorted((e["time"] for e in EVENTS), reverse=True)
    obs = c.get("/api/alerts", params={"kind": "observed_flare"}).json()
    assert (
        obs["total"] == 1
        and obs["events"][0]["class"] == "M2.1"
        and obs["events"][0]["kind"] == "observed_flare"
    )
    risk = c.get("/api/alerts", params={"kind": "predicted_risk", "since": "2026-11-02"}).json()
    assert risk["total"] == 1 and risk["events"][0]["time"].startswith("2026-11-05")
    page = c.get("/api/alerts", params={"limit": 2, "offset": 2}).json()
    assert (
        page["total"] == 4
        and len(page["events"]) == 2
        and page["events"][0]["id"] == EVENTS[1]["id"]
    )


def test_alerts_are_empty_not_an_error_when_nothing_has_been_published():
    d = TestClient(main.app).get("/api/alerts").json()
    assert d["origin"] == "none" and d["total"] == 0 and d["events"] == []


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 501}, {"offset": -1}, {"kind": "rumour"}, {"since": "yesterday"}, {"since": "2026-13-45; DROP TABLE alert_events"},
])  # fmt: skip
def test_invalid_alert_parameters_are_a_422_with_details(params):
    r = TestClient(main.app).get("/api/alerts", params=params)
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "validation_error" and e["details"] and "field" in e["details"][0]


def test_alert_events_are_stored_once_and_kept_when_the_file_shrinks(monkeypatch, tmp_path):
    write_alerts(tmp_path, monkeypatch)
    c = TestClient(main.app)
    for _ in range(3):  # repeated syncs never duplicate
        assert c.get("/api/alerts").json()["total"] == 4
    assert main.get_store().counts() == {"predicted_risk": 3, "observed_flare": 1}
    write_alerts(tmp_path, monkeypatch, EVENTS[-1:])  # the file's window moved on
    main.clear_caches()
    assert c.get("/api/alerts").json()["total"] == 4  # the database still has the older events


def test_malformed_events_do_not_break_the_store(monkeypatch, tmp_path):
    write_alerts(
        tmp_path,
        monkeypatch,
        [EVENTS[0], {"id": "x", "kind": "weird", "time": "2026-11-01T00:00:00Z"}],
    )
    d = TestClient(main.app).get("/api/alerts").json()
    assert [e["id"] for e in d["events"]] == [EVENTS[0]["id"]]


def test_alerts_use_memory_when_the_database_is_off(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "off")
    write_alerts(tmp_path, monkeypatch)
    c = TestClient(main.app)
    assert c.get("/api/health").json()["database"] == "off"
    d = c.get("/api/alerts", params={"kind": "observed_flare"}).json()
    assert d["total"] == 1


def test_an_unsupported_database_url_degrades_instead_of_failing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host/db")
    write_alerts(tmp_path, monkeypatch)
    c = TestClient(main.app)
    assert c.get("/api/health").json()["database"] == "unavailable"
    assert c.get("/api/alerts").json()["total"] == 4  # still served, from memory


def test_alerts_url_then_stale_when_it_goes_down(monkeypatch):
    body = json.dumps({"updated": "t", "events": EVENTS}).encode()
    state = {"up": True}

    def fake(url):
        if not state["up"]:
            raise OSError("down")
        return body

    monkeypatch.setattr(main, "_from_url", fake)
    monkeypatch.setenv("ALERTS_URL", "https://example.test/alerts.json")
    monkeypatch.setenv("AUX_CACHE_S", "0")
    c = TestClient(main.app)
    assert c.get("/api/alerts").json()["origin"] == "url"
    state["up"] = False
    d = c.get("/api/alerts").json()
    assert d["origin"] == "url-stale" and d["total"] == 4


# ---------------------------------------------------------------- forward log


def write_log(tmp_path, monkeypatch):
    rows = [
        {
            "for_day": "2026-10-25",
            "issued_utc": "2026-10-24T22:00:00Z",
            "target": "M",
            "p_flarenet": "0.043",
            "threshold": "0.204",
            "alert": "0",
            "p_noaa": "0.1",
            "noaa_source": "https://x",
            "y": "0.0",
            "y_cov": "1.0",
            "resolved_utc": "2026-10-26T00:25:27Z",
        },
        {
            "for_day": "2026-10-26",
            "issued_utc": "2026-10-25T22:00:00Z",
            "target": "M",
            "p_flarenet": "0.31",
            "threshold": "0.204",
            "alert": "1",
            "p_noaa": "0.2",
            "noaa_source": "https://x",
            "y": "1.0",
            "y_cov": "1.0",
            "resolved_utc": "2026-10-27T01:30:40Z",
        },
        {
            "for_day": "2026-10-27",
            "issued_utc": "2026-10-26T22:00:00Z",
            "target": "M",
            "p_flarenet": "0.07",
            "threshold": "0.204",
            "alert": "0",
            "p_noaa": "0.1",
            "noaa_source": "https://x",
            "y": "",
            "y_cov": "",
            "resolved_utc": "",
        },
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    p = tmp_path / "forward_log.csv"
    p.write_text(buf.getvalue())
    monkeypatch.setenv("FORWARD_LOG_FILE", str(p))


def test_forward_log_publishes_our_forecast_and_the_outcome_only(monkeypatch, tmp_path):
    write_log(tmp_path, monkeypatch)
    c = TestClient(main.app)
    d = c.get("/api/forward-log").json()
    assert d["total"] == 3 and [r["day"] for r in d["rows"]] == [
        "2026-10-27",
        "2026-10-26",
        "2026-10-25",
    ]
    hit = next(r for r in d["rows"] if r["day"] == "2026-10-26")
    assert (
        hit["p"] == 0.31
        and hit["alert"] is True
        and hit["resolved"] is True
        and hit["flare"] is True
    )
    pending = next(r for r in d["rows"] if r["day"] == "2026-10-27")
    assert pending["resolved"] is False and pending["flare"] is None
    assert not any(
        k.startswith("p_noaa") or "noaa" in k for r in d["rows"] for k in r
    )  # other columns are not republished
    only = c.get("/api/forward-log", params={"resolved": "true", "limit": 1}).json()
    assert only["total"] == 2 and len(only["rows"]) == 1


def test_forward_log_missing_is_empty_and_bad_limits_are_rejected():
    c = TestClient(main.app)
    assert c.get("/api/forward-log").json() == {"origin": "none", "total": 0, "rows": []}
    assert c.get("/api/forward-log", params={"limit": 0}).status_code == 422


# ---------------------------------------------------------------- admin


def test_admin_routes_do_not_exist_without_a_key():
    c = TestClient(main.app)
    assert (
        c.get("/api/admin/status", headers={"X-Admin-Key": KEY}).status_code == 404
    )  # no ADMIN_API_KEY configured
    assert c.post("/api/admin/refresh").status_code == 404


def test_a_short_admin_key_is_not_accepted_as_configuration(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "short")
    assert (
        TestClient(main.app).get("/api/admin/status", headers={"X-Admin-Key": "short"}).status_code
        == 404
    )


def test_admin_requires_the_right_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    c = TestClient(main.app)
    for headers in ({}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": KEY + "x"}, {"X-Admin-Key": ""}):
        r = c.get("/api/admin/status", headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert c.post("/api/admin/refresh").status_code == 401
    ok = c.get("/api/admin/status", headers={"X-Admin-Key": KEY})
    assert ok.status_code == 200 and ok.json()["database"]["enabled"] is True


def test_admin_refresh_refetches_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    write_alerts(tmp_path, monkeypatch)
    r = TestClient(main.app).post("/api/admin/refresh", headers={"X-Admin-Key": KEY})
    done = r.json()["refreshed"]
    assert (
        r.status_code == 200
        and done["forecast"] == "file"
        and done["alerts"] == "file"
        and done["new_alert_events"] == 4
    )
    assert done["forward_log"] == "none"
    st = TestClient(main.app).get("/api/admin/status", headers={"X-Admin-Key": KEY}).json()
    assert st["database"]["events"]["predicted_risk"] == 3 and st["alerts"]["origin"] == "file"


def test_the_admin_status_never_shows_secrets_or_urls(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    monkeypatch.setenv("FORECAST_URL", "https://user:secret@example.test/f.json")
    text = TestClient(main.app).get("/api/admin/status", headers={"X-Admin-Key": KEY}).text
    assert "secret" not in text and KEY not in text and "example.test" not in text


# ---------------------------------------------------------------- settings


def test_background_ingestion_runs_only_when_asked_or_when_remote_sources_are_set(monkeypatch):
    monkeypatch.delenv("FLARENET_BACKGROUND", raising=False)
    assert settings.background_enabled() is False
    monkeypatch.setenv("FORECAST_URL", "https://example.test/f.json")
    assert settings.background_enabled() is True
    monkeypatch.setenv("FLARENET_BACKGROUND", "0")
    assert settings.background_enabled() is False


def test_database_url_parsing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    assert settings.database_path() == ":memory:"
    monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/x.db")
    assert settings.database_path() == "/tmp/x.db"
    monkeypatch.setenv("DATABASE_URL", "OFF")
    assert settings.database_path() is None
    monkeypatch.setenv("DATABASE_URL", "mysql://h/db")
    with pytest.raises(ValueError):
        settings.database_path()


def test_the_lifespan_starts_and_stops_cleanly(monkeypatch):
    monkeypatch.setenv("FLARENET_BACKGROUND", "1")
    monkeypatch.setenv("INGEST_INTERVAL_S", "30")
    with TestClient(main.app) as c:  # runs startup and shutdown
        assert c.get("/").json()["service"] == "flarenet-api"

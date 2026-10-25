import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import main  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("FORECAST_URL", raising=False)
    monkeypatch.delenv("FORECAST_FILE", raising=False)
    main._cache.update(body=None, fetched=0.0, origin=None)


def test_serves_bundled_demo():
    c = TestClient(main.app)
    r = c.get("/api/forecast")
    assert r.status_code == 200 and r.headers["x-forecast-origin"] == "file"
    d = r.json()
    assert d["demo"] is True and len(d["series"]["t"]) > 100
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["demo"] is True


def test_missing_file_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FORECAST_FILE", str(tmp_path / "none.json"))
    assert TestClient(main.app).get("/api/forecast").status_code == 503


def test_rejects_malformed_payload(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"hello": 1}))
    monkeypatch.setenv("FORECAST_FILE", str(p))
    with pytest.raises(ValueError):
        TestClient(main.app).get("/api/forecast")


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

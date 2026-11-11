"""Turn the newest forecast.json into alert events, history and (optionally) emails.

    python scripts/process_alerts.py --forecast out/forecast.json --state prev/alert_state.json \
        --history prev/alerts.json --out-dir out

Writes ``alert_state.json`` (what the next run needs), ``alerts.json`` (the event history the website shows) and ``notify.json``
(what to announce) to --out-dir. ``--send-only out/notify.json`` then sends those as email, using SMTP_HOST, SMTP_PORT,
SMTP_USER, SMTP_PASSWORD, SMTP_SECURITY (starttls | ssl), ALERT_EMAIL_FROM and ALERT_EMAIL_TO (comma-separated). With no
SMTP settings nothing is sent and nothing fails. Rules (no alert from stale or demo data, one alert per episode, a cooldown,
predicted risk kept apart from observed flares): flarenet/alerts/engine.py.

Settings: --level sensitive|balanced|strict (env ALERT_LEVEL), --threshold (env ALERT_THRESHOLD, overrides the level),
--cooldown-min (env ALERT_COOLDOWN_MIN). The script never fails the calling workflow over an email problem.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from flarenet.alerts.engine import AlertConfig, process
from flarenet.alerts.notify import build_message, send_email, smtp_settings


def _read(path: str | None, default):
    if not path or not Path(path).exists():
        return default
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        print(f"could not read {path}; starting from empty", file=sys.stderr)
        return default


def _send(items: list[dict], site_url: str) -> int:
    settings = smtp_settings()
    if settings is None:
        print("email not configured (SMTP_HOST / ALERT_EMAIL_TO); nothing sent")
        return 0
    sent = 0
    for it in items:
        try:
            send_email(it["subject"], it["body"], settings)
            sent += 1
            print(f"sent: {it['subject']}")
        except Exception as e:  # noqa: BLE001 - an email problem must not fail the forecast job; never print the password
            print(f"email failed ({type(e).__name__}): {it['subject']}", file=sys.stderr)
    return sent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--forecast", help="forecast.json from scripts/export_web.py")
    ap.add_argument("--state", help="alert_state.json from the previous run (optional)")
    ap.add_argument("--history", help="alerts.json from the previous run (optional)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--level", default=os.environ.get("ALERT_LEVEL") or "balanced")
    ap.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ["ALERT_THRESHOLD"]) if os.environ.get("ALERT_THRESHOLD") else None,
    )
    ap.add_argument(
        "--cooldown-min", type=float, default=float(os.environ.get("ALERT_COOLDOWN_MIN") or "180")
    )
    ap.add_argument("--site-url", default=os.environ.get("SITE_URL", ""))
    ap.add_argument(
        "--send-only",
        metavar="NOTIFY_JSON",
        help="send the messages listed in this notify.json and do nothing else",
    )
    a = ap.parse_args(argv)

    if a.send_only:
        _send(_read(a.send_only, []), a.site_url)
        return 0
    if not a.forecast:
        ap.error("--forecast is required")
    payload = _read(a.forecast, None)
    if payload is None:
        print("no readable forecast; no alerts processed", file=sys.stderr)
        return 0
    cfg = AlertConfig(level=a.level, threshold=a.threshold, cooldown_min=a.cooldown_min)
    res = process(
        payload,
        _read(a.state, {}),
        _read(a.history, {}).get("events", [])
        if isinstance(_read(a.history, {}), dict)
        else _read(a.history, []),
        cfg,
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alert_state.json").write_text(json.dumps(res.state, indent=1))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "alerts.json").write_text(
        json.dumps({"updated": stamp, "events": res.history}, separators=(",", ":"))
    )
    msgs = []
    for ev in res.notify:
        subject, body = build_message(ev, payload, a.site_url)
        msgs.append({"event": ev["id"], "subject": subject, "body": body})
    (out / "notify.json").write_text(json.dumps(msgs, indent=1))
    if res.skipped:
        print(f"no alert decision made: {res.skipped}")
    print(f"{len(res.events)} new event(s), {len(msgs)} to announce, {len(res.history)} in history")
    return 0


if __name__ == "__main__":
    sys.exit(main())

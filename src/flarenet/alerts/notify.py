"""Words and delivery of an alert: plain-text messages and SMTP email (standard library only)."""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

NOAA = "https://www.swpc.noaa.gov/"

RISK_NOTE = (
    "This is a PREDICTED RISK from the FlareNet model. No flare has been observed because of this alert. "
    "It is a probability, not a guarantee that a flare will happen."
)
OBSERVED_NOTE = (
    "This is an OBSERVED event from the live GOES X-ray reading, not an official NOAA event record. "
    f"For official space weather information see {NOAA}"
)


def build_message(event: dict, payload: dict | None = None, site_url: str = "") -> tuple[str, str]:
    """``(subject, body)`` for an event. A predicted risk never says a flare happened; an observed flare never says it was forecast
    unless the forecast before it really was."""
    payload = payload or {}
    link = f"\nDashboard: {site_url}\n" if site_url else "\n"
    if event["kind"] == "predicted_risk":
        p = event["p"]
        meaning = (
            f"The model's calibrated probability of an M or X flare starting within the next 6 hours is {100 * p:.0f}%."
            if event.get("calibrated")
            else f"The model's score for an M or X flare within the next 6 hours is {p:.2f} (not a calibrated probability)."
        )
        lines = [
            f"FlareNet: elevated M/X flare risk ({100 * p:.0f}%)",
        ]
        body = [
            meaning,
            f"Alert level: {event['level']} (threshold {event['threshold']:.2f}). Forecast time: {event['time']}.",
        ]
        why = [
            g
            for g in (payload.get("explain") or {}).get("groups", [])[:3]
            if g.get("effect", 0) > 0
        ]
        if why:
            body.append(
                "Main reasons: "
                + "; ".join(
                    f"{g['label']}" + (f" ({g['detail']})" if g.get("detail") else "") for g in why
                )
                + "."
            )
        body += ["", RISK_NOTE]
        return lines[0], "\n".join(body) + link
    cls = event["class"]
    fc = event.get("forecast")
    said = {
        "WARNED": f"The model had an alert before it (lead time {event['lead_min']} min)."
        if event.get("lead_min") is not None
        else "The model had an alert before it.",
        "MISSED": "The model did not warn of this flare in time.",
        "NO COVERAGE": "The model had no forecast covering the time before this flare.",
    }.get(fc or "", "")
    body = [f"Peak: {event['time']}.", said, "", OBSERVED_NOTE]
    return f"FlareNet: {cls}-class flare observed", "\n".join(
        b for b in body if b is not None
    ) + link


def recipients(env=None) -> list[str]:
    env = os.environ if env is None else env
    return [
        a.strip() for a in env.get("ALERT_EMAIL_TO", "").replace(";", ",").split(",") if a.strip()
    ]


def smtp_settings(env=None) -> dict | None:
    """SMTP settings from the environment, or None when email is not configured (then nothing is sent, nothing fails)."""
    env = os.environ if env is None else env
    host, to = env.get("SMTP_HOST", "").strip(), recipients(env)
    if not host or not to:
        return None
    port = int(env.get("SMTP_PORT", "587") or 587)
    user = env.get("SMTP_USER", "")
    return {"host": host, "port": port, "user": user, "password": env.get("SMTP_PASSWORD", ""),
            "sender": env.get("ALERT_EMAIL_FROM") or user or "flarenet@localhost", "to": to,
            "security": (env.get("SMTP_SECURITY") or ("ssl" if port == 465 else "starttls")).lower()}  # fmt: skip


def send_email(
    subject: str, body: str, settings: dict, timeout: float = 20.0, smtp_factory=None
) -> None:
    """Send one plain-text message to every recipient. Raises on failure (the caller records it)."""
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, settings["sender"], ", ".join(settings["to"])
    msg.set_content(body)
    ctx = ssl.create_default_context()
    if settings["security"] == "ssl":
        make = smtp_factory or (
            lambda: smtplib.SMTP_SSL(
                settings["host"], settings["port"], timeout=timeout, context=ctx
            )
        )
    else:
        make = smtp_factory or (
            lambda: smtplib.SMTP(settings["host"], settings["port"], timeout=timeout)
        )
    with make() as s:
        if settings["security"] == "starttls":
            s.starttls(context=ctx)
        if settings["user"]:
            s.login(settings["user"], settings["password"])
        s.send_message(msg)

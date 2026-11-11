"""Authentication of administrative endpoints: a shared key in the ``X-Admin-Key`` header, compared in constant time."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.settings import admin_key


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Dependency for admin routes. 404 (not 401) when no key is configured, so a deployment without one does not advertise
    admin routes; 401 for a missing or wrong key."""
    expected = admin_key()
    if expected is None:
        raise HTTPException(404, "Not found")
    if not x_admin_key or not hmac.compare_digest(x_admin_key.encode(), expected.encode()):
        raise HTTPException(
            401, "Invalid or missing admin key", headers={"WWW-Authenticate": "X-Admin-Key"}
        )

"""A small SQLite store for the alert events the API serves (standard library only).

The source of truth for events is ``alerts.json`` on the data branch, written by the refresh workflow. The API copies new events
into this store (``INSERT OR IGNORE`` by event id) so that it can filter and page them with SQL, and so that history older than
the file's 500-event window is kept for as long as the database file lives. On Render's free plan the disk is reset when the
service restarts; the store is then rebuilt from ``alerts.json``, so nothing is lost that the file still has.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_events (
    id        TEXT PRIMARY KEY,
    kind      TEXT NOT NULL CHECK (kind IN ('predicted_risk', 'observed_flare')),
    type      TEXT NOT NULL,
    time      TEXT NOT NULL,
    p         REAL,
    threshold REAL,
    level     TEXT,
    class     TEXT,
    forecast  TEXT,
    lead_min  REAL,
    notified  INTEGER NOT NULL DEFAULT 0,
    detail    TEXT NOT NULL,
    stored_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alert_events_time ON alert_events (time DESC);
CREATE INDEX IF NOT EXISTS ix_alert_events_kind ON alert_events (kind, time DESC);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add_events(self, events: list[dict]) -> int:
        """Store events not seen before; returns how many were new. Malformed events are skipped, never fatal."""
        new = 0
        with self._lock:
            for e in events:
                try:
                    cur = self._db.execute(
                        "INSERT OR IGNORE INTO alert_events "
                        "(id, kind, type, time, p, threshold, level, class, forecast, lead_min, notified, detail, stored_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            str(e["id"]), e["kind"], str(e.get("type", "")), str(e["time"]), e.get("p"), e.get("threshold"),
                            e.get("level"), e.get("class"), e.get("forecast"), e.get("lead_min"),
                            1 if e.get("notified") else 0, json.dumps(e, separators=(",", ":")), _now(),
                        ),
                    )  # fmt: skip
                    new += cur.rowcount
                except (KeyError, TypeError, ValueError, sqlite3.IntegrityError):
                    continue
            self._db.commit()
        return new

    def query(
        self, limit: int = 50, offset: int = 0, kind: str | None = None, since: str | None = None
    ) -> tuple[int, list[dict]]:
        """``(total matching, events newest first)``. All values are bound parameters; nothing is interpolated into SQL."""
        where, args = [], []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if since:
            where.append("time >= ?")
            args.append(since)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM alert_events{clause}", args).fetchone()[
                0
            ]  # noqa: S608 (fixed fragments)
            rows = self._db.execute(
                f"SELECT detail FROM alert_events{clause} ORDER BY time DESC, id DESC LIMIT ? OFFSET ?",  # noqa: S608
                [*args, limit, offset],
            ).fetchall()
        return int(total), [json.loads(r["detail"]) for r in rows]

    def counts(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, COUNT(*) AS n FROM alert_events GROUP BY kind"
            ).fetchall()
        return {r["kind"]: int(r["n"]) for r in rows}

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._db.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

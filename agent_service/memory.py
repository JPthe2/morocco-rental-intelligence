"""Per-session conversation memory: message history + extracted preferences,
stored in a local SQLite file. Sessions expire after config.SESSION_TTL_HOURS
of inactivity.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import config


def get_db(db_path=None) -> sqlite3.Connection:
    path = db_path or config.SESSIONS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "session_id TEXT PRIMARY KEY, created_at TEXT, last_active_at TEXT, preferences TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
        "content TEXT, created_at TEXT)"
    )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def touch_session(session_id: str, db_path=None) -> None:
    """Create the session if it doesn't exist yet, else bump last_active_at."""
    db = get_db(db_path)
    now = _now()
    db.execute(
        "INSERT INTO sessions (session_id, created_at, last_active_at, preferences) "
        "VALUES (?, ?, ?, '{}') "
        "ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at",
        (session_id, now, now),
    )
    db.commit()
    db.close()


def append_message(session_id: str, role: str, content: str, db_path=None) -> None:
    db = get_db(db_path)
    db.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now()),
    )
    db.commit()
    db.close()


def get_history(session_id: str, limit: int = 20, db_path=None) -> list[dict]:
    db = get_db(db_path)
    rows = db.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    db.close()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def get_preferences(session_id: str, db_path=None) -> dict:
    db = get_db(db_path)
    row = db.execute("SELECT preferences FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    db.close()
    return json.loads(row[0]) if row and row[0] else {}


def update_preferences(session_id: str, updates: dict, db_path=None) -> dict:
    """Merge `updates` into the session's stored preferences and return the merged dict."""
    current = get_preferences(session_id, db_path=db_path)
    current.update(updates)
    db = get_db(db_path)
    db.execute(
        "UPDATE sessions SET preferences = ? WHERE session_id = ?",
        (json.dumps(current), session_id),
    )
    db.commit()
    db.close()
    return current


def expire_stale_sessions(ttl_hours: Optional[float] = None, db_path=None) -> int:
    """Delete sessions (and their messages) inactive for longer than the TTL.
    Returns the number of sessions removed."""
    ttl_hours = ttl_hours if ttl_hours is not None else config.SESSION_TTL_HOURS
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
    db = get_db(db_path)
    stale_ids = [
        row[0] for row in
        db.execute("SELECT session_id FROM sessions WHERE last_active_at < ?", (cutoff,)).fetchall()
    ]
    for session_id in stale_ids:
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    db.commit()
    db.close()
    return len(stale_ids)

"""Turn-level logging: every agent turn (tool calls, arguments, latency, token
counts, success/failure) is recorded to a local SQLite DB. This is the
foundation for a future eval harness — for now it's just structured logging.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import config


def get_db(db_path=None) -> sqlite3.Connection:
    path = db_path or config.LOGS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turns ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_message TEXT, "
        "tool_calls TEXT, total_latency_ms REAL, prompt_tokens INTEGER, "
        "completion_tokens INTEGER, success INTEGER, error TEXT, created_at TEXT)"
    )
    conn.commit()
    return conn


class TurnLogger:
    """Accumulates tool-call records for a single agent turn, then writes one row."""

    def __init__(self, session_id: str, user_message: str, db_path=None):
        self.session_id = session_id
        self.user_message = user_message
        self.db_path = db_path
        self.tool_calls: list[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._start = time.perf_counter()

    @contextmanager
    def record_tool_call(self, name: str, arguments: dict):
        start = time.perf_counter()
        record = {"name": name, "arguments": arguments, "latency_ms": None, "error": None}
        try:
            yield record
        except Exception as exc:  # noqa: BLE001 - we want to log and re-raise
            record["error"] = str(exc)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.tool_calls.append(record)

    def add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0

    def finish(self, success: bool, error: str = ""):
        total_latency_ms = round((time.perf_counter() - self._start) * 1000, 1)
        db = get_db(self.db_path)
        db.execute(
            "INSERT INTO turns (session_id, user_message, tool_calls, total_latency_ms, "
            "prompt_tokens, completion_tokens, success, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.session_id,
                self.user_message,
                json.dumps(self.tool_calls),
                total_latency_ms,
                self.prompt_tokens,
                self.completion_tokens,
                int(success),
                error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
        db.close()

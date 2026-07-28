import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory


def test_touch_session_creates_with_empty_preferences(tmp_path):
    db_path = tmp_path / "sessions.db"
    memory.touch_session("s1", db_path=db_path)
    assert memory.get_preferences("s1", db_path=db_path) == {}


def test_append_and_get_history_preserves_order(tmp_path):
    db_path = tmp_path / "sessions.db"
    memory.touch_session("s1", db_path=db_path)
    memory.append_message("s1", "user", "hello", db_path=db_path)
    memory.append_message("s1", "assistant", "hi there", db_path=db_path)
    history = memory.get_history("s1", db_path=db_path)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "hello"


def test_update_preferences_merges(tmp_path):
    db_path = tmp_path / "sessions.db"
    memory.touch_session("s1", db_path=db_path)
    memory.update_preferences("s1", {"max_budget_mad": "6000"}, db_path=db_path)
    merged = memory.update_preferences("s1", {"min_bedrooms": "2"}, db_path=db_path)
    assert merged == {"max_budget_mad": "6000", "min_bedrooms": "2"}


def test_expire_stale_sessions_removes_only_old_ones(tmp_path):
    db_path = tmp_path / "sessions.db"
    memory.touch_session("old", db_path=db_path)
    memory.touch_session("fresh", db_path=db_path)

    old_time = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE sessions SET last_active_at = ? WHERE session_id = 'old'", (old_time,))
    conn.commit()
    conn.close()

    removed = memory.expire_stale_sessions(ttl_hours=6, db_path=db_path)
    assert removed == 1

    remaining = memory.get_db(db_path).execute("SELECT session_id FROM sessions").fetchall()
    assert ("fresh",) in remaining
    assert ("old",) not in remaining

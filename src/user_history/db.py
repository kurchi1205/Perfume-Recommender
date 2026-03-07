import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "user_history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id          TEXT PRIMARY KEY,
    accords          TEXT NOT NULL DEFAULT '[]',   -- JSON array of preferred accord strings
    preferred_gender TEXT,                          -- "For Men" | "For Women" | "Unisex" | NULL
    past_perfume_ids TEXT NOT NULL DEFAULT '[]',   -- JSON array of Milvus perfume ids shown before
    session_count    INTEGER NOT NULL DEFAULT 0,
    summary          TEXT,                          -- natural-language summary of user preferences
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(CREATE_TABLE_SQL)


def get_or_create_user(user_id: str | None = None) -> str:
    """Return an existing user_id or create a new row and return its id."""
    if user_id is None:
        user_id = str(uuid.uuid4())

    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)",
            (user_id,),
        )
    return user_id

if __name__ == "__main__":
    init_db()
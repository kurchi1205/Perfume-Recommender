import json
from typing import List

from db import get_conn, get_or_create_user
from models import UserPreferenceSignal


def load_preferences(user_id: str) -> UserPreferenceSignal:
    """Load the preference row for user_id. Creates the row if it doesn't exist."""
    get_or_create_user(user_id)

    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
        ).fetchone()

    return UserPreferenceSignal(
        user_id=row["user_id"],
        accords=json.loads(row["accords"]),
        preferred_gender=row["preferred_gender"],
        past_perfume_ids=json.loads(row["past_perfume_ids"]),
        session_count=row["session_count"],
        summary=row["summary"],
    )


def upsert_preferences(
    user_id: str,
    new_accords: List[str] | None = None,
    preferred_gender: str | None = None,
    new_perfume_ids: List[str] | None = None,
    summary: str | None = None,
) -> UserPreferenceSignal:
    """
    Merge new data into the user's preference row.

    - new_accords: accords from the current session (deduplicated, appended)
    - preferred_gender: overwrites if provided
    - new_perfume_ids: Milvus ids shown this session (deduplicated, appended)

    Returns the updated UserPreferenceSignal.
    """
    current = load_preferences(user_id)

    # Merge accords — keep unique, preserve order
    merged_accords = current.accords
    if new_accords:
        existing = set(merged_accords)
        merged_accords = merged_accords + [a for a in new_accords if a not in existing]

    # Merge past perfume ids — keep unique, preserve order
    merged_ids = current.past_perfume_ids
    if new_perfume_ids:
        existing_ids = set(merged_ids)
        merged_ids = merged_ids + [p for p in new_perfume_ids if p not in existing_ids]

    gender = preferred_gender if preferred_gender is not None else current.preferred_gender
    new_summary = summary if summary is not None else current.summary

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE user_preferences
            SET accords          = ?,
                preferred_gender = ?,
                past_perfume_ids = ?,
                summary          = ?,
                session_count    = session_count + 1,
                updated_at       = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (
                json.dumps(merged_accords),
                gender,
                json.dumps(merged_ids),
                new_summary,
                user_id,
            ),
        )

    return load_preferences(user_id)

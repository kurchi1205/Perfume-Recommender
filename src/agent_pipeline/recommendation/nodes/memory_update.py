"""
Memory Update — persists the completed session to user_history.db.
Runs after Explanation Generator so result_accords can be denormalized
at write time, making future profile reads cheap (no Milvus calls needed).
"""
import asyncio
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[5] / "user_history.db"


def _write(user_id: str, query_text: str, intent_summary: str,
           accords_queried: list, recommendations: list):
    result_ids     = [r["perfume_id"] for r in recommendations]
    result_accords = {r["perfume_id"]: r.get("main_accords", []) for r in recommendations}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO user_sessions
                    (user_id, query_text, intent_summary, accords_queried, result_ids, result_accords)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    query_text,
                    intent_summary,
                    json.dumps(accords_queried),
                    json.dumps(result_ids),
                    json.dumps(result_accords),
                ),
            )
            conn.commit()
        logger.info("[memory_update] session saved for user=%s results=%s", user_id, result_ids)
    except Exception as e:
        logger.warning("[memory_update] DB write failed: %s", e)


async def memory_update_node(state: dict) -> dict:
    user_id         = state.get("user_id", "default")
    query_text      = state.get("query_text", "")
    intent_summary  = state.get("intent_summary", "")
    accords_queried = state.get("extracted_accords", [])
    recommendations = state.get("recommendations", [])

    # Run the blocking SQLite write in a thread so the event loop isn't stalled
    await asyncio.to_thread(
        _write, user_id, query_text, intent_summary, accords_queried, recommendations
    )

    return {}

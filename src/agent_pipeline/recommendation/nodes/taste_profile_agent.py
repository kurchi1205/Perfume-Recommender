import json
import logging
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from schemas import ExtractedList

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[5] / "user_history.db"
MAX_RETRIES = 3

llm = ChatOllama(model="ministral-3:3b", temperature=0.3)

SYSTEM_PROMPT = """\
You are a perfume expert. Analyze the user's input and return a JSON object with:
- "moods": array of 3-7 mood/feeling words (e.g. "warm", "cozy", "mysterious")
- "accords": array of 3-7 scent accords (e.g. "woody", "oriental", "citrus")
- "intent_summary": one sentence describing what the user is looking for

{profile_section}

Rules:
- Use lowercase for all moods and accords
- Do not include explanations outside the JSON
- Return ONLY the JSON object
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT DEFAULT 'default',
            query_text      TEXT,
            intent_summary  TEXT,
            accords_queried TEXT,
            result_ids      TEXT,
            result_accords  TEXT,
            timestamp       TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id    TEXT DEFAULT 'default',
            perfume_id TEXT NOT NULL,
            reaction   TEXT NOT NULL,
            accords    TEXT,
            timestamp  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (user_id, perfume_id)
        );
    """)
    conn.commit()


def _build_profile_context(user_id: str) -> str:
    """Read user_history.db and build a profile string to inject into the LLM prompt."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _ensure_tables(conn)

            liked = conn.execute(
                "SELECT accords FROM user_preferences WHERE user_id=? AND reaction='liked' ORDER BY timestamp DESC LIMIT 10",
                (user_id,)
            ).fetchall()

            disliked = conn.execute(
                "SELECT accords FROM user_preferences WHERE user_id=? AND reaction='disliked' ORDER BY timestamp DESC LIMIT 5",
                (user_id,)
            ).fetchall()

            past_accords_rows = conn.execute(
                "SELECT accords_queried FROM user_sessions WHERE user_id=? ORDER BY timestamp DESC LIMIT 5",
                (user_id,)
            ).fetchall()

    except Exception as e:
        logger.warning("[taste_profile] DB read failed: %s", e)
        return ""

    liked_accords = []
    for (raw,) in liked:
        try:
            liked_accords.extend(json.loads(raw or "[]"))
        except Exception:
            pass

    disliked_accords = []
    for (raw,) in disliked:
        try:
            disliked_accords.extend(json.loads(raw or "[]"))
        except Exception:
            pass

    freq_accords = {}
    for (raw,) in past_accords_rows:
        try:
            for a in json.loads(raw or "[]"):
                freq_accords[a] = freq_accords.get(a, 0) + 1
        except Exception:
            pass
    repeat_accords = [a for a, count in freq_accords.items() if count >= 3]

    lines = []
    if liked_accords:
        lines.append(f"User has liked perfumes with these accords: {', '.join(set(liked_accords))}")
    if disliked_accords:
        lines.append(f"User dislikes: {', '.join(set(disliked_accords))}")
    if repeat_accords:
        lines.append(f"User often searches for: {', '.join(repeat_accords)}")

    if not lines:
        return ""

    return "User profile (use this to inform your extraction):\n" + "\n".join(lines)


# ── Node ──────────────────────────────────────────────────────────────────────

def taste_profile_node(state: dict) -> dict:
    mood_input = state.get("mood_input", "")
    input_type = state.get("input_type", "text")
    user_id    = state.get("user_id", "default")

    profile_context = _build_profile_context(user_id)
    system = SYSTEM_PROMPT.format(
        profile_section=profile_context if profile_context else ""
    )

    if input_type == "image":
        content = [
            {"type": "image_url", "image_url": {"url": mood_input}},
            {"type": "text", "text": "Extract moods, accords and intent from this image."},
        ]
    else:
        content = mood_input[:1000]

    extracted_moods   = []
    extracted_accords = []
    intent_summary    = ""

    for attempt in range(1, MAX_RETRIES + 1):
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=content),
        ])
        try:
            import re
            raw = response.content
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

            extracted_moods   = ExtractedList(items=parsed.get("moods", [])).items
            extracted_accords = ExtractedList(items=parsed.get("accords", [])).items
            intent_summary    = str(parsed.get("intent_summary", "")).strip()

            if not intent_summary:
                raise ValueError("intent_summary is empty")
            break
        except Exception as e:
            logger.warning("[taste_profile] attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logger.error("[taste_profile] all attempts failed — using empty extraction")

    logger.info("[taste_profile] moods=%s accords=%s", extracted_moods, extracted_accords)
    return {
        "extracted_moods":   extracted_moods,
        "extracted_accords": extracted_accords,
        "intent_summary":    intent_summary,
    }

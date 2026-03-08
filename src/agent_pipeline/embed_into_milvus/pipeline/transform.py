"""
Stage 2 — Transform
Clean fields and build the summary text that will be embedded.

Summary = "Moods: {moods}. Scent accords: {accords}. {description}"

All three are also stored as separate fields on the record.
"""
import uuid
from typing import Iterator


def _join(lst: list) -> str:
    return ", ".join(str(x).strip() for x in lst if x)


def _build_summary(moods_str: str, accords_str: str, description: str) -> str:
    parts = []
    if moods_str:
        parts.append(f"Moods: {moods_str}")
    if accords_str:
        parts.append(f"Scent accords: {accords_str}")
    if description:
        parts.append(description.strip())
    return ". ".join(parts)


def transform(records: Iterator[dict]) -> Iterator[dict]:
    """
    Yield Milvus-ready dicts (without embedding — added in Stage 3).
    Each dict has a `summary` field ready to be embedded.
    """
    for item in records:
        notes = item.get("notes", {})

        moods_str   = _join(item.get("moods", []))
        accords_str = _join(item.get("main_accords", []))
        description = item.get("description", "") or ""
        summary     = _build_summary(moods_str, accords_str, description)

        yield {
            "id":           str(uuid.uuid4()),
            "name":         item.get("name", "").strip(),
            "description":  description,
            "url":          item.get("url", "").strip(),
            "brand":        (item.get("brand") or "").strip(),
            "gender":       (item.get("gender") or "").strip(),
            "top_notes":    _join(notes.get("top", [])),
            "middle_notes": _join(notes.get("middle", [])),
            "base_notes":   _join(notes.get("base", [])),
            "main_accords": accords_str,
            "moods":        moods_str,
            "summary":      summary,
            # embedding added in Stage 3
        }

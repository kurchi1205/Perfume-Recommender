"""
Stage 1 — Extract
Read records from a JSONL file, validate required fields, skip bad rows.
"""
import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"name", "url"}


def extract(input_path: str) -> Iterator[dict]:
    """
    Yield valid perfume dicts from a JSONL file.
    Skips blank lines and records missing required fields.
    Logs a warning for every skipped record.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total = skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: JSON parse error — %s", lineno, e)
                skipped += 1
                continue

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                logger.warning(
                    "Line %d: missing fields %s — skipping '%s'",
                    lineno, missing, record.get("name", "<unknown>")
                )
                skipped += 1
                continue

            yield record

    logger.info("Extract done: %d total, %d skipped, %d valid", total, skipped, total - skipped)

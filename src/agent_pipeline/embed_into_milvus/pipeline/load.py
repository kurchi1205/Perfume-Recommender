"""
Stage 5 — Load
Batch-insert embedded records into Milvus.
Skips URLs already present in the collection (resumable).
Failed records are written to failed.jsonl for inspection.
"""
import json
import logging
from pathlib import Path
from typing import Iterator

from pymilvus import MilvusClient
from tqdm import tqdm

logger = logging.getLogger(__name__)

INSERT_BATCH_SIZE = 100
COLLECTION        = "perfume_collection"


def _fetch_existing_urls(client: MilvusClient) -> set:
    """Return the set of URLs already stored in the collection."""
    existing = set()
    offset = 0
    limit  = 1000
    while True:
        results = client.query(
            collection_name=COLLECTION,
            filter="url != ''",
            output_fields=["url"],
            limit=limit,
            offset=offset,
        )
        if not results:
            break
        for row in results:
            existing.add(row["url"])
        if len(results) < limit:
            break
        offset += limit
    logger.info("Found %d existing URLs in collection", len(existing))
    return existing


def load(
    records: Iterator[dict],
    client: MilvusClient,
    failed_path: str = "failed.jsonl",
    total: int = 0,
) -> None:
    existing_urls = _fetch_existing_urls(client)

    batch        = []
    inserted     = skipped = failed = 0
    failed_file  = open(failed_path, "w", encoding="utf-8")
    bar          = tqdm(total=total or None, desc="Loading", unit="perfume")

    def flush(b: list):
        nonlocal inserted, failed
        try:
            client.insert(collection_name=COLLECTION, data=b)
            inserted += len(b)
        except Exception as e:
            logger.error("Batch insert failed: %s — writing %d records to failed.jsonl", e, len(b))
            for r in b:
                r.pop("moods_embedding", None)  # don't serialise the big vector
                failed_file.write(json.dumps(r) + "\n")
            failed += len(b)

    for record in records:
        bar.update(1)
        if record.get("url") in existing_urls:
            skipped += 1
            continue

        batch.append(record)
        if len(batch) == INSERT_BATCH_SIZE:
            flush(batch)
            batch = []

    if batch:
        flush(batch)

    bar.close()
    failed_file.close()

    logger.info(
        "Load done — inserted: %d  skipped: %d  failed: %d",
        inserted, skipped, failed,
    )

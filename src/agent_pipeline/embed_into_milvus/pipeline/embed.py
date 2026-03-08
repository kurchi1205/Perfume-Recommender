"""
Stage 3 — Embed
Generate BGE-M3 embeddings for the summary field in batches.
Adds `moods_embedding` (1024-dim) to each record.
"""
import logging
from typing import Iterator

from tqdm import tqdm

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64  # texts sent to the model at once


def embed(records: Iterator[dict], embedder, total: int = 0) -> Iterator[dict]:
    """
    Yield records with `moods_embedding` populated.
    Processes summaries in batches for GPU/CPU efficiency.
    """
    batch_records = []
    bar = tqdm(total=total or None, desc="Embedding", unit="perfume")

    for record in records:
        batch_records.append(record)
        if len(batch_records) == EMBED_BATCH_SIZE:
            yield from _embed_batch(batch_records, embedder, bar)
            batch_records = []

    if batch_records:
        yield from _embed_batch(batch_records, embedder, bar)

    bar.close()
    logger.info("Embedding complete.")


def _embed_batch(batch: list, embedder, bar) -> list:
    summaries = [r["summary"] for r in batch]
    vectors = embedder.embed_documents(summaries)  # batched call
    for record, vector in zip(batch, vectors):
        record["moods_embedding"] = vector
    bar.update(len(batch))
    return batch

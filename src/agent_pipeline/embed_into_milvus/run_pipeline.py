"""
Data ingestion pipeline entry point.

Usage:
    python run_pipeline.py --input ../../datasets/perfumes_with_moods.jsonl
    python run_pipeline.py --input ../../datasets/perfumes_with_moods.jsonl --device cuda
"""
import argparse
import logging
import sys
from pathlib import Path

# Make utils importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import init_bge_embedder
from pipeline.extract  import extract
from pipeline.transform import transform
from pipeline.embed    import embed
from pipeline.db_setup import setup
from pipeline.load     import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Perfume data ingestion pipeline")
    parser.add_argument("--input",  required=True, help="Path to perfumes JSONL file")
    parser.add_argument("--device", default="cpu",  help="Embedding device: cpu | cuda (default: cpu)")
    parser.add_argument("--failed", default="failed.jsonl", help="Output path for failed records")
    args = parser.parse_args()

    logger.info("=== Perfume Ingestion Pipeline ===")
    logger.info("Input : %s", args.input)
    logger.info("Device: %s", args.device)

    # Count for progress bars
    total = count_lines(args.input)
    logger.info("Records in file: %d", total)

    # Stage 1 — Extract
    logger.info("--- Stage 1: Extract ---")
    records = extract(args.input)

    # Stage 2 — Transform
    logger.info("--- Stage 2: Transform ---")
    records = transform(records)

    # Stage 3 — Embed  (load model once here)
    logger.info("--- Stage 3: Embed (loading BGE-M3 on %s) ---", args.device)
    embedder = init_bge_embedder(device=args.device)
    records = embed(records, embedder, total=total)

    # Stage 4 — DB Setup
    logger.info("--- Stage 4: DB Setup ---")
    client = setup()

    # Stage 5 — Load
    logger.info("--- Stage 5: Load ---")
    load(records, client, failed_path=args.failed, total=total)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()

"""
Perfume search tools — plain Python functions (no MCP).
Shared utilities for embedding, Milvus search, reranking, and image URL derivation.
"""
import re
import sys
from pathlib import Path

from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embed_into_milvus.utils import init_bge_embedder, embed_text_bge

MILVUS_URI      = "http://localhost:19530"
MILVUS_TOKEN    = "root:Milvus"
DB_NAME         = "perfume_db"
COLLECTION_NAME = "perfume_collection"

_embedder = init_bge_embedder()
_client   = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
_client.using_database(DB_NAME)


def embed_query(extracted_moods: list[str], extracted_accords: list[str]) -> list[float]:
    """Embed extracted moods and accords into a 1024-dim query vector using BGE-M3."""
    text_summary = f"Moods: {', '.join(extracted_moods)} Accords: {', '.join(extracted_accords)}"
    return embed_text_bge(_embedder, [text_summary])


def search_milvus(
    query_vector: list[float],
    preferred_gender: str = "",
    top_k: int = 20,
) -> list[dict]:
    """Search perfume_collection in Milvus. Filters by gender if provided. Returns top_k candidates."""
    filter_expr = None
    if preferred_gender:
        filter_expr = f'gender == "{preferred_gender}" or gender == "unisex"'

    results = _client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="moods_embedding",
        search_params={"metric_type": "COSINE"},
        limit=top_k,
        filter=filter_expr,
        output_fields=["id", "name", "brand", "description", "url", "gender", "main_accords", "moods"],
    )

    candidates = []
    for hit in results[0]:
        candidates.append({
            "perfume_id":   hit["id"],
            "name":         hit["entity"]["name"],
            "brand":        hit["entity"]["brand"],
            "description":  hit["entity"]["description"],
            "url":          hit["entity"]["url"],
            "gender":       hit["entity"]["gender"],
            "main_accords": [a.strip() for a in hit["entity"]["main_accords"].split(",")],
            "moods":        [m.strip() for m in (hit["entity"].get("moods") or "").split(",") if m.strip()],
            "search_score": hit["distance"],
            "rerank_score": 0.0,
        })
    return candidates


def rerank_by_past_accords(candidates: list[dict], past_accords: list[str]) -> list[dict]:
    """Rerank candidates by overlap with user's historical preferred accords. Returns top-5."""
    if not past_accords:
        return candidates[:5]

    past_set = {a.lower() for a in past_accords}
    for c in candidates:
        overlap = len({a.lower() for a in c["main_accords"]} & past_set)
        c["rerank_score"] = overlap / len(past_set)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:5]


def extract_image_from_url(url: str) -> str:
    """Derive the Fragrantica social-card image URL from a perfume page URL.

    Pattern:
      page:  https://www.fragrantica.com/perfume/Brand/Name-{id}.html
      image: https://www.fragrantica.com/mdimg/perfume-social-cards/en-p_c_{id}.jpeg
    """
    match = re.search(r"-(\d+)\.html$", url or "")
    if not match:
        return ""
    return f"https://www.fragrantica.com/mdimg/perfume-social-cards/en-p_c_{match.group(1)}.jpeg"

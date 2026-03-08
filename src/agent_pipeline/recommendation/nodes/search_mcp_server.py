"""
MCP server exposing perfume search tools:
- load_user_history
- embed_query
- search_milvus
- rerank_by_past_accords
"""
import re
import sys
from pathlib import Path


from mcp.server.fastmcp import FastMCP
from pymilvus import MilvusClient

_nodes_dir = Path(__file__).resolve().parent          # nodes/
_agent_pipeline = _nodes_dir.parent.parent            # src/agent_pipeline/
_src = _agent_pipeline.parent                         # src/
sys.path.insert(0, str(_agent_pipeline))
sys.path.insert(0, str(_src))

from embed_into_milvus.utils import init_bge_embedder, embed_text_bge

mcp = FastMCP("perfume-search")

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
DB_NAME = "perfume_db"
COLLECTION_NAME = "perfume_collection"

GENDER_MAP = {
    "For Men": "men",
    "For Women": "women",
    "Unisex": "unisex",
}

embedder = init_bge_embedder()


@mcp.tool()
def embed_query(extracted_accords: list[str]) -> list[float]:
    """Embed extracted mood accords into a 1024-dim query vector using BGE-M3."""
    return embed_text_bge(embedder, extracted_accords)


@mcp.tool()
def search_milvus(
    query_vector: list[float],
    preferred_gender: str = "",
    top_k: int = 20,
) -> list[dict]:
    """
    Search perfume_collection in Milvus using the query vector.
    Filters by preferred_gender (also includes unisex). Returns top_k candidates.
    """
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    client.using_database(DB_NAME)

    filter_expr = ""
    if preferred_gender:
        gender_val = preferred_gender
        filter_expr = f'gender == "{gender_val}" or gender == "unisex"'

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="moods_embedding",
        search_params={"metric_type": "COSINE"},
        limit=top_k,
        filter=filter_expr or None,
        output_fields=["id", "name", "brand", "description", "url", "gender", "main_accords"],
    )

    candidates = []
    for hit in results[0]:
        candidates.append({
            "perfume_id": hit["id"],
            "name": hit["entity"]["name"],
            "brand": hit["entity"]["brand"],
            "description": hit["entity"]["description"],
            "url": hit["entity"]["url"],
            "gender": hit["entity"]["gender"],
            "main_accords": [a.strip() for a in hit["entity"]["main_accords"].split(",")],
            "search_score": hit["distance"],
            "rerank_score": 0.0,
        })

    return candidates


@mcp.tool()
def rerank_by_past_accords(candidates: list[dict], past_accords: list[str]) -> list[dict]:
    """
    Rerank candidates by overlap with user's historical preferred accords.
    Score = overlapping accords / total past accords. Returns top-5.
    """
    if not past_accords:
        return candidates[:5]

    past_set = set(a.lower() for a in past_accords)

    for candidate in candidates:
        candidate_accords = set(a.lower() for a in candidate["main_accords"])
        overlap = len(candidate_accords & past_set)
        candidate["rerank_score"] = overlap / len(past_set)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:5]


@mcp.tool()
def extract_image_from_url(url: str) -> str:
    """Derive the Fragrantica social card image URL from a perfume page URL.

    Fragrantica image URLs follow a fixed pattern:
      page:  https://www.fragrantica.com/perfume/Brand/Name-{id}.html
      image: https://www.fragrantica.com/mdimg/perfume-social-cards/en-p_c_{id}.jpeg
    """
    match = re.search(r"-(\d+)\.html$", url)
    if not match:
        raise ValueError(f"Could not extract perfume ID from URL: {url}")
    perfume_id = match.group(1)
    return f"https://www.fragrantica.com/mdimg/perfume-social-cards/en-p_c_{perfume_id}.jpeg"


if __name__ == "__main__":
    mcp.run(transport="stdio")

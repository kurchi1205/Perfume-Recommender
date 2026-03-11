import json
import logging
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from schemas import CandidatePerfume

_MCP_SERVER = str(Path(__file__).resolve().parent / "search_mcp_server.py")

logger = logging.getLogger(__name__)


def _parse_mcp_result(raw):
    """Parse MCP tool output into a Python object.
    The adapter returns each scalar as a separate content block:
      [{'type': 'text', 'text': '0.123', 'id': '...'}, ...]
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "type" in raw[0]:
        values = [json.loads(block["text"]) for block in raw if block.get("type") == "text"]
        return values[0] if len(values) == 1 else values
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _rerank_by_extracted_accords(candidates: list, extracted_accords: list, top_k: int = 20) -> list:
    """
    Rerank Milvus candidates against the mood-extracted accords.

    Two accord signals:
      - token_score:  word-level overlap between extracted accord phrases and
                      main_accord tokens (catches "tropical" inside "retro tropical")
      - exact_score:  full extracted phrase matches a main_accord exactly
                      (rare but strong signal)

    Final score = 0.6 * search_score + 0.3 * token_score + 0.1 * exact_score
    """
    extracted_phrases = {a.lower().strip() for a in extracted_accords}
    extracted_tokens = set()
    for a in extracted_accords:
        extracted_tokens.update(a.lower().split())

    for c in candidates:
        main_phrases = {a.lower().strip() for a in c["main_accords"]}
        main_tokens = set()
        for a in c["main_accords"]:
            main_tokens.update(a.lower().split())

        exact_score = len(extracted_phrases & main_phrases) / len(extracted_phrases) if extracted_phrases else 0
        token_score = len(extracted_tokens & main_tokens) / len(extracted_tokens) if extracted_tokens else 0

        c["rerank_score"] = 0.6 * c["search_score"] + 0.3 * token_score + 0.1 * exact_score

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


async def _run_search(extracted_accords: list, extracted_moods: list, state) -> list:
    client = MultiServerMCPClient({
        "perfume-search": {
            "command": sys.executable,
            "args": [_MCP_SERVER],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}

    # Step 1: Embed extracted accords into a query vector
    raw_vector = await by_name["embed_query"].ainvoke({"extracted_moods": extracted_moods, "extracted_accords": extracted_accords})
    query_vector = _parse_mcp_result(raw_vector)
    print(f"[Step 1] query_vector: {len(query_vector)}-dim")

    # Step 2: Search Milvus with the query vector
    raw_candidates = await by_name["search_milvus"].ainvoke({"query_vector": query_vector, "preferred_gender": ""})
    candidates_raw = _parse_mcp_result(raw_candidates)
    candidates = []
    for c in candidates_raw:
        try:
            candidates.append(CandidatePerfume.model_validate(c).model_dump())
        except Exception as e:
            logger.warning("[search] skipping invalid candidate %s: %s", c.get("name", "?"), e)
    state["candidates"] = candidates
    print(f"[Step 2] candidates: {len(candidates)} results")

    # Step 3: Rerank by extracted accords
    reranked = _rerank_by_extracted_accords(candidates, extracted_accords)
    return reranked


async def search_node(state):
    reranked = await _run_search(
        extracted_moods=state["extracted_moods"],
        extracted_accords=state["extracted_accords"],
        state=state
    )

    state["reranked"] = reranked
    return state

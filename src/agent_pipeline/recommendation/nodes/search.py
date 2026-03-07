import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


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


async def _run_search(extracted_accords: list, state) -> list:
    client = MultiServerMCPClient({
        "perfume-search": {
            "command": sys.executable,
            "args": ["nodes/search_mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}

    # Step 1: Embed extracted accords into a query vector
    raw_vector = await by_name["embed_query"].ainvoke({"extracted_accords": extracted_accords})
    query_vector = _parse_mcp_result(raw_vector)
    print(f"[Step 1] query_vector: {len(query_vector)}-dim")
    print(f"[Step 1] query_vector: {query_vector}")

    # Step 2: Search Milvus with the query vector
    raw_candidates = await by_name["search_milvus"].ainvoke({"query_vector": query_vector, "preferred_gender": ""})
    candidates = _parse_mcp_result(raw_candidates)
    state["candidates"] = candidates
    print(f"[Step 2] extracted accords: {extracted_accords}")
    print(f"[Step 2] candidates: {len(candidates)} results")

    return candidates


def search_node(state):
    candidates = asyncio.run(_run_search(
        extracted_accords=state["extracted_accords"],
        state=state
    ))

    state["reranked"] = candidates
    return state

"""
Explanation Generator — takes the top-5 reranked candidates, generates a 1-2 sentence
explanation per result, derives image URLs, and validates into RecommendedPerfume.
LLM calls are parallelised with asyncio.gather.
"""
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from schemas import RecommendedPerfume
from nodes.search_mcp_server import extract_image_from_url

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

logger = logging.getLogger(__name__)

llm = ChatOllama(model="mistral:latest", temperature=0.4)

SYSTEM_PROMPT = (
    "You are a perfume expert. Write 1-2 sentences explaining why this perfume "
    "matches what the user is looking for. Be specific — mention the accords or "
    "mood qualities that make it a good fit. Do not start with 'This perfume'."
)


# ── Per-result explanation ────────────────────────────────────────────────────

async def _explain(perfume: dict, intent_summary: str) -> str:
    prompt = (
        f"User wanted: {intent_summary}\n"
        f"Perfume: {perfume['name']} by {perfume['brand']}\n"
        f"Accords: {', '.join(perfume.get('main_accords', []))}\n"
        f"Description: {(perfume.get('description') or '')[:150]}\n\n"
        "Write 1-2 sentences."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning("[explanation_generator] failed for %s: %s", perfume.get("name"), e)
        return ""


# ── Node ──────────────────────────────────────────────────────────────────────

async def explanation_generator_node(state: dict) -> dict:
    reranked       = state.get("reranked", [])
    intent_summary = state.get("intent_summary", "")

    top5 = reranked[:5]
    if not top5:
        logger.warning("[explanation_generator] no candidates to explain")
        return {"recommendations": []}

    # Parallelise LLM calls across the 5 results
    explanations = await asyncio.gather(*[_explain(p, intent_summary) for p in top5])

    recommendations = []
    for perfume, explanation in zip(top5, explanations):
        perfume["image_url"]   = extract_image_from_url(perfume.get("url", ""))
        perfume["explanation"] = explanation
        try:
            recommendations.append(RecommendedPerfume.model_validate(perfume).model_dump())
        except Exception as e:
            logger.warning("[explanation_generator] skipping invalid perfume %s: %s", perfume.get("name", "?"), e)

    logger.info("[explanation_generator] explained: %s", [r["name"] for r in recommendations])
    return {"recommendations": recommendations}

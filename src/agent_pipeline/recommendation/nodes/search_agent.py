"""
Search Agent — combines query building, Milvus retrieval, and LLM reranking.

Internal steps:
  1. Build query text + embed (BGE-M3)
  2. Search Milvus → top-20 candidates
  3. Accord overlap pre-score
  4. LLM agent reranks via score_perfumes → normalize_scores → rerank_candidates
"""
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from schemas import CandidatePerfume
from nodes.search_mcp_server import search_milvus as _milvus_search, embed_query as _embed_query

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

logger = logging.getLogger(__name__)

llm = ChatOllama(model="mistral:latest", temperature=0)

# ── Step 1: Query Builder ─────────────────────────────────────────────────────

def _build_query(moods: list, accords: list, intent_summary: str, mood_input: str):
    query_text = (
        f"{intent_summary}. "
        f"Scent accords: {', '.join(accords)}. "
        f"Mood: {', '.join(moods)}."
    )
    query_vector = _embed_query(moods, accords)

    lower = mood_input.lower()
    if any(w in lower for w in ["woman", "women", "feminine", "female", "her ", " she "]):
        gender_filter = "female"
    elif any(w in lower for w in ["man", "men", "masculine", "male", "him ", " he "]):
        gender_filter = "male"
    else:
        gender_filter = ""

    return query_text, query_vector, gender_filter


# ── Step 2: Retrieval ─────────────────────────────────────────────────────────

def _search_milvus(query_vector: list, gender_filter: str = "", top_k: int = 20) -> list:
    candidates_raw = _milvus_search(query_vector, preferred_gender=gender_filter, top_k=top_k)
    candidates = []
    for raw in candidates_raw:
        try:
            candidates.append(CandidatePerfume.model_validate(raw).model_dump())
        except Exception as e:
            logger.warning("[search_agent] skipping invalid candidate %s: %s", raw.get("name", "?"), e)
    logger.info("[search_agent] %d valid candidates from Milvus", len(candidates))
    return candidates


# ── Step 3: Accord overlap pre-score ─────────────────────────────────────────

def _accord_overlap_score(candidates: list, extracted_accords: list) -> list:
    extracted_phrases = {a.lower().strip() for a in extracted_accords}
    extracted_tokens  = {t for a in extracted_accords for t in a.lower().split()}

    for c in candidates:
        main_phrases = {a.lower().strip() for a in c["main_accords"]}
        main_tokens  = {t for a in c["main_accords"] for t in a.lower().split()}

        exact = len(extracted_phrases & main_phrases) / len(extracted_phrases) if extracted_phrases else 0
        token = len(extracted_tokens  & main_tokens)  / len(extracted_tokens)  if extracted_tokens  else 0
        c["rerank_score"] = 0.6 * c["search_score"] + 0.3 * token + 0.1 * exact

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# ── Step 4: LLM reranker tools ────────────────────────────────────────────────

SCORER_SYSTEM = """\
You are a perfume expert. Score each candidate 0-10 on how well it matches the user's mood and accords.
Rules:
- Use the FULL range: great matches 8-10, poor matches 0-3, mediocre 4-6.
- Consider BOTH mood words AND scent accords.
- Scores must be meaningfully spread — do NOT give everyone similar scores.
- Return ONLY a JSON array of numbers (one per candidate) in the same order. No extra text."""


@tool
def score_perfumes(candidates_json: str, moods: str, accords: str) -> str:
    """
    Ask the LLM to score each perfume candidate 0-10 based on mood/accord alignment.
    candidates_json: JSON array of candidate dicts.
    moods: comma-separated mood words.
    accords: comma-separated scent accords.
    Returns a JSON array of float scores in the same order.
    """
    candidates = json.loads(candidates_json)
    lines = [
        f"User mood: {moods}",
        f"Desired scent accords: {accords}",
        "",
        "Perfume candidates (score each 0-10):",
    ]
    for i, c in enumerate(candidates, 1):
        acc        = ", ".join(c.get("main_accords", [])) or "unknown"
        perf_moods = ", ".join(c.get("moods", [])) or "unknown"
        desc       = (c.get("description") or "")[:120]
        lines.append(f"{i}. \"{c['name']}\" by {c.get('brand', '?')}")
        lines.append(f"   Accords: {acc}")
        lines.append(f"   Moods: {perf_moods}")
        if desc:
            lines.append(f"   Description: {desc}")
    lines.append(f"\nReturn a JSON array of exactly {len(candidates)} scores (0-10), spread across the range.")

    response = llm.invoke([HumanMessage(content=SCORER_SYSTEM + "\n\n" + "\n".join(lines))])
    match = re.search(r"\[[\d\s.,]+\]", response.content)
    if match:
        scores = json.loads(match.group())
        if len(scores) == len(candidates):
            return json.dumps([float(s) for s in scores])

    logger.warning("[search_agent] score parse failed, defaulting to 5.0")
    return json.dumps([5.0] * len(candidates))


@tool
def normalize_scores(llm_scores_json: str, candidates_json: str) -> str:
    """
    Normalise LLM scores and rerank_score of each candidate to [0, 1].
    Returns JSON array of dicts with llm_norm, llm_score, rerank_score per candidate.
    """
    llm_scores = json.loads(llm_scores_json)
    candidates = json.loads(candidates_json)

    def norm(values):
        lo, hi = min(values), max(values)
        return [1.0] * len(values) if hi == lo else [(v - lo) / (hi - lo) for v in values]

    llm_norm = norm(llm_scores)
    result = [
        {
            "name":         c["name"],
            "llm_score":    llm_scores[i],
            "rerank_score": c.get("rerank_score", 0.0),
            "llm_norm":     llm_norm[i],
        }
        for i, c in enumerate(candidates)
    ]
    return json.dumps(result)


@tool
def rerank_candidates(normalized_json: str, candidates_json: str) -> str:
    """
    Compute final_score = 0.9 * llm_norm + 0.1 * rerank_score for each candidate.
    Returns JSON array of top-10 candidates sorted by final_score descending.
    """
    normalized = json.loads(normalized_json)
    candidates = json.loads(candidates_json)
    for c, n in zip(candidates, normalized):
        c["llm_score"]   = n["llm_score"]
        c["final_score"] = 0.9 * n["llm_norm"] + 0.1 * n["rerank_score"]

    return json.dumps(sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:10])


TOOLS = [score_perfumes, normalize_scores, rerank_candidates]

AGENT_SYSTEM = """\
You are a reranking agent. Follow these steps in order:
1. Call score_perfumes with the candidates, moods and accords.
2. Call normalize_scores with the scores and candidates.
3. Call rerank_candidates with the normalized output and candidates.
Return the final JSON array from rerank_candidates as your answer."""


def _llm_rerank(candidates: list, moods: list, accords: list) -> list:
    agent = create_agent(llm, tools=TOOLS, system_prompt=AGENT_SYSTEM)
    response = agent.invoke({"messages": [HumanMessage(
        content=(
            f"candidates_json: {json.dumps(candidates)}\n"
            f"moods: {', '.join(moods)}\n"
            f"accords: {', '.join(accords)}"
        )
    )]})

    last = response["messages"][-1].content
    try:
        match = re.search(r"\[.*\]", last, re.DOTALL)
        return json.loads(match.group()) if match else candidates[:10]
    except Exception:
        logger.warning("[search_agent] rerank parse failed, falling back to top-10")
        return candidates[:10]


# ── Node ──────────────────────────────────────────────────────────────────────

def search_node(state: dict) -> dict:
    moods      = state.get("extracted_moods", [])
    accords    = state.get("extracted_accords", [])
    summary    = state.get("intent_summary", "")
    mood_input = state.get("mood_input", "")

    # Step 1 — build query
    query_text, query_vector, gender_filter = _build_query(moods, accords, summary, mood_input)
    logger.info("[search_agent] query=%r gender=%r", query_text[:80], gender_filter)

    # Step 2 — retrieve
    candidates = _search_milvus(query_vector, gender_filter)
    if not candidates:
        logger.warning("[search_agent] no candidates returned from Milvus")
        return {"query_text": query_text, "reranked": []}

    # Step 3 — accord overlap pre-score
    candidates = _accord_overlap_score(candidates, accords)

    # Step 4 — LLM rerank
    reranked = _llm_rerank(candidates, moods, accords)
    logger.info("[search_agent] top reranked: %s", [c.get("name") for c in reranked[:5]])

    return {"query_text": query_text, "reranked": reranked}

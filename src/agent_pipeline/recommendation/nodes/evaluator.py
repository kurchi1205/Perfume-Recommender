"""
Evaluator node — LangChain agent with tools for LLM scoring + reranking.

The agent uses three tools in sequence:
  1. score_perfumes       — LLM rates each candidate 0-10 against the mood
  2. normalize_scores     — normalise both llm_score and rerank_score to [0,1]
  3. rerank_candidates    — combine 70% LLM + 30% rerank, return top-5
"""
import json
import logging
import re

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from schemas import ScoredPerfume

logger = logging.getLogger(__name__)

llm = ChatOllama(model="mistral", temperature=0)

SCORER_SYSTEM = """\
You are a perfume expert. Score each candidate 0-10 on how well its accords match \
the user's mood. Return ONLY a JSON array of numbers in the same order. No extra text."""


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def score_perfumes(candidates_json: str, moods: str, accords: str) -> str:
    """
    Ask the LLM to score each perfume candidate 0-10 based on mood/accord alignment.
    candidates_json: JSON array of candidate dicts (must have 'name','brand','main_accords').
    moods:   comma-separated extracted moods.
    accords: comma-separated extracted accords.
    Returns a JSON array of float scores in the same order.
    """
    candidates = json.loads(candidates_json)
    lines = [f"User mood: {moods}", f"Scent accords: {accords}", "", "Perfume candidates:"]
    for i, c in enumerate(candidates, 1):
        acc = ", ".join(c.get("main_accords", [])) or "unknown"
        lines.append(f"{i}. \"{c['name']}\" by {c.get('brand','?')} — accords: {acc}")
    lines += ["", f"Return a JSON array of {len(candidates)} scores (0-10)."]

    response = llm.invoke([
        HumanMessage(content=SCORER_SYSTEM + "\n\n" + "\n".join(lines))
    ])
    text = response.content
    match = re.search(r"\[[\d\s.,]+\]", text)
    if match:
        scores = json.loads(match.group())
        if len(scores) == len(candidates):
            return json.dumps([float(s) for s in scores])

    logger.warning("[evaluator] score parse failed, defaulting to 5.0")
    return json.dumps([5.0] * len(candidates))


@tool
def normalize_scores(llm_scores_json: str, candidates_json: str) -> str:
    """
    Normalise LLM scores and the existing rerank_score of each candidate to [0, 1].
    Returns JSON array of dicts: [{llm_norm, rerank_norm, llm_score, rerank_score, name}, ...]
    """
    llm_scores = json.loads(llm_scores_json)
    candidates = json.loads(candidates_json)

    def norm(values):
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    rerank_vals = [c.get("rerank_score", 0.0) for c in candidates]
    llm_norm    = norm(llm_scores)

    result = [
        {
            "name":         c["name"],
            "llm_score":    llm_scores[i],
            "rerank_score": rerank_vals[i],
            "llm_norm":     llm_norm[i],
        }
        for i, c in enumerate(candidates)
    ]
    return json.dumps(result)


@tool
def rerank_candidates(normalized_json: str, candidates_json: str) -> str:
    """
    Compute final_score = 0.7 * llm_norm + 0.3 * rerank_norm for each candidate.
    Returns JSON array of top-5 candidates sorted by final_score descending.
    """
    normalized = json.loads(normalized_json)
    candidates = json.loads(candidates_json)

    for c, n in zip(candidates, normalized):
        c["llm_score"]   = n["llm_score"]
        c["final_score"] = 0.7 * n["llm_norm"] + 0.3 * n["rerank_score"]
        

    top5 = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:5]
    return json.dumps(top5)


# ── Node ───────────────────────────────────────────────────────────────────────

TOOLS = [score_perfumes, normalize_scores, rerank_candidates]

AGENT_SYSTEM = """\
You are an evaluator agent. Follow these steps in order:
1. Call score_perfumes with the candidates, moods and accords.
2. Call normalize_scores with the scores and candidates.
3. Call rerank_candidates with the normalized output and candidates.
Return the final JSON array from rerank_candidates as your answer."""


def evaluate_node(state: dict) -> dict:
    candidates = state.get("reranked", [])
    moods      = state.get("extracted_moods", [])
    accords    = state.get("extracted_accords", [])

    if not candidates:
        return {"reranked": []}

    candidates_json = json.dumps(candidates)
    moods_str       = ", ".join(moods)
    accords_str     = ", ".join(accords)

    agent = create_agent(llm, tools=TOOLS, system_prompt=AGENT_SYSTEM)
    response = agent.invoke({"messages": [HumanMessage(
        content=(
            f"candidates_json: {candidates_json}\n"
            f"moods: {moods_str}\n"
            f"accords: {accords_str}"
        )
    )]})

    # Extract final JSON from last message
    last = response["messages"][-1].content
    try:
        match = re.search(r"\[.*\]", last, re.DOTALL)
        raw_top5 = json.loads(match.group()) if match else candidates[:5]
    except Exception:
        logger.warning("[evaluator] could not parse agent output, falling back to top-5")
        raw_top5 = candidates[:5]

    top5 = []
    for c in raw_top5:
        try:
            top5.append(ScoredPerfume.model_validate(c).model_dump())
        except Exception as e:
            logger.warning("[evaluator] skipping invalid scored perfume %s: %s", c.get("name", "?"), e)

    if not top5:
        top5 = candidates[:5]

    logger.info("[evaluator] top-5: %s", [c["name"] for c in top5])
    return {"reranked": top5}

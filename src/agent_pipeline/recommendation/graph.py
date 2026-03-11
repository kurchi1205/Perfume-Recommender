import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from states import (
    RecommendationInputState,
    RecommendationWorkingState,
    RecommendationOutputState
)
from nodes.accord_extractor import accord_extracting_agent
from nodes.mood_extractor import mood_extracting_agent
from nodes.search import search_node
from nodes.evaluator import evaluate_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def extract_mood(merged: dict):
    input_state = {k: merged[k] for k in RecommendationInputState.__annotations__ if k in merged}
    state = {k: merged[k] for k in RecommendationWorkingState.__annotations__ if k in merged}
    result = mood_extracting_agent(input_state, state)
    return {"extracted_moods": result["extracted_moods"]}

def extract_accord(merged: dict):
    input_state = {k: merged[k] for k in RecommendationInputState.__annotations__ if k in merged}
    state = {k: merged[k] for k in RecommendationWorkingState.__annotations__ if k in merged}
    result = accord_extracting_agent(input_state, state)
    return {"extracted_accords": result["extracted_accords"]}


def build_graph():
    graph = StateGraph(
        RecommendationWorkingState,
        input=RecommendationInputState,
        output=RecommendationOutputState,
    )

    graph.add_node("extract_mood", extract_mood)
    graph.add_node("extract_accord", extract_accord)
    graph.add_node("search", search_node)
    graph.add_node("evaluator", evaluate_node)

    graph.add_edge(START, "extract_mood")
    graph.add_edge(START, "extract_accord")
    graph.add_edge("extract_mood", "search")
    graph.add_edge("extract_accord", "search")
    graph.add_edge("search", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()

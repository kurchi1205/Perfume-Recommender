import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from states import (
    RecommendationInputState,
    RecommendationWorkingState,
)
from nodes.mood_extractor import mood_extracting_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def extract_mood(merged: dict):
    print(merged)
    input_state = {k: merged[k] for k in RecommendationInputState.__annotations__ if k in merged}
    state = {k: merged[k] for k in RecommendationWorkingState.__annotations__ if k in merged}
    return mood_extracting_agent(input_state, state)


def build_graph():
    graph = StateGraph(
        RecommendationWorkingState,
        input=RecommendationInputState,
        output=RecommendationWorkingState,
    )

    graph.add_node("extract_mood", extract_mood)

    graph.add_edge(START, "extract_mood")
    graph.add_edge("extract_mood", END)

    return graph.compile()

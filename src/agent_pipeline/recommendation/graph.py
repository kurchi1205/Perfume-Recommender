import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from states import (
    RecommendationInputState,
    RecommendationWorkingState,
    RecommendationOutputState,
)
from nodes.taste_profile_agent import taste_profile_node
from nodes.search_agent import search_node
from nodes.explanation_generator import explanation_generator_node
from nodes.memory_update import memory_update_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_graph():
    graph = StateGraph(
        RecommendationWorkingState,
        input=RecommendationInputState,
        output=RecommendationOutputState,
    )

    graph.add_node("taste_profile_agent",   taste_profile_node)
    graph.add_node("search_agent",          search_node)
    graph.add_node("explanation_generator", explanation_generator_node)
    graph.add_node("memory_update",         memory_update_node)

    graph.add_edge(START,                   "taste_profile_agent")
    graph.add_edge("taste_profile_agent",   "search_agent")
    graph.add_edge("search_agent",          "explanation_generator")
    graph.add_edge("explanation_generator", "memory_update")
    graph.add_edge("memory_update",         END)

    return graph.compile()

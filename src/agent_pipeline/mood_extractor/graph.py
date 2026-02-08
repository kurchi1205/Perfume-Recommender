from langgraph.graph import StateGraph, START, END
from states import (
    PerfumeInputState,
    PerfumeWorkingState,
    PerfumeOutputState,
)
from nodes import (
    initialize,
    next_perfume,
    extract_moods,
    assemble_output,
    save_jsonl,   # optional: writes updated JSON to disk
)


def build_graph():
    graph = StateGraph(
        PerfumeWorkingState,
        input=PerfumeInputState,
        output=PerfumeOutputState,
    )

    # -----------------------
    # Nodes
    # -----------------------
    graph.add_node("initialize", initialize)
    graph.add_node("next_perfume", next_perfume)
    graph.add_node("extract_moods", extract_moods)
    graph.add_node("assemble_output", assemble_output)

    # -----------------------
    # Edges
    # -----------------------
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "next_perfume")
    graph.add_edge("next_perfume", "extract_moods")
    graph.add_edge("extract_moods", "assemble_output")

    # -----------------------
    # Loop condition
    # -----------------------
    def should_continue(state: PerfumeWorkingState):
        return "loop" if state["current_perfume"] else "done"

    graph.add_conditional_edges(
        "assemble_output",
        should_continue,
        {
            "loop": "next_perfume",
            "done": "assemble_output",
        }
    )

    graph.add_edge("assemble_output", END)

    return graph

"""
LangGraph-based agent for setting up Milvus DB and inserting perfume data.

Reuses the tools defined in update_db_agent.py but orchestrates them
via a LangGraph StateGraph instead of a ReAct agent.
"""

import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from update_db_tools import (
    check_if_db_exists,
    create_milvus_db,
    check_if_collection_exists,
    create_collection,
    insert_into_collection,
    DB_NAME,
    COLLECTIONS_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a database engineer. You MUST call the provided tools to complete tasks.\n"
    "Follow these steps in order:\n"
    "1. Call check_if_db_exists to check whether the database exists.\n"
    "2. If the database does not exist, call create_milvus_db to create it.\n"
    "3. Call check_if_collection_exists to check whether the collection exists.\n"
    "4. If the collection does not exist, call create_collection to create it.\n"
)

TOOLS = [
    check_if_db_exists,
    create_milvus_db,
    check_if_collection_exists,
    create_collection,
    insert_into_collection,
]


def call_model(state: MessagesState):
    """Invoke the LLM with the current messages and bound tools."""
    model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(TOOLS)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_langgraph_agent(input_path: str | None = None):
    """Build and compile the LangGraph agent.

    Returns the compiled StateGraph ready for .invoke().
    """
    if input_path is None:
        input_path = str(Path("../../../datasets/perfumes_with_moods.jsonl"))

    graph = StateGraph(MessagesState)

    # Nodes
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    # Edges
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition)
    graph.add_edge("tools", "call_model")

    agent = graph.compile()
    return agent


if __name__ == "__main__":
    INPUT_PATH = str(Path("../../../datasets/perfumes_with_moods.jsonl"))
    agent = build_langgraph_agent(input_path=INPUT_PATH)
    config = {"configurable": {"thread_id": "1"}}
    user_message_1 = (
        f"{SYSTEM_PROMPT}\n\n"
        f"INPUT_PATH: {INPUT_PATH}, "
        f"Database_name: {DB_NAME}, "
        f"Collection_name: {COLLECTIONS_NAME}\n\n"
        f"Set up the database '{DB_NAME}' and collection '{COLLECTIONS_NAME}', "
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message_1}]},
        config
    )

    user_message_2 = (
        f"Then insert data from {INPUT_PATH} into it."
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message_2}]},
        config
    )

    # Print final agent message
    for m in result['messages']:
        m.pretty_print()

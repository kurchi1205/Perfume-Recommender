import asyncio
import json
import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


llm = ChatOllama(model="mistral", temperature=0)


def _get_tool_result(messages):
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list) and content and isinstance(content[0], dict):
                return content[0].get("text", str(content[0]))
            return content
    return None


async def _enrich(reranked: list) -> list:
    client = MultiServerMCPClient({
        "perfume-search": {
            "command": sys.executable,
            "args": ["nodes/search_mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    agent = create_agent(llm, tools=tools, system_prompt="No parallel tool calls")

    enriched = []
    for perfume in reranked:
        resp = await agent.ainvoke({"messages": [
            HumanMessage(content=f'Use the extract_image_from_url tool with url="{perfume["url"]}".')
        ]})
        image_url = _get_tool_result(resp["messages"]) or ""
        enriched.append({**perfume, "image_url": image_url})
        print(f"[enricher] {perfume['name']}: {image_url}")

    return enriched


def result_enricher(state):
    reranked = state.get("reranked", [])
    enriched = asyncio.run(_enrich(reranked))
    return {"user_id": state["user_id"], "recommendations": enriched}

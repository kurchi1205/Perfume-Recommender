from typing import List
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.agents.middleware import dynamic_prompt, ModelRequest

import json


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    template="""
        You are a perfume sensory analyst.

        Given perfume data, extract 5 sensory mood of the fragrance.
        Base your answer ONLY on:
        - description
        - main accords
        - gender

        Return ONLY and only the valid JSON in the format, no additional text:

        {
        "moods": list of 5 abstract mood strings.
        }
        where "moods" is a list of 5 abstract mood strings. Dont include main accords.
        Perfume data:
    """
    perfume_content = request.runtime.context.get("perfume_content", "{}")
    final_prompt = template + perfume_content
    return final_prompt


def create_mood_model_agent():
    model = OllamaLLM(model="smollm2:latest", max_tokens=100, temperature=0.4)
    agent = create_agent(model, middleware=[user_role_prompt])
    return agent


def run_mood_model_agent(agent, perfume_content):
    result = agent.invoke(input={}, context={"perfume_content": json.dumps(perfume_content, ensure_ascii=False)})
    return result


def extract_moods_from_agent_result(result) -> list[str]:
    if not result or "messages" not in result:
        return []

    message = result["messages"][-1]  # last message
    content = message.content

    try:
        parsed = json.loads(content)
        return parsed.get("moods", [])
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    agent = create_mood_model_agent()
    perfume_content = {
        "description": "A warm and sensual fragrance for women.",
        "main_accords": ["spicy", "woody"],
        "gender": "women"
    }
    result = run_mood_model_agent(agent, perfume_content)
    moods = extract_moods_from_agent_result(result)
    print(moods)
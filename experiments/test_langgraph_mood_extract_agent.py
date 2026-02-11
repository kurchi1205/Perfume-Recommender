from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


def load_model():
    model = ChatOllama(model="smollm2:latest", max_tokens=100, temperature=0.5)
    return model

@tool
def get_scent_moods(
    moods: list[str]
) -> list[str]:
    """Return list of 5 abstract mood strings."""
    return moods


def create_mood_agent(model):
    mood_AGENT_PROMPT = (
    """You are a perfume sensory analyst.

    Task:
    - Parse natural language Perfume data (e.g., 'A warm and sensual fragrance for women.')
    - give EXACTLY 5 abstract sensory moods.

    Tool usage:
    - You MUST call `get_scent_moods`
    - Pass a LIST of 5 short mood strings to the tool
    - The tool call is your final answer
    - Return a list of 5 abstract mood strings
    """
    )

    mood_agent = create_agent(
        model,
        tools=[get_scent_moods],
        system_prompt=mood_AGENT_PROMPT,
    )

    return mood_agent


def query_mood_agent(mood_agent, perfume_content):
    QUERY_PROMPT = f"""
    You are a perfume sensory analyst.

        Given perfume data, extract 5 sensory mood of the fragrance.
        Base your answer ONLY on:
        - description
        - main accords
        - gender

        Return ONLY valid JSON with moods key
        "moods": "list of 5 abstract mood strings."
        where "moods" is a list of 5 abstract mood strings. Dont include main accords.
        Perfume data: {perfume_content}
    
    """
    for step in mood_agent.stream(
        {"messages": [{"role": "user", "content": QUERY_PROMPT}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    model = load_model()
    mood_agent = create_mood_agent(model)
    perfume_content = """
        "description": "A warm and sensual fragrance for women.",
        "main_accords": ["spicy", "woody"],
        "gender": "women"
    """
    query_mood_agent(mood_agent, perfume_content)
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


def load_model():
    model = ChatOllama(model="smollm2:latest", max_tokens=20, temperature=0)
    return model


@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"



@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


def create_cal_agent(model):
    CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "if attendees are not specified, never create attendees"
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
    )

    calendar_agent = create_agent(
        model,
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=CALENDAR_AGENT_PROMPT,
    )

    return calendar_agent


def query_cal_agent(calendar_agent):
    query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"

    for step in calendar_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    model = load_model()
    calendar_agent = create_cal_agent(model)
    query_cal_agent(calendar_agent) 
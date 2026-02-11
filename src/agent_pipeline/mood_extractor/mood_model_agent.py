import json
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class PerfumeMoods(BaseModel):
    """Schema for the LLM's mood extraction output."""

    moods: List[str]

    @field_validator("moods")
    @classmethod
    def validate_moods(cls, v: List[str]) -> List[str]:
        v = [mood.strip() for mood in v if mood.strip()]
        if len(v) != 5:
            raise ValueError(f"Expected exactly 5 moods, got {len(v)}: {v}")
        return v


MOOD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a perfume sensory analyst. "
        "Given perfume data, extract exactly 5 abstract sensory moods that describe "
        "the emotional feeling of the fragrance. "
        "Base your answer ONLY on the description, notes, main accords, and gender. "
        "Do NOT repeat the main accords as moods. "
        "Moods should be single abstract adjectives like: "
        "Romantic, Mysterious, Confident, Nostalgic, Serene, Energetic, Sultry, "
        "Sophisticated, Rebellious, Cozy, Ethereal, Playful, Melancholic, Bold, Warm."
    )),
    ("human", (
        "Extract exactly 5 moods for this perfume:\n\n"
        "{perfume_content}\n\n"
        'Respond with JSON: {{"moods": ["Mood1", "Mood2", "Mood3", "Mood4", "Mood5"]}}'
    )),
])


def create_mood_extraction_chain():
    """Create the mood extraction chain with structured output."""
    model = ChatOllama(
        model="smollm2:latest",
        temperature=0.4,
        num_predict=150,
    )

    structured_model = model.with_structured_output(
        PerfumeMoods,
        method="json_schema",
        include_raw=True,
    )

    chain = MOOD_EXTRACTION_PROMPT | structured_model
    return chain


def extract_moods(chain, perfume_content: str) -> list[str]:
    """
    Invoke the chain with retry logic.
    Returns a list of exactly 5 mood strings, or an empty list on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = chain.invoke({"perfume_content": perfume_content})

            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")

            if parsing_error is not None:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES}: parsing error: {parsing_error}"
                )
                continue

            if parsed is None:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES}: parsed result is None"
                )
                continue

            return parsed.moods

        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES}: exception: {e}"
            )
            continue

    logger.error(f"All {MAX_RETRIES} attempts failed for perfume content")
    return []


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    chain = create_mood_extraction_chain()

    perfume_content = (
        "Gender: women\n"
        "Description: A warm and sensual amber floral fragrance for women.\n"
        "Top notes: Citruses\n"
        "Middle notes: Floral Notes\n"
        "Base notes: Musk, Vanille\n"
        "Main accords: floral, musky, vanilla, powdery, citrus, fresh, sweet"
    )

    moods = extract_moods(chain, perfume_content)
    print(f"Extracted moods: {moods}")

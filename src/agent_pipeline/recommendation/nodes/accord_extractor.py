import base64
import json
import logging
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from PIL import Image

from states import RecommendationWorkingState
from schemas import ExtractedList

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

llm = ChatOpenRouter(model="qwen/qwen3-vl-235b-a22b-thinking", temperature=0.5)

SYSTEM_PROMPT = """You are a perfume mood extractor. Your job is to analyze the user's mood description or image and return a list of scent accords.
Rules:
- Return ONLY an array of accord strings
- Choose 3 to 7 accords that best capture the mood or feeling conveyed
- Use lowercase accord names
- Do not include explanations, strictly give an array of accords
"""


def convert_to_base64(pil_image_path):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(pil_image_path)
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def form_user_content(data):
    def prompt_func(data):
        text = data.get("text")
        image_url = data.get("image_url")
        content_parts = []

        if image_url:
            content_parts.append({"type": "image", "url": image_url})
        if text:
            content_parts.append({"type": "text", "text": text})

        return [HumanMessage(content=content_parts)]

    return prompt_func(data)


def accord_extracting_agent(input_state, state: RecommendationWorkingState):
    data = {}
    if input_state["input_type"] == "text":
        data["text"] = input_state["mood_input"]
    else:
        data["image_url"] = input_state["mood_input"]  # served HTTP URL


    agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT
    )

    messages = form_user_content(data)
    for attempt in range(1, MAX_RETRIES + 1):
        response = agent.invoke({"messages": messages})
        try:
            validated = ExtractedList(items=response["messages"][-1].content)
            state["extracted_accords"] = validated.items
            break
        except Exception as e:
            logger.info("Accord extraction attempt %d/%d failed validation: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logger.error("All accord extraction attempts failed — defaulting to []")
                state["extracted_accords"] = []

    return state


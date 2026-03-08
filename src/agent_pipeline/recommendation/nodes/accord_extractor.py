import logging
import sys
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, field_validator
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

import base64
from PIL import Image
from io import BytesIO
import json

from states import RecommendationWorkingState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

llm = ChatOllama(model="ministral-3:3b", temperature=0.5, num_predict=150)

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
        image = data.get("image")
        content_parts = []

        if image:
            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            }
            content_parts.append(image_part)
        if text:
            text_part = {"type": "text", "text": text}
            content_parts.append(text_part)

        return [HumanMessage(content=content_parts)]

    return prompt_func(data)




def accord_extracting_agent(input_state, state: RecommendationWorkingState):
    data = {}
    if input_state["input_type"] == "text":
        user_input = input_state["mood_input"]
        data["text"] = user_input
    else:
        image_path = input_state["mood_input"]
        data["image"] = convert_to_base64(image_path)


    agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT
    )

    response = agent.invoke({"messages": form_user_content(data)})
    state["extracted_accords"] = json.loads(response["messages"][-1].content)

    return state


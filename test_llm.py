from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

repo_id = "deepseek-ai/DeepSeek-R1"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    provider="hyperbolic",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    # provider="auto",  # set your provider here hf.co/settings/inference-providers
    # provider="hyperbolic",
    # provider="nebius",
    # provider="together",
)

chat_model = ChatHuggingFace(llm=llm)

from langchain.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)
print(ai_msg.content)
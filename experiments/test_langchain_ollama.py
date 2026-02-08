'''
brew install ollama
brew services start ollama
ollama pull <name-of-model>

'''
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

def test_ollama():
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3", )

    chain = prompt | model

    print(chain.invoke({"question": "What is LangChain?"}))


def test_ollama_2():
    template="""
        You are a perfume sensory analyst.

        Given perfume data, extract 5 sensory mood of the fragrance.
        Base your answer ONLY on:
        - description
        - main accords
        - gender

        Do NOT invent notes.
        Do NOT mention brands.
        Do NOT repeat the description.

        Return ONLY valid JSON in the format:

        {{
        "moods": []
        }}

        Perfume data:
        {perfume_json}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3")

    chain = prompt | model

    perfume_content = {
        "description": "A warm and sensual fragrance for women.",
        "main_accords": ["spicy", "woody"],
        "gender": "women"
    }

    print(chain.invoke({"perfume_json": json.dumps(perfume_content)}))



if __name__=="__main__":
    test_ollama_2()
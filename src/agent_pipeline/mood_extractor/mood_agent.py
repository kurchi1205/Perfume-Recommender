"""
ReAct agent that reads perfume data, checks what's already processed,
and extracts 5 moods for each unprocessed perfume into a JSONL file.

Uses llama3.1:8b for agent reasoning, smollm2 for mood extraction.
"""

import json
import logging
from pathlib import Path

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from mood_model_agent import create_mood_extraction_chain, extract_moods

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Shared state across tools ---
_perfumes: list[dict] = []
_processed_urls: set[str] = set()
_unprocessed: list[dict] = []

INPUT_PATH = Path("../../../datasets/fragrantica_perfumes.json")
OUTPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")

# Initialize the mood extraction chain (smollm2) once
_mood_chain = create_mood_extraction_chain()


def _form_input_content(data_item: dict) -> str:
    """Format perfume data as text for the mood extraction LLM."""
    text_content = ""
    if data_item.get("gender"):
        text_content += f"Gender: {data_item['gender']}\n"
    if data_item.get("description"):
        text_content += f"Description: {data_item['description']}\n"
    notes = data_item.get("notes") or {}
    if notes.get("top"):
        text_content += f"Top notes: {', '.join(notes['top'])}\n"
    if notes.get("middle"):
        text_content += f"Middle notes: {', '.join(notes['middle'])}\n"
    if notes.get("base"):
        text_content += f"Base notes: {', '.join(notes['base'])}\n"
    if data_item.get("main_accords"):
        text_content += f"Main accords: {', '.join(data_item['main_accords'])}\n"
    return text_content


@tool
def read_input_perfumes(file_path: str) -> str:
    """Read perfume data from a JSON file. Returns a summary of loaded perfumes."""
    global _perfumes
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found at {file_path}"

    with open(path, "r", encoding="utf-8") as f:
        _perfumes = json.load(f)

    return f"Loaded {len(_perfumes)} perfumes from {file_path}."


@tool
def read_processed_urls(file_path: str) -> str:
    """Read already-processed perfume URLs from a JSONL output file. Returns how many are already done and how many remain."""
    global _processed_urls, _unprocessed
    path = Path(file_path)

    _processed_urls = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    url = record.get("url")
                    if url:
                        _processed_urls.add(url)
                except json.JSONDecodeError:
                    continue

    _unprocessed = [
        p for p in _perfumes
        if p.get("url") and p["url"] not in _processed_urls
    ]

    return (
        f"Already processed: {len(_processed_urls)} perfumes. "
        f"Remaining to process: {len(_unprocessed)} perfumes."
    )


@tool
def extract_and_save_moods(start_index: int, count: int) -> str:
    """Extract moods for a batch of unprocessed perfumes and save to the output JSONL file. start_index is the position in the unprocessed list, count is how many to process."""
    if not _unprocessed:
        return "No unprocessed perfumes available. Call read_input_perfumes and read_processed_urls first."

    end_index = min(start_index + count, len(_unprocessed))
    batch = _unprocessed[start_index:end_index]

    if not batch:
        return f"No perfumes in range [{start_index}, {end_index}). Total unprocessed: {len(_unprocessed)}."

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, perfume in enumerate(batch):
            name = perfume.get("name", "Unknown")
            idx = start_index + i
            logger.info(f"Processing {idx + 1}/{len(_unprocessed)}: {name}")

            content = _form_input_content(perfume)
            moods = extract_moods(_mood_chain, content)

            if moods:
                record = {**perfume, "moods": moods}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                succeeded += 1
            else:
                logger.warning(f"Failed to extract moods for: {name}")
                failed += 1

    return (
        f"Processed {len(batch)} perfumes (index {start_index} to {end_index - 1}). "
        f"Succeeded: {succeeded}, Failed: {failed}. "
        f"Remaining unprocessed: {len(_unprocessed) - end_index}."
    )


SYSTEM_PROMPT = (
    "You are a perfume mood extraction assistant. Your job is to:\n"
    "1. Read the input perfume data from the JSON file.\n"
    "2. Check which perfumes have already been processed in the output JSONL file.\n"
    "3. For all unprocessed perfumes, extract 5 moods and save them to the output JSONL.\n\n"
    "Use the tools provided. Process perfumes in batches of 50 using extract_and_save_moods.\n"
    "Keep calling extract_and_save_moods with incrementing start_index until all perfumes are done.\n\n"
    f"Input file: {INPUT_PATH}\n"
    f"Output file: {OUTPUT_PATH}"
)


def build_agent():
    """Build the ReAct agent with llama3.1:8b and the 3 tools."""
    model = ChatOllama(model="llama3.1:8b", temperature=0)

    agent = create_react_agent(
        model=model,
        tools=[read_input_perfumes, read_processed_urls, extract_and_save_moods],
        prompt=SYSTEM_PROMPT,
    )
    return agent


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Process all unprocessed perfumes."}]}
    )

    # Print final agent message
    final_message = result["messages"][-1]
    print(f"\nAgent: {final_message.content}")

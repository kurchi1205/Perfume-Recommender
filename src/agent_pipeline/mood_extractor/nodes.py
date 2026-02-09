import json
from mood_model_agent import (
    create_mood_model_agent,
    run_mood_model_agent,
    extract_moods_from_agent_result,
)
from states import PerfumeInputState, PerfumeWorkingState
from pathlib import Path
import logging

MOOD_AGENT = create_mood_model_agent()
OUTPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def load_existing_perfume_urls(output_path: Path) -> set[str]:
    """
    Reads existing JSONL file and returns a set of perfume URLs already processed.
    """
    existing = set()

    if not output_path.exists():
        return existing

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                url = record.get("url")
                if url:
                    existing.add(url)
            except json.JSONDecodeError:
                continue

    return existing


def form_input_content(data_item):
    text_content = ""
    if data_item["gender"]:
        text_content += f"Gender: {data_item['gender']}\n"
    if data_item["description"]:
        text_content += f"Description: {data_item['description']}\n"
    if data_item["notes"] and len(data_item["notes"]["top"]) > 0:
        text_content += f"Top notes: {', '.join(data_item['notes']['top'])}\n"
    if data_item["notes"] and len(data_item["notes"]["middle"]) > 0:
        text_content += f"Middle notes: {', '.join(data_item['notes']['middle'])}\n"
    if data_item["notes"] and len(data_item["notes"]["base"]) > 0:
        text_content += f"Base notes: {', '.join(data_item['notes']['base'])}\n"
    if data_item["main_accords"]:
        text_content += f"Main accords: {', '.join(data_item['main_accords'])}\n"
    return text_content


def initialize(state: PerfumeInputState) -> PerfumeWorkingState:
    total = len(state["perfumes"])
    existing_urls = load_existing_perfume_urls(OUTPUT_PATH)
    logger.info(f"🚀 Starting mood extraction for {total} perfumes")

    return {
        "perfumes": state["perfumes"],
        "current_index": 0,
        "current_perfume": None,
        "current_moods": None,
        "batch": [],
        "batch_size": 10,
        "total_perfumes": total,   # helpful for progress
        "existing_urls": existing_urls
    }



def next_perfume(state: PerfumeWorkingState) -> PerfumeWorkingState:
    total = state["total_perfumes"]
    existing_urls = state["existing_urls"]

    while state["current_index"] < total:
        idx = state["current_index"]
        perfume = state["perfumes"][idx]
        state["current_index"] += 1

        url = perfume.get("url")

        if url and url in existing_urls:
            logger.info(
                f"Skipping already processed perfume "
                f"{idx}/{total}: {perfume.get('name', 'Unknown')}"
            )
            continue

        state["current_perfume"] = perfume

        # Progress log
        if idx % 10 == 0 or idx == total - 1:
            logger.info(
                f"🔄 Processing perfume {idx + 1}/{total}: "
                f"{perfume.get('name', 'Unknown')}"
            )

        return state

    # If we reach here, we're done
    logger.info("All perfumes processed")
    state["current_perfume"] = None
    return state



def extract_moods(state: PerfumeWorkingState) -> PerfumeWorkingState:
    perfume = state["current_perfume"]

    if not perfume:
        state["current_moods"] = None
        return state

    try:
        content = form_input_content(perfume)
        agent_result = run_mood_model_agent(MOOD_AGENT, content)
        moods = extract_moods_from_agent_result(agent_result)

        if not moods:
            logger.warning(
                f"⚠️ No moods extracted for: {perfume.get('name', 'Unknown')}"
            )

        state["current_moods"] = moods

    except Exception as e:
        logger.error(
            f"Failed to extract moods for {perfume.get('name', 'Unknown')}: {e}"
        )
        state["current_moods"] = []

    return state



def assemble_output(state: PerfumeWorkingState):
    perfume = state["current_perfume"]
    moods = state["current_moods"]

    if not perfume or not moods:
        return state

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    state["batch"].append({
        **perfume,
        "moods": moods,
    })

    # Mark as processed immediately
    url = perfume.get("url")
    if url:
        state["existing_urls"].add(url)

    if len(state["batch"]) >= state["batch_size"]:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for item in state["batch"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            f"💾 Flushed batch of {state['batch_size']} perfumes "
            f"(up to index {state['current_index']})"
        )

        state["batch"].clear()

    return state


def save_jsonl(state: PerfumeWorkingState):
    if state["batch"]:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for item in state["batch"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"💾 Flushed final batch of {len(state['batch'])} perfumes")
        state["batch"].clear()

    logger.info("🎉 Mood extraction pipeline completed successfully")
    return {"perfumes": []}



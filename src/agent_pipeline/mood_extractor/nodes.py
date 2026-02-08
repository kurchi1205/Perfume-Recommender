import json
from mood_model_agent import (
    create_mood_model_agent,
    run_mood_model_agent,
    extract_moods_from_agent_result,
)
from states import PerfumeInputState, PerfumeWorkingState
from pathlib import Path

MOOD_AGENT = create_mood_model_agent()
OUTPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")

def form_input_content(data_item):
    text_content = ""
    if data_item["gender"]:
        text_content += f"Gender: {data_item['gender']}\n"
    if data_item["description"]:
        text_content += f"Description: {data_item['description']}\n"
    if data_item["notes"] and len(data_item["notes"]["top"]) > 0:
        text_content += f"Top notes: {', '.join(data_item['notes']['top'])}\n"
    if data_item["notes"] and len(data_item["notes"]["middle"]) > 0:
        text_content += f"Heart notes: {', '.join(data_item['notes']['heart'])}\n"
    if data_item["notes"] and len(data_item["notes"]["base"]) > 0:
        text_content += f"Base notes: {', '.join(data_item['notes']['base'])}\n"
    if data_item["main_accords"]:
        text_content += f"Main accords: {', '.join(data_item['main_accords'])}\n"
    return text_content


def initialize(state: PerfumeInputState) -> PerfumeWorkingState:
    return {
        "perfumes": state["perfumes"],
        "current_index": 0,
        "current_perfume": None,
        "current_moods": None,
        "batch": [],
        "batch_size": 10,
    }



def next_perfume(state: PerfumeWorkingState) -> PerfumeWorkingState:
    idx = state["current_index"]

    if idx >= len(state["perfumes"]):
        state["current_perfume"] = None
        return state

    state["current_perfume"] = state["perfumes"][idx]
    state["current_index"] += 1
    return state


def extract_moods(state: PerfumeWorkingState) -> PerfumeWorkingState:
    perfume = state["current_perfume"]

    if not perfume:
        state["current_moods"] = None
        return state

    content = form_input_content(perfume)

    # 2. Run mood agent
    agent_result = run_mood_model_agent(MOOD_AGENT, content)

    # 3. Extract moods safely
    moods = extract_moods_from_agent_result(agent_result)

    state["current_moods"] = moods
    return state



def assemble_output(state: PerfumeWorkingState):
    perfume = state["current_perfume"]
    moods = state["current_moods"]

    if not perfume or not moods:
        return state

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Add to batch
    state["batch"].append({
        **perfume,
        "moods": moods,
    })

    # Flush batch if full
    if len(state["batch"]) >= state["batch_size"]:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for item in state["batch"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        state["batch"].clear()


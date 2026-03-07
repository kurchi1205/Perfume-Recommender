from typing import TypedDict, List, Optional
import sys

sys.path.insert(0, "../../user_history")
from models import UserPreferenceSignal


# ---------------------------------------------------------------------------
# Input — what the user provides to start the graph
# ---------------------------------------------------------------------------

class RecommendationInputState(TypedDict):
    user_id: str
    input_type: str          # "text" | "image"
    mood_input: str          # free-text mood description OR path to image file


# ---------------------------------------------------------------------------
# Working state — passed between all nodes
# ---------------------------------------------------------------------------

class RecommendedPerfume(TypedDict):
    perfume_id: str          # Milvus id
    name: str
    brand: str
    description: str
    url: str
    main_accords: List[str]
    gender: str
    search_score: float
    rerank_score: float
    


class RecommendationWorkingState(TypedDict):
    # --- from input ---
    user_id: str
    input_type: str
    mood_input: str

    # --- after mood extraction ---
    extracted_accords: List[str]   # accords pulled from mood_input

    # --- after user history load ---
    user_signal: UserPreferenceSignal

    # --- after Milvus search ---
    candidates: List[dict]         # top-20 raw results from Milvus

    # --- after reranker ---
    reranked: List[RecommendedPerfume]   # top-5

    # --- retry control ---
    retry_count: int
    user_intent_summary: str


# ---------------------------------------------------------------------------
# Output — what the graph returns to the caller
# ---------------------------------------------------------------------------

class RecommendationOutputState(TypedDict):
    user_id: str
    recommendations: List[RecommendedPerfume]

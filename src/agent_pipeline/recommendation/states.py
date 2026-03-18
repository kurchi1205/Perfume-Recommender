from typing import List, TypedDict

from schemas import RecommendedPerfume


# ---------------------------------------------------------------------------
# Input — what the user provides to start the graph
# ---------------------------------------------------------------------------

class RecommendationInputState(TypedDict):
    input_type: str   # "text" | "image"
    mood_input: str   # free-text mood description OR image URL
    user_id:    str   # defaults to "default" for anonymous users


# ---------------------------------------------------------------------------
# Working state — passed between all nodes
# ---------------------------------------------------------------------------

class RecommendationWorkingState(TypedDict):
    # --- from input ---
    input_type: str
    mood_input: str
    user_id:    str

    # --- [1] Taste + Profile Agent ---
    extracted_moods:   List[str]
    extracted_accords: List[str]
    intent_summary:    str

    # --- [2] Search Agent ---
    query_text: str        # kept for Memory Update
    reranked:   List[dict] # top-5 scored candidates

    # --- [3] Explanation Generator ---
    recommendations: List[RecommendedPerfume]


# ---------------------------------------------------------------------------
# Output — what the graph returns to the caller
# ---------------------------------------------------------------------------

class RecommendationOutputState(TypedDict):
    recommendations: List[RecommendedPerfume]

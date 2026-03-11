"""
Shared Pydantic schemas for the recommendation pipeline.
Used at node boundaries to validate inputs and outputs.
"""
import json
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator, AnyHttpUrl


# ── Enums ─────────────────────────────────────────────────────────────────────

class InputType(str, Enum):
    text  = "text"
    image = "image"


# ── LLM output validators ─────────────────────────────────────────────────────

class ExtractedList(BaseModel):
    """Validates a list of strings returned by the LLM (moods or accords)."""
    items: List[str]

    @field_validator("items", mode="before")
    @classmethod
    def parse_and_clean(cls, v):
        # If the LLM returned a JSON string instead of a list, parse it
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                # Try extracting a JSON array from somewhere inside the string
                match = re.search(r"\[.*?\]", v, re.DOTALL)
                if match:
                    v = json.loads(match.group())
                else:
                    raise ValueError(f"Cannot parse list from LLM output: {v!r}")

        if not isinstance(v, list):
            raise ValueError(f"Expected a list, got {type(v).__name__}")

        # Clean each item
        cleaned = [str(item).strip().lower() for item in v if str(item).strip()]
        if not cleaned:
            raise ValueError("Extracted list is empty after cleaning")
        return cleaned

    @field_validator("items")
    @classmethod
    def check_length(cls, v):
        if len(v) > 15:
            return v[:15]   # silently cap — LLM sometimes over-generates
        return v


# ── Milvus search result ───────────────────────────────────────────────────────

class CandidatePerfume(BaseModel):
    """A single result returned by search_milvus before reranking."""
    perfume_id: str
    name: str
    brand: str
    description: str
    url: str
    gender: str
    main_accords: List[str]
    search_score: float
    rerank_score: float = 0.0

    @field_validator("perfume_id", "name", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v).strip()

    @field_validator("name", "perfume_id")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("Field must not be empty")
        return v

    @field_validator("main_accords", mode="before")
    @classmethod
    def parse_accords(cls, v):
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return [str(a).strip() for a in v if str(a).strip()]

    @field_validator("search_score")
    @classmethod
    def valid_score(cls, v):
        # COSINE similarity in Milvus returns [-1.0, 1.0]
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"search_score {v} out of range [-1, 1]")
        return v


# ── Evaluator output ──────────────────────────────────────────────────────────

class ScoredPerfume(CandidatePerfume):
    """Candidate after LLM evaluation."""
    llm_score:   float = 0.0
    final_score: float = 0.0

    @field_validator("llm_score")
    @classmethod
    def valid_llm_score(cls, v):
        if not (0.0 <= v <= 10.0):
            raise ValueError(f"llm_score {v} out of range [0, 10]")
        return v


# ── Final recommendation ──────────────────────────────────────────────────────

class RecommendedPerfume(BaseModel):
    """Final validated output perfume shown to the user."""
    perfume_id:  str
    name:        str
    brand:       str
    description: str
    url:         str
    image_url:   Optional[str] = None
    main_accords: List[str]
    gender:      str
    llm_score:   Optional[float] = None
    final_score: Optional[float] = None

    @field_validator("main_accords", mode="before")
    @classmethod
    def parse_accords(cls, v):
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return [str(a).strip() for a in v if str(a).strip()]

    @field_validator("name", "perfume_id")
    @classmethod
    def not_empty(cls, v):
        return v.strip() if v else v

    class Config:
        extra = "allow"   # pass-through extra fields (rerank_score etc.)

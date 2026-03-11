"""
API-layer Pydantic schemas for request validation and SSE response structure.
"""
from enum import Enum
from typing import List, Optional, Literal

from pydantic import BaseModel, field_validator, model_validator


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES     = 10 * 1024 * 1024   # 10 MB
MAX_TEXT_LENGTH     = 2000


class InputType(str, Enum):
    text  = "text"
    image = "image"


class RecommendRequest(BaseModel):
    input_type: InputType
    user_id:    str = "anonymous"
    text:       str = ""

    @field_validator("text")
    @classmethod
    def cap_text_length(cls, v):
        return v[:MAX_TEXT_LENGTH]

    @model_validator(mode="after")
    def text_required_for_text_mode(self):
        if self.input_type == InputType.text and not self.text.strip():
            raise ValueError("text must not be empty when input_type is 'text'")
        return self


# ── SSE event payloads ────────────────────────────────────────────────────────

class MoodsEvent(BaseModel):
    type:   Literal["moods"] = "moods"
    moods:  List[str]

    @field_validator("moods")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("moods list must not be empty")
        return v


class AccordsEvent(BaseModel):
    type:    Literal["accords"] = "accords"
    accords: List[str]

    @field_validator("accords")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("accords list must not be empty")
        return v


class ResultEvent(BaseModel):
    type:            Literal["result"] = "result"
    recommendations: List[dict]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type:    Literal["error"] = "error"
    message: str

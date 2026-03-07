from dataclasses import dataclass, field
from typing import List


@dataclass
class UserPreferenceSignal:
    user_id: str
    accords: List[str] = field(default_factory=list)          # e.g. ["oud", "vanilla", "woody"]
    preferred_gender: str | None = None                        # "For Men" | "For Women" | "Unisex"
    past_perfume_ids: List[str] = field(default_factory=list)  # Milvus ids already recommended
    session_count: int = 0
    summary: str | None = None   # LLM-generated summary of user's overall preferences

    @property
    def cold_start(self) -> bool:
        return self.session_count < 3

from typing import TypedDict, List, Dict, Optional


class PerfumeInputState(TypedDict):
    perfumes: List[Dict]  # raw perfume JSON objects

class PerfumeWorkingState(TypedDict):
    perfumes: List[dict]
    current_index: int
    current_perfume: Optional[dict]
    current_moods: Optional[List[str]]

    batch: List[dict]       
    batch_size: int   
    total_perfumes: int      


class PerfumeMoodOutput(TypedDict):
    url: str
    moods: List[str]


class PerfumeOutputState(TypedDict):
    perfumes: List[PerfumeMoodOutput]



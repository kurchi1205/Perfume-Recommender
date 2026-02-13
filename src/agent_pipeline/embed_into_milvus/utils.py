
from typing import List
import uuid
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def init_bge_embedder(device: str = "cpu"):
    model_name = "BAAI/bge-m3"
    model_kwargs = {
        "device": device,
        "trust_remote_code": True
    }
    encode_kwargs = {
        "normalize_embeddings": True
    }

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def embed_text_bge(
    embedder: HuggingFaceBgeEmbeddings,
    mood_list: List[str],
) -> List[float]:
    """
    Embeds a list of texts using BGE.
    Returns List[List[float]] suitable for Milvus FLOAT_VECTOR.
    """
    processed_moods = []
    for item in mood_list:
        if isinstance(item, dict):
            # Extract all values from the dictionary
            processed_moods.extend(item.values())
        else:
            # Keep string items as-is
            processed_moods.append(item)
    
    # Join all moods into a single string
    moods = ", ".join(str(mood) for mood in processed_moods)
    
    # LangChain BGE supports embed_documents for batching
    embeddings = embedder.embed_query(moods)
    return embeddings


def build_record(embedder, item: dict) -> dict:
    notes = item.get("notes", {})
    
    return {
        "id": str(uuid.uuid4()),

        "name": item.get("name", ""),
        "description": item.get("description", ""),
        "url": item.get("url", ""),
        "brand": item.get("brand") or "",
        "gender": item.get("gender", ""),

        "top_notes": ", ".join(notes.get("top", [])),
        "middle_notes": ", ".join(notes.get("middle", [])),
        "base_notes": ", ".join(notes.get("base", [])),
        "main_accords": ", ".join(item.get("main_accords", [])),

        "moods_embedding": embed_text_bge(embedder, item.get("moods", []))
    }
# Perfume Recommender

An AI-powered perfume recommendation system that uses local LLMs to extract mood attributes from perfume data and stores them in a vector database for semantic search and recommendations.

## Architecture

```
Raw Data (Fragrantica CSV / Parfumo scraping)
        ↓
  Data Extraction & Normalization
        ↓
  fragrantica_perfumes.json (20k+ perfumes)
        ↓
  Mood Extraction (LangGraph + Ollama)
        ↓
  perfumes_with_moods.jsonl
        ↓
  Vector Embeddings → Milvus DB
        ↓
  Semantic Search & Recommendations
```

## Project Structure

```
├── datasets/
│   ├── fragrantica/              # Raw CSV data from Kaggle
│   ├── fragrantica_perfumes.json # Processed perfume dataset (~20k entries)
│   └── perfumes_with_moods.jsonl # Perfumes with extracted moods
├── src/
│   ├── agent_pipeline/
│   │   ├── mood_extractor/       # LLM-based mood extraction
│   │   │   ├── mood_agent.py           # ReAct agent (LLaMA 3.1)
│   │   │   ├── mood_model_agent.py     # Mood extraction chain (SmollM2)
│   │   │   └── langgraph_agent/        # Stateful LangGraph agent (preferred)
│   │   ├── embed_into_milvus/    # Vector DB embedding pipeline
│   │   └── raw_meta_scraper/     # Web scraping utilities
│   ├── milvus_setup/             # Milvus vector DB configuration
│   └── utils/                    # Data processing utilities
│       ├── extract_json_from_csv.py  # CSV → JSON conversion
│       ├── normalize_data.py         # Data normalization
│       └── parfumo_scraping.py       # Parfumo.com scraper
└── experiments/                  # Testing and prototyping scripts
```

## Tech Stack

- **Python** — Core language
- **LangChain + LangGraph** — Agent orchestration and stateful workflows
- **Ollama** — Local LLM inference (LLaMA 3.1, SmollM2)
- **Milvus** — Vector database for semantic search
- **Pydantic** — Structured output validation
- **BeautifulSoup** — Web scraping

## Setup

### Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Docker (for Milvus)

### Installation

```bash
pip install -r requirements.txt
```

### Pull the required Ollama models

```bash
ollama pull llama3.1:8b
ollama pull smollm2:latest
```

### Start Milvus

```bash
cd src/milvus_setup
bash standalone_embed.sh start
python create_db.py
```

## Usage

### 1. Prepare the dataset

Download the Fragrantica dataset from Kaggle and convert it to JSON:

```bash
python src/utils/extract_json_from_csv.py
python src/utils/normalize_data.py
```

### 2. Extract moods

Run the LangGraph-based mood extraction agent (recommended):

```bash
python src/agent_pipeline/mood_extractor/mood_agent.py
```

This processes perfumes, extracting 5 mood attributes per perfume (e.g., Romantic, Mysterious, Confident, Sultry, Ethereal). Progress is saved incrementally to `perfumes_with_moods.jsonl`, so the process can be resumed if interrupted.

### 3. Embed into Milvus

```bash
ongoing
```

## Mood Extraction

Each perfume is analyzed using its description, notes (top/middle/base), accords, and gender to produce exactly 5 mood labels from a curated vocabulary:

> Romantic, Mysterious, Confident, Nostalgic, Serene, Energetic, Sultry, Sophisticated, Rebellious, Cozy, Ethereal, Playful, Melancholic, Bold, Warm, ...

Example output:

```json
{
  "name": "Chanel No. 5",
  "brand": "Chanel",
  "gender": "women",
  "notes": { "top": ["Aldehydes", "Ylang-Ylang"], "middle": ["Rose", "Jasmine"], "base": ["Sandalwood", "Vanilla"] },
  "main_accords": ["powdery", "floral", "aldehydic"],
  "moods": ["Sophisticated", "Romantic", "Elegant", "Nostalgic", "Warm"]
}
```

## Data Sources

- [Fragrantica dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) (Kaggle) — ~20,000 perfumes
- [Parfumo.com](https://www.parfumo.com/) — Supplementary scraping

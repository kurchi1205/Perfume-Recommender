# Perfume Recommender — System Planner

---

## High-Level Flow

```
User Input (text mood / image)
        │
        ▼
[1] Mood Extraction Agent
        │  structured terms: mood tags, notes, accords, occasion, season
        ▼
[2] User History Service
        │  fetch past interactions, liked/disliked perfumes,
        │  preferred accords/notes/brands/occasions for user_id
        ▼
[2b] User Understanding Agent
        │  LLM synthesizes current input + history into a coherent
        │  "user intent profile" (what they REALLY want right now)
        ▼
[3] Query Builder
        │  encodes user intent profile → blended query vector
        │  + Milvus filter expression for exclusions
        ▼
[4] Milvus Vector Search  (perfume_db / perfume_collection)
        │  top-K candidates via COSINE on moods_embedding
        ▼
[5] Reranker
        │  cross-encoder scores candidates against full user intent profile
        ▼
[5b] Evaluator
        │  checks alignment between user intent and reranked results
        │  if alignment is low → loop back to [3] with adjusted query
        ▼
[5c] Summarizer
        │  generates a short natural-language explanation of why each
        │  top result matches the user's intent (shown to user)
        ▼
[6] Result Enrichment
        │  fetch perfume images + descriptions from URL
        ▼
[7] Response to User
        │  formatted recommendations with images + summaries
        ▼
[8] History Writer  ──────────────────────────────────────────────────────┐
        stores session, results, and feedback in user_db                  │
        (feeds back into [2] for the next session)  ──────────────────────┘
```

---

## Databases

### 1. Perfume DB (existing — Milvus)
- **Host:** localhost:19530
- **DB:** `perfume_db`
- **Collection:** `perfume_collection`
- **Schema:** id, name, description, url, brand, gender, top_notes,
  middle_notes, base_notes, main_accords, moods_embedding (FLOAT_VECTOR 1024)
- **Index:** IVF_FLAT, COSINE on `moods_embedding`
- **Populated by:** `embed_into_milvus/agent.py` from `perfumes_with_moods.jsonl`

### 2. User DB (new — SQLite / PostgreSQL)
Stores user preferences separately from Milvus — kept simple and relational.

#### Table

```sql
-- User preference profile: upserted after each session / feedback event.
-- Captures ALL preference dimensions, not just mood.
CREATE TABLE user_preferences (
    user_id           TEXT PRIMARY KEY,    -- UUID, generated on first visit
    liked_brands      TEXT,               -- JSON array  e.g. ["Chanel","Tom Ford"]
    disliked_brands   TEXT,               -- JSON array
    liked_accords     TEXT,               -- JSON array  e.g. ["oud","vanilla","woody"]
    disliked_accords  TEXT,               -- JSON array
    liked_notes       TEXT,               -- JSON array  e.g. ["bergamot","amber","musk"]
    disliked_notes    TEXT,               -- JSON array
    liked_moods       TEXT,               -- JSON array  e.g. ["warm","cozy","sensual"]
    disliked_moods    TEXT,               -- JSON array
    preferred_gender  TEXT,               -- "For Men" | "For Women" | "Unisex" | null
    preferred_occasions TEXT,             -- JSON array  e.g. ["evening","office","date"]
    preferred_seasons TEXT,               -- JSON array  e.g. ["winter","spring"]
    session_count     INTEGER DEFAULT 0,  -- used to detect cold-start (< 3)
    updated_at        TIMESTAMP DEFAULT NOW()
);
```

**Why all dimensions beyond mood?**
A user might love woody/oud regardless of their current mood, or always dislike
aquatic notes. Mood is one axis; this table captures the full picture.
`session_count < 3` → cold-start: skip history blend, use pure current input.

---

## Component Details

### [1] Mood Extraction Agent  (`mood_extractor/`)
- **Text input:** LLM prompt → structured terms (existing pipeline)
- **Image input:** vision model (LLaVA via Ollama) → scene description → terms
- **Output:**
  ```python
  {
    "moods":    ["warm", "cozy", "intimate"],   # may be empty
    "notes":    ["bergamot", "vanilla"],         # if user mentions specific notes
    "accords":  ["woody", "oriental"],           # if mentioned
    "occasion": "evening",                       # optional
    "season":   "winter"                         # optional
  }
  ```
  All fields are optional — the agent extracts whatever is present.

### [2] User History Service  (`user_history/`)
Thin wrapper around the `user_preferences` table.

```
user_history/
    __init__.py
    db.py          # init_db(), get_or_create_user()
    profile.py     # load_preferences(), upsert_preferences()
    models.py      # UserPreferenceSignal dataclass
```

**`load_preferences(user_id) → UserPreferenceSignal`**
- Single SELECT on `user_preferences` by `user_id`
- Deserializes JSON arrays
- Sets `cold_start = True` if `session_count < 3`
- Returns:
  ```python
  @dataclass
  class UserPreferenceSignal:
      liked_accords:       List[str]
      disliked_accords:    List[str]
      liked_notes:         List[str]
      disliked_notes:      List[str]
      liked_moods:         List[str]
      disliked_moods:      List[str]
      liked_brands:        List[str]
      preferred_gender:    str | None
      preferred_occasions: List[str]
      preferred_seasons:   List[str]
      cold_start:          bool
  ```

**`upsert_preferences(user_id, feedback)`**
- Called after each session / explicit feedback
- Merges new liked/disliked items into the existing JSON arrays
- Increments `session_count`, updates `updated_at`

### [2b] User Understanding Agent  (`recommendation/understanding_agent.py`)
An LLM node that synthesizes the extracted terms + user preference signal into a
single coherent **intent profile** used downstream by the Query Builder and Reranker.

**Input:**
- `extracted_terms` from Mood Extraction Agent
- `user_signal` from User History Service

**Prompt (system):**
```
You are a perfume expert helping to understand what a user truly wants.
Given their current request and their preference history, produce a concise
"intent profile" that combines both into a single description of the ideal
perfume for this person right now.
```

**Output:**
```python
@dataclass
class UserIntentProfile:
    intent_text: str           # e.g. "warm, sensual evening perfume with oud and vanilla,
                               #        avoiding aquatic notes, preferably unisex"
    query_terms: List[str]     # tokenized terms for embedding
    hard_exclusions: List[str] # notes/accords to filter out (from disliked history)
    soft_preferences: List[str]# nice-to-have from history (for reranker context)
    cold_start: bool
```

**Why this agent is needed:**
The raw mood tags and history signal are separate signals. The Understanding Agent
resolves conflicts (e.g. user says "fresh" but history shows they always like
oriental fragrances), merges them intelligently, and produces a single clean input
for the rest of the pipeline.

### [3] Query Builder  (`recommendation/query_builder.py`)

```python
def build_query(
    intent: UserIntentProfile,
    embedder,
    user_signal: UserPreferenceSignal,
    alpha: float = 0.7        # current input weight vs. history
) -> tuple[List[float], str]:
    """Returns (query_vector, milvus_filter_expr)."""

    current_vec = embed_text_bge(embedder, intent.query_terms)

    if not intent.cold_start and user_signal.liked_accords:
        history_terms = (
            user_signal.liked_accords +
            user_signal.liked_notes +
            user_signal.liked_moods
        )
        history_vec = embed_text_bge(embedder, history_terms)
        query_vec = normalize(alpha * current_vec + (1 - alpha) * history_vec)
    else:
        query_vec = current_vec   # cold-start: pure current input

    filter_expr = build_exclusion_filter(intent.hard_exclusions)
    return query_vec, filter_expr
```

Cold-start uses `alpha=1.0` (pure current input). History blend is applied once
at least 3 sessions with explicit feedback exist.

### [4] Milvus Search  (`recommendation/search.py`)

```python
def search_perfumes(
    client: MilvusClient,
    query_vec: List[float],
    top_k: int = 20,
    filter_expr: str = "",
    gender_filter: str = ""
) -> List[dict]:
    results = client.search(
        collection_name=COLLECTIONS_NAME,
        data=[query_vec],
        limit=top_k,
        output_fields=["id", "name", "brand", "description", "url",
                       "main_accords", "top_notes", "base_notes", "gender"],
        filter=filter_expr,
    )
    return results[0]
```

### [5] Reranker  (`recommendation/reranker.py`)
Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
The rerank query is built from the full intent profile — not just mood:

```python
def rerank(
    intent: UserIntentProfile,
    candidates: List[dict],
    top_n: int = 5
) -> List[dict]:
    # Use intent_text as the rich query for cross-encoding
    pairs = [(intent.intent_text, c["description"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]
```

### [5b] Evaluator  (`recommendation/evaluator.py`)
An LLM node that checks whether the top-5 reranked results genuinely align
with the user's intent profile.

**Input:** `intent: UserIntentProfile`, `top_results: List[dict]`

**Output:**
```python
@dataclass
class EvaluationResult:
    aligned: bool              # True if results are good enough to show
    alignment_score: float     # 0.0–1.0
    issues: List[str]          # e.g. ["top result is aquatic, user dislikes aquatic"]
    retry_hint: str | None     # if not aligned, adjusted query hint for [3]
```

If `aligned=False`, the graph loops back to [3] Query Builder with `retry_hint`
incorporated into the query terms. Maximum 2 retry loops to avoid infinite cycles.

### [5c] Summarizer  (`recommendation/summarizer.py`)
An LLM node that generates a short, user-facing explanation for each top result
explaining why it matches the user's intent.

**Input:** `intent: UserIntentProfile`, `result: dict` (one perfume at a time)

**Output per perfume:** 1–2 sentence natural language summary.

Example:
> "Noir de Noir by Tom Ford is a rich, dark rose built on oud and patchouli —
>  perfect for the warm, intimate evening feel you're looking for."

These summaries are stored in `recommendations.summary` in the user DB and shown
alongside images in the final response.

### [6] Result Enrichment  (`recommendation/enricher.py`)
- For each top result, fetch/cache the perfume image URL from `perfume["url"]`
- Reuse `utils/parfumo_scraping.py`
- Return: `{name, brand, image_url, description, main_accords, top_notes, base_notes, score, summary}`

### [7] Main Recommendation Agent  (`recommendation/agent.py`)
LangGraph pipeline wiring all components. Includes a conditional retry loop.

```
START
  │
  ▼
extract_input             (MoodExtractionAgent → structured terms)
  │
  ▼
load_user_history         (UserHistoryService → UserPreferenceSignal)
  │
  ▼
understand_user           (UserUnderstandingAgent → UserIntentProfile)
  │
  ▼
build_query               (QueryBuilder → query_vec + filter_expr)
  │
  ▼
search_milvus             (top-20 candidates)
  │
  ▼
rerank                    (top-5 via cross-encoder on intent_text)
  │
  ▼
evaluate_results          (Evaluator → EvaluationResult)
  │
  ├─ aligned=False, retries<2 ──→ adjust query → search_milvus (retry loop)
  │
  ▼ aligned=True
summarize_results         (Summarizer → per-result explanation)
  │
  ▼
enrich_results            (images + descriptions)
  │
  ▼
save_session              (write to user_db)
  │
  ▼
END → return formatted recommendations + summaries to user
```

**State schema for the LangGraph:**
```python
class RecommendationState(TypedDict):
    user_id:        str
    raw_input:      str
    input_type:     str                    # "text" | "image"
    extracted_terms: dict
    user_signal:    UserPreferenceSignal
    intent:         UserIntentProfile
    query_vec:      List[float]
    filter_expr:    str
    candidates:     List[dict]             # top-20 from Milvus
    reranked:       List[dict]             # top-5 after reranking
    evaluation:     EvaluationResult
    retry_count:    int
    final_results:  List[dict]             # enriched + summarized
    session_id:     str
```

### [8] History Writer
Called at the end of the recommendation agent graph:
```python
upsert_preferences(user_id, feedback={
    "liked_accords": [...],
    "liked_notes": [...],
    "liked_moods": [...],
    # etc. — derived from what user reacted to this session
})
```
No separate sessions/recommendations tables needed — preferences are
merged directly into the single `user_preferences` row.

---

## Directory Layout (target)

```
src/
├── agent_pipeline/
│   ├── embed_into_milvus/              # existing — DB setup & data ingestion
│   │   ├── agent.py
│   │   ├── update_db_tools.py
│   │   └── utils.py
│   ├── mood_extractor/                 # existing — text/image → structured terms
│   │   └── langgraph_agent/
│   └── recommendation/                 # NEW — core recommendation pipeline
│       ├── agent.py                    # LangGraph orchestrator + state
│       ├── understanding_agent.py      # [2b] UserUnderstandingAgent
│       ├── query_builder.py            # [3]  blended query vec + filter
│       ├── search.py                   # [4]  Milvus search wrapper
│       ├── reranker.py                 # [5]  cross-encoder rerank
│       ├── evaluator.py                # [5b] alignment evaluator
│       ├── summarizer.py               # [5c] per-result LLM summarizer
│       └── enricher.py                 # [6]  image + description fetcher
├── user_history/                       # NEW — user preferences DB
│   ├── db.py                           # schema creation, get_or_create_user
│   ├── profile.py                      # load_preferences(), upsert_preferences()
│   └── models.py                       # UserPreferenceSignal dataclass
├── milvus_setup/
│   └── create_db.py
└── utils/
    ├── normalize_data.py
    ├── parfumo_scraping.py             # reused by enricher.py
    └── extract_json_from_csv.py
```

---

## Implementation Order

1. **`user_history/models.py`** — `UserPreferenceSignal` dataclass
2. **`user_history/db.py`** — create `user_preferences` table, `get_or_create_user()`
3. **`user_history/profile.py`** — `load_preferences()`, `upsert_preferences()`
4. **`recommendation/search.py`** — verify Milvus search end-to-end
6. **`recommendation/understanding_agent.py`** — LLM intent synthesis
7. **`recommendation/query_builder.py`** — blended embedding query
8. **`recommendation/reranker.py`** — cross-encoder rerank with intent_text
9. **`recommendation/evaluator.py`** — alignment check + retry logic
10. **`recommendation/summarizer.py`** — per-result LLM explanation
11. **`recommendation/enricher.py`** — image scraping (reuse parfumo_scraping.py)
12. **`recommendation/agent.py`** — LangGraph graph wiring all nodes + retry loop
13. **Feedback collection** — CLI prompt or endpoint; triggers `profile.rebuild()`

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| User history store | SQLite (dev) / Postgres (prod) | Relational queries on history; Milvus not suited for this |
| What history tracks | accords + notes + brands + moods + occasions + seasons | Mood is just one axis; preferences are richer |
| User Understanding Agent | LLM synthesis of input + history | Resolves conflicts between current input and past preferences |
| Embedding model | BAAI/bge-m3 (existing) | Must match the model used at ingestion time |
| Reranker query | Full `intent_text` (not just moods) | Leverages the full preference profile for scoring |
| Evaluator retry | Max 2 loops | Prevents infinite retries while allowing correction |
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast, local, no API key needed |
| History blend weight α | 0.7 current / 0.3 history | Current input dominates; history nudges results |
| Cold-start | α=1.0, skip history blend | No feedback yet → pure current input search |
| Session persistence | user_id passed in (CLI env var or API header) | Ties sessions to one user across conversations |

---

## Free Deployment Plan

### The Problem: 3 Local Services That Can't Run on Free Cloud Tiers

| Component | Current | Problem |
|---|---|---|
| **LLM** | Ollama + Mistral (local) | Can't run a 7B model on free cloud servers |
| **Vector DB** | Milvus via Docker | Needs Docker + high memory |
| **Embeddings** | BGE-M3 (local HF model) | ~570MB model, memory-heavy |

---

### Required Code Changes

#### 1. Replace Ollama with Groq (free LLM API)

Groq has a free tier with Mistral/Llama support, is fast, and is drop-in compatible via LangChain.

- `nodes/mood_extractor.py:23` — change `ChatOllama(model="ministral-3:3b")` → `ChatGroq(model="mistral-saba-24b")`
- `nodes/result_enricher.py:13` — same swap
- `nodes/accord_extractor.py` — same swap
- Add `langchain-groq` to `requirements.txt`
- Set `GROQ_API_KEY` as an environment variable

#### 2. Replace Local Milvus with Zilliz Cloud (free tier)

Zilliz is the managed version of Milvus — uses the exact same `pymilvus` SDK, so it's a config-only change.

- `nodes/search_mcp_server.py:27-28` — change `MILVUS_URI` and `MILVUS_TOKEN` to Zilliz Cloud endpoint + API key
- Re-run the embedding pipeline once (`src/agent_pipeline/embed_into_milvus/`) pointed at the cloud URI to populate the collection

#### 3. BGE-M3 Embeddings — no change needed

HuggingFace Spaces free CPU tier provides 16GB RAM, which is enough for BGE-M3. No code change required.

---

### Hosting Platform Choices (All Free)

| Part | Platform | Free Tier Details |
|---|---|---|
| **Frontend** (React/Vite) | Vercel or Netlify | Unlimited static sites |
| **Backend** (FastAPI + BGE-M3) | HuggingFace Spaces (Docker SDK) | 2 vCPU, 16GB RAM |
| **Vector DB** | Zilliz Cloud | 1 free cluster unit |
| **LLM** | Groq API | Free tier (rate-limited) |
| **User history** (SQLite) | Lives on HF Space disk | Persists within the Space |

---

### Step-by-Step Deployment Order

1. Sign up for [Groq](https://console.groq.com) and [Zilliz Cloud](https://zilliz.com/cloud)
2. Make code changes: swap Ollama → Groq in `mood_extractor.py`, `accord_extractor.py`, `result_enricher.py`
3. Update `MILVUS_URI` and `MILVUS_TOKEN` in `search_mcp_server.py` to point at Zilliz Cloud
4. Re-run the embedding pipeline to populate the Zilliz collection
5. Write a `Dockerfile` for the backend and deploy to HuggingFace Spaces (Docker Space)
6. Set secrets in HF Space: `GROQ_API_KEY`, `MILVUS_URI`, `MILVUS_TOKEN`
7. Deploy frontend to Vercel — connect GitHub repo and set `VITE_API_URL` to the HF Space URL

---

### Caveats

- **HF Spaces free tier spins down after ~15 min of inactivity** — first request after idle takes ~30s cold start. If always-on is needed, Railway's $5/month free credit is a better option.
- **Groq free tier is rate-limited** — fine for demos, not for high traffic.
- **Zilliz free tier** has limited storage (~1GB) — sufficient for the current perfume dataset.

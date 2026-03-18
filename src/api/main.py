import base64
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pipeline/recommendation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import build_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from events import (
    AccordsEvent, DoneEvent, ErrorEvent, FeedbackRequest,
    IntentEvent, MoodsEvent, ResultEvent,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

DB_PATH = Path(__file__).resolve().parents[2] / "user_history.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _upload_to_imgbb(image_bytes: bytes) -> str:
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="IMGBB_API_KEY not set in environment")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.imgbb.com/1/upload",
            params={"key": api_key},
            data={"image": b64},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["url"]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/recommend")
async def recommend(
    input_type: str       = Form(...),
    text:       str       = Form(default=""),
    user_id:    str       = Form(default="default"),
    image:      UploadFile = File(default=None),
):
    if input_type == "text" and not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty when input_type is 'text'")

    if input_type == "image" and image:
        image_bytes = await image.read()
        image_url   = await _upload_to_imgbb(image_bytes)
        graph_input = {"input_type": "image", "mood_input": image_url, "user_id": user_id}
    else:
        graph_input = {"input_type": "text", "mood_input": text[:2000], "user_id": user_id}

    async def generate():
        try:
            async for chunk in graph.astream(graph_input):
                if "taste_profile_agent" in chunk:
                    node = chunk["taste_profile_agent"]
                    if node.get("extracted_moods"):
                        yield _sse(MoodsEvent(moods=node["extracted_moods"]).model_dump())
                    if node.get("extracted_accords"):
                        yield _sse(AccordsEvent(accords=node["extracted_accords"]).model_dump())
                    if node.get("intent_summary"):
                        yield _sse(IntentEvent(intent_summary=node["intent_summary"]).model_dump())

                if "explanation_generator" in chunk:
                    recs = chunk["explanation_generator"].get("recommendations", [])
                    yield _sse(ResultEvent(recommendations=recs).model_dump())

            yield _sse(DoneEvent().model_dump())
        except Exception as e:
            yield _sse(ErrorEvent(message=str(e)).model_dump())

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (user_id, perfume_id, reaction, accords)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, perfume_id) DO UPDATE SET
                    reaction  = excluded.reaction,
                    accords   = excluded.accords,
                    timestamp = datetime('now')
                """,
                (req.user_id, req.perfume_id, req.reaction, json.dumps(req.accords)),
            )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["../../src/"])

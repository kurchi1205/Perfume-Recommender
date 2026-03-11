import base64
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Make the recommendation graph importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pipeline/recommendation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from graph import build_graph
from nodes.search import close_mcp_client

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/api/
from events import AccordsEvent, DoneEvent, ErrorEvent, MoodsEvent, ResultEvent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield                        # startup — MCP client is lazy-initialized on first request
    await close_mcp_client()     # shutdown — terminate the MCP subprocess cleanly


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


async def _upload_to_imgbb(image_bytes: bytes) -> str:
    """Upload image bytes to imgbb and return the public URL."""
    api_key = os.getenv("IMGBB_API_KEY")
    print(f"IMGBB_API_KEY: {api_key}")
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


@app.post("/recommend")
async def recommend(
    input_type: str = Form(...),
    text: str = Form(default=""),
    image: UploadFile = File(default=None),
):
    # Validate text input
    if input_type == "text" and not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty when input_type is 'text'")

    if input_type == "image" and image:
        image_bytes = await image.read()
        image_url = await _upload_to_imgbb(image_bytes)
        graph_input = {"input_type": "image", "mood_input": image_url}
    else:
        graph_input = {"input_type": "text", "mood_input": text[:2000]}

    async def generate():
        try:
            async for chunk in graph.astream(graph_input):
                # extract_mood node finished — stream moods immediately
                if "extract_mood" in chunk:
                    moods = chunk["extract_mood"].get("extracted_moods", [])
                    if moods:
                        yield _sse(MoodsEvent(moods=moods).model_dump())

                # extract_accord node finished
                if "extract_accord" in chunk:
                    accords = chunk["extract_accord"].get("extracted_accords", [])
                    if accords:
                        yield _sse(AccordsEvent(accords=accords).model_dump())

                # evaluator finished — stream final recommendations
                if "evaluator" in chunk:
                    recs = chunk["evaluator"].get("recommendations", [])
                    yield _sse(ResultEvent(recommendations=recs).model_dump())

            yield _sse(DoneEvent().model_dump())
        except Exception as e:
            yield _sse(ErrorEvent(message=str(e)).model_dump())
        finally:
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")

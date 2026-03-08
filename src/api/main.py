import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Make the recommendation graph importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pipeline/recommendation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from graph import build_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/recommend")
async def recommend(
    input_type: str = Form(...),
    text: str = Form(default=""),
    image: UploadFile = File(default=None),
):
    if input_type == "image" and image:
        suffix = Path(image.filename).suffix or ".jpeg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image.read())
        tmp.close()
        graph_input = {"input_type": "image", "mood_input": tmp.name}
    else:
        tmp = None
        graph_input = {"input_type": "text", "mood_input": text}

    async def generate():
        try:
            async for chunk in graph.astream(graph_input):
                # extract_mood node finished — stream moods immediately
                if "extract_mood" in chunk:
                    moods = chunk["extract_mood"].get("extracted_moods", [])
                    if moods:
                        yield _sse({"type": "moods", "moods": moods})

                # extract_accord node finished
                if "extract_accord" in chunk:
                    accords = chunk["extract_accord"].get("extracted_accords", [])
                    if accords:
                        yield _sse({"type": "accords", "accords": accords})

                # result_enricher finished — stream final recommendations
                if "result_enricher" in chunk:
                    recs = chunk["result_enricher"].get("recommendations", [])
                    yield _sse({"type": "result", "recommendations": recs})

            yield _sse({"type": "done"})
        finally:
            if tmp:
                os.unlink(tmp.name)

    return StreamingResponse(generate(), media_type="text/event-stream")

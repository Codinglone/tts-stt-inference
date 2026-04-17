"""
FastAPI service exposing TTS (Text-to-Speech) endpoint for Kinyarwanda.

Endpoints
---------
GET  /health   - Liveness check.
POST /tts      - Convert text → WAV audio.
"""

import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from xtt_tts_script import TTSService

_tts: TTSService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts
    print("Loading TTS model …")
    _tts = TTSService()
    print("TTS model loaded — service ready.")
    yield


app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech for Kinyarwanda (XTTS-based).",
    version="1.0.0",
    lifespan=lifespan,
)


class TTSRequest(BaseModel):
    text: str


@app.get("/health", tags=["utility"])
def health() -> dict:
    return {
        "status": "ok",
        "tts_loaded": _tts is not None,
    }


@app.post(
    "/tts",
    response_class=FileResponse,
    tags=["tts"],
    summary="Text → Speech (Kinyarwanda)",
    responses={200: {"content": {"audio/wav": {}}}},
)
def text_to_speech(request: TTSRequest) -> FileResponse:
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded yet.")

    output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    try:
        _tts.synthesize(text=request.text, output_path=output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="output.wav",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tts_main:app", host="0.0.0.0", port=8001, reload=False)

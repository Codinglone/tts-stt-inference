"""
FastAPI service exposing STT (Speech-to-Text) endpoint only.

Endpoints
---------
GET  /health   - Liveness check.
POST /stt      - Convert uploaded WAV → transcribed text.
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from stt_service import STTService

# ---------------------------------------------------------------------------
# Shared model instance (loaded once on startup)
# ---------------------------------------------------------------------------

_stt: STTService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stt
    print("Loading STT model …")
    _stt = STTService()
    print("STT model loaded — service ready.")
    yield


app = FastAPI(
    title="STT Service",
    description="Speech-to-Text (OWSM v4) API.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class STTResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["utility"])
def health() -> dict:
    """Return service status."""
    return {
        "status": "ok",
        "stt_loaded": _stt is not None,
    }


@app.post("/stt", response_model=STTResponse, tags=["stt"], summary="Speech → Text")
async def speech_to_text(file: UploadFile = File(...)) -> STTResponse:
    """
    Transcribe an uploaded WAV audio file.

    Upload a WAV file as `multipart/form-data` under the key **file**.
    Returns the recognised text.
    """
    if _stt is None:
        raise HTTPException(status_code=503, detail="STT model not loaded yet.")

    ext = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = _stt.transcribe(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return STTResponse(text=text)


# ---------------------------------------------------------------------------
# Entry point (for local development: python stt_main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("stt_main:app", host="0.0.0.0", port=8000, reload=False)

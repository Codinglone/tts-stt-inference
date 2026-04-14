"""
FastAPI service exposing TTS and STT endpoints.

Endpoints
---------
GET  /health         - Liveness check.
POST /tts            - Convert text → WAV audio.
POST /stt            - Convert uploaded WAV → transcribed text.

Environment variables
---------------------
HF_TOKEN   - Hugging Face token (required for private Kinyarwanda model).
MODEL_DIR  - Local path for the Kinyarwanda model (default: ./kinyarwanda_health_model).
"""

import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stt_service import STTService
from xtt_tts_script import TTSService

# ---------------------------------------------------------------------------
# Shared model instances (loaded once on startup)
# ---------------------------------------------------------------------------

_tts: TTSService | None = None
_stt: STTService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts, _stt
    print("Loading models …")
    _tts = TTSService()
    _stt = STTService()
    print("All models loaded — service ready.")
    yield
    # Nothing to clean up on shutdown.


app = FastAPI(
    title="TTS / STT Service",
    description="Text-to-Speech (XTTS v2 + Kinyarwanda) and Speech-to-Text (OWSM v4) API.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    use_kinyarwanda: bool = False


class TTSResponse(BaseModel):
    message: str
    file: str


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
        "tts_loaded": _tts is not None,
        "stt_loaded": _stt is not None,
    }


@app.post(
    "/tts",
    response_class=FileResponse,
    tags=["tts"],
    summary="Text → Speech",
    responses={200: {"content": {"audio/wav": {}}}},
)
def text_to_speech(request: TTSRequest) -> FileResponse:
    """
    Synthesise speech from *text*.

    - **language**: BCP-47 code used by XTTS v2 (e.g. `"en"`, `"fr"`, `"rw"`).
    - **use_kinyarwanda**: When `true`, uses the custom Kinyarwanda health model.
    """
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded yet.")

    output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    try:
        _tts.synthesize(
            text=request.text,
            output_path=output_path,
            language=request.language,
            use_kinyarwanda=request.use_kinyarwanda,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="output.wav",
        background=None,  # File is deleted after response in cleanup below.
    )


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
# Entry point (for local development: python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

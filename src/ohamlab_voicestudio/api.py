
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .studio import VoiceStudio


MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"
}


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    seed: int = 0


def create_app(
    model: str = "rahul7star/tts-nano-o-hi",
    device: str = "cpu",
) -> FastAPI:

    app = FastAPI(
        title="OhamLab VoiceStudio API",
        version="0.1.0",
    )

    studio = VoiceStudio(model=model, device=device)
    inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "ohamlab-voicestudio",
            "model": model,
            "device": device,
        }

    @app.get("/v1/voices")
    def voices():
        try:
            with inference_lock:
                return {"voices": studio.voices()}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest):
        try:
            with inference_lock:
                sample_rate, audio = studio.speak(
                    request.text.strip(),
                    voice=request.voice,
                    seed=request.seed,
                )

            buffer = io.BytesIO()
            sf.write(buffer, np.asarray(audio), int(sample_rate),
                     format="WAV")

            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'attachment; filename="speech.wav"'
                },
            )

        except Exception as exc:
            raise HTTPException(500, str(exc))

    def save_upload(upload: UploadFile) -> str:
        extension = Path(upload.filename or "").suffix.lower()

        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(400, "Unsupported audio format")

        fd, path = tempfile.mkstemp(suffix=extension)
        size = 0

        try:
            with os.fdopen(fd, "wb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)

                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Audio file too large")

                    output.write(chunk)

            return path

        except Exception:
            if os.path.exists(path):
                os.remove(path)
            raise

    @app.post("/v1/audio/clone")
    def clone(
        text: str = Form(...),
        consent: bool = Form(False),
        seed: int = Form(0),
        reference_audio: UploadFile = File(...),
    ):
        if not consent:
            raise HTTPException(
                400,
                "Confirm you have permission to use this voice."
            )

        path = save_upload(reference_audio)

        try:
            with inference_lock:
                sample_rate, audio = studio.clone(
                    text.strip(),
                    path,
                    seed=seed,
                )

            buffer = io.BytesIO()
            sf.write(buffer, np.asarray(audio), int(sample_rate),
                     format="WAV")

            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
            )

        except Exception as exc:
            raise HTTPException(500, str(exc))

        finally:
            if os.path.exists(path):
                os.remove(path)

    @app.post("/v1/audio/transcriptions")
    def transcribe(
        audio: UploadFile = File(...),
        language: str | None = Form(None),
    ):
        path = save_upload(audio)

        try:
            with inference_lock:
                result = studio.transcribe(path, language=language)

            return result

        except Exception as exc:
            raise HTTPException(500, str(exc))

        finally:
            if os.path.exists(path):
                os.remove(path)

    return app

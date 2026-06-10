import asyncio

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.speaker_embed import get_speaker_embedding
from app.runtime.limiter import CapacityTimeout, gpu_gate
from app.schemas.audio import AudioInput, EmbeddingResponse
from app.utils.audio_io import (
    AudioInputError,
    load_waveform,
    resolve_audio_bytes,
    to_mono_resampled,
)
from app.utils.vad import trim_to_speech

router = APIRouter(prefix="/embed", tags=["embedding"])


def _embed(mono):
    """VAD trim + speaker embedding (the gated compute step)."""
    return get_speaker_embedding(trim_to_speech(mono))


@router.post("/speaker", response_model=EmbeddingResponse)
async def speaker_embedding(request: AudioInput):
    # I/O, decode and resample run off the event loop, ungated.
    try:
        raw = await asyncio.to_thread(
            resolve_audio_bytes, request.base64, request.audio_uri
        )
        waveform, sr = await asyncio.to_thread(load_waveform, raw)
    except AudioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration = waveform.shape[-1] / sr
    if duration > settings.max_audio_duration_sec:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too long: {duration:.1f}s > "
            f"{settings.max_audio_duration_sec}s",
        )

    mono = await asyncio.to_thread(
        to_mono_resampled, waveform, sr, settings.speaker_sample_rate
    )

    try:
        async with gpu_gate.slot("speaker"):
            emb = await asyncio.to_thread(_embed, mono)
    except CapacityTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail="GPU at capacity, retry shortly",
            headers={"Retry-After": str(int(settings.gpu_acquire_timeout_sec))},
        ) from exc

    return EmbeddingResponse(embedding=emb, dim=len(emb))

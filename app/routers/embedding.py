import asyncio

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse

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
    """Compute a speaker voiceprint from JSON audio input.

    Accepts **exactly one** of `base64` (inline audio) or `audioUri` (a public
    HTTP(S) URL). The audio is downmixed to mono, resampled to 16 kHz, trimmed to
    speech with Silero VAD, then embedded with ECAPA-TDNN. Returns the **raw**
    (un-normalized) 192-dimensional embedding and its dimension — normalize
    client-side for cosine similarity.

    Errors: **400** undecodable audio / failed download · **413** audio longer than
    the configured limit · **422** zero or both inputs provided · **503** GPU at
    capacity (retry after the `Retry-After` header).
    """
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


@router.post("/speaker/upload", response_model=EmbeddingResponse)
async def embed_upload(file: UploadFile = File(...)):
    """Compute a speaker voiceprint from a multipart file upload.

    Same pipeline as `POST /embed/speaker` (mono → 16 kHz → VAD trim → ECAPA-TDNN),
    but takes a binary audio blob as the multipart `file` field instead of JSON —
    use this for uploads from a browser form or app. Returns the raw 192-dimensional
    embedding and its dimension.

    Errors: **400** undecodable audio · **413** upload over the size cap or audio
    longer than the configured limit · **503** GPU at capacity (retry after the
    `Retry-After` header).
    """
    # Read in chunks with a running size cap so absurd uploads are rejected before
    # we fully buffer them.
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > settings.max_download_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds max size ({settings.max_download_bytes} bytes)",
            )
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
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


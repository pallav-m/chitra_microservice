import asyncio
import base64
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.demucs_model import separate
from app.models.loader import ModelRegistry
from app.runtime.limiter import CapacityTimeout, gpu_gate
from app.schemas.audio import SeparationRequest, SeparationResponse
from app.utils.audio_io import (
    AudioInputError,
    load_waveform,
    postprocess,
    resolve_audio_bytes,
    tensor_to_wav_bytes,
)

router = APIRouter(prefix="/separate", tags=["separation"])


def _encode(stems, source_sr, response_format, stem):
    """Post-process + encode stems (CPU-bound; run off the event loop, ungated)."""
    vocals, out_sr = postprocess(stems["vocals"], source_sr)
    no_vocals, _ = postprocess(stems["no_vocals"], source_sr)

    if response_format == "wav":
        chosen = vocals if stem == "vocals" else no_vocals
        return tensor_to_wav_bytes(chosen, out_sr)

    return SeparationResponse(
        vocals=base64.b64encode(tensor_to_wav_bytes(vocals, out_sr)).decode(),
        no_vocals=base64.b64encode(tensor_to_wav_bytes(no_vocals, out_sr)).decode(),
        sample_rate=out_sr,
    )


@router.post("", response_model=None)
async def separate_endpoint(request: SeparationRequest):
    # I/O and decode run off the event loop but are NOT GPU-gated — we never
    # hold a GPU slot while waiting on the network.
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

    # Gate only the GPU forward pass; run it in a thread so the event loop stays
    # responsive (health checks, other requests).
    try:
        async with gpu_gate.slot("demucs"):
            stems = await asyncio.to_thread(separate, waveform, sr)
    except CapacityTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail="GPU at capacity, retry shortly",
            headers={"Retry-After": str(int(settings.gpu_acquire_timeout_sec))},
        ) from exc

    source_sr = ModelRegistry.get_demucs().samplerate
    result = await asyncio.to_thread(
        _encode, stems, source_sr, request.response_format, request.stem
    )

    if request.response_format == "wav":
        return StreamingResponse(io.BytesIO(result), media_type="audio/wav")
    return result

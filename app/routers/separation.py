import asyncio
import base64
import io
import zipfile

from fastapi import APIRouter, File, HTTPException, UploadFile
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


# --- shared core --------------------------------------------------------- #

async def _separate_waveform(waveform, sr):
    """Gated GPU separation run off the event loop. Shared by every transport.

    Raises HTTPException(503) when the per-model GPU concurrency cap is saturated.
    """
    try:
        async with gpu_gate.slot("demucs"):
            stems = await asyncio.to_thread(separate, waveform, sr)
    except CapacityTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail="GPU at capacity, retry shortly",
            headers={"Retry-After": str(int(settings.gpu_acquire_timeout_sec))},
        ) from exc
    return stems, ModelRegistry.get_demucs().samplerate


def _check_duration(waveform, sr):
    duration = waveform.shape[-1] / sr
    if duration > settings.max_audio_duration_sec:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too long: {duration:.1f}s > "
            f"{settings.max_audio_duration_sec}s",
        )


def _encode(stems, source_sr, response_format, stem):
    """Post-process + encode stems for the JSON route (CPU-bound, ungated)."""
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


def _encode_zip(stems, source_sr) -> bytes:
    """Both stems as a ZIP (CPU-bound, ungated). STORED: WAV PCM compresses little."""
    vocals, out_sr = postprocess(stems["vocals"], source_sr)
    no_vocals, _ = postprocess(stems["no_vocals"], source_sr)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("vocals.wav", tensor_to_wav_bytes(vocals, out_sr))
        zf.writestr("bgm.wav", tensor_to_wav_bytes(no_vocals, out_sr))
    return buf.getvalue()


# --- routes -------------------------------------------------------------- #

@router.post("", response_model=None)
async def separate_endpoint(request: SeparationRequest):
    """Separate audio into vocals and accompaniment from JSON input.

    Accepts **exactly one** of `base64` (inline audio) or `audioUri` (a public
    HTTP(S) URL). Splits the audio with Demucs into a **vocals** stem and a
    **no_vocals** stem (drums + bass + other).

    Response depends on `responseFormat`:
    - `"json"` (default) — both stems base64-encoded WAV plus the output sample rate.
    - `"wav"` — streams a single stem (`stem` = `vocals` | `no_vocals`) as `audio/wav`.

    Errors: **400** undecodable audio / failed download · **413** audio longer than
    the configured limit · **422** zero or both inputs provided · **503** GPU at
    capacity (retry after the `Retry-After` header).
    """
    # I/O and decode run off the event loop but are NOT GPU-gated — we never
    # hold a GPU slot while waiting on the network.
    try:
        raw = await asyncio.to_thread(
            resolve_audio_bytes, request.base64, request.audio_uri
        )
        waveform, sr = await asyncio.to_thread(load_waveform, raw)
    except AudioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _check_duration(waveform, sr)
    stems, source_sr = await _separate_waveform(waveform, sr)

    result = await asyncio.to_thread(
        _encode, stems, source_sr, request.response_format, request.stem
    )
    if request.response_format == "wav":
        return StreamingResponse(io.BytesIO(result), media_type="audio/wav")
    return result


@router.post("/upload", response_model=None)
async def separate_upload(file: UploadFile = File(...)):
    """Separate audio from a multipart file upload, returning both stems as a ZIP.

    Takes a binary audio blob as the multipart `file` field (use this for uploads
    from a browser form or app). Runs the same Demucs separation as `POST /separate`
    and streams back an `application/zip` containing **`vocals.wav`** and
    **`bgm.wav`** (`Content-Disposition: attachment`).

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

    _check_duration(waveform, sr)
    stems, source_sr = await _separate_waveform(waveform, sr)

    zip_bytes = await asyncio.to_thread(_encode_zip, stems, source_sr)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="stems.zip"'},
    )

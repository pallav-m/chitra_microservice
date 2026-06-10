import base64
import binascii
import io

import httpx
import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import settings


class AudioInputError(ValueError):
    """Raised when audio input cannot be resolved or decoded."""


def resolve_audio_bytes(b64: str | None, audio_uri: str | None) -> bytes:
    """Resolve raw audio bytes from inline base64 or a remote URL.

    Isolated so an NVCF asset-download branch can slot in later without
    touching callers.
    """
    if b64 is not None:
        try:
            return base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AudioInputError(f"Invalid base64 audio: {exc}") from exc

    if audio_uri is not None:
        return _download(audio_uri)

    raise AudioInputError("No audio input provided")


def _download(url: str) -> bytes:
    try:
        with httpx.Client(
            timeout=settings.download_timeout_sec, follow_redirects=True
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > settings.max_download_bytes:
                        raise AudioInputError(
                            f"audioUri exceeds max_download_bytes "
                            f"({settings.max_download_bytes} bytes)"
                        )
                    chunks.append(chunk)
        return b"".join(chunks)
    except httpx.HTTPError as exc:
        raise AudioInputError(f"Failed to download audioUri: {exc}") from exc


def load_waveform(raw: bytes) -> tuple[torch.Tensor, int]:
    """Decode raw audio bytes into a (C, T) waveform and sample rate.

    Uses libsndfile (soundfile) — handles WAV/FLAC/OGG without a system FFmpeg
    install. Compressed formats like MP3 require libsndfile >= 1.1.
    """
    try:
        # always_2d -> (frames, channels); transpose to torch's (channels, frames).
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioInputError(f"Could not decode audio: {exc}") from exc
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sr


def postprocess(stem: torch.Tensor, source_sr: int) -> tuple[torch.Tensor, int]:
    """Optionally downmix to mono and resample a stem for output.

    When `output_sample_rate` is 0 the stem is returned untouched at its native
    rate; otherwise it is averaged to mono and resampled.
    """
    if settings.output_sample_rate <= 0:
        return stem, source_sr
    mono = stem.mean(dim=0, keepdim=True)
    resampled = torchaudio.functional.resample(
        mono, source_sr, settings.output_sample_rate
    )
    return resampled, settings.output_sample_rate


def to_mono_resampled(
    waveform: torch.Tensor, source_sr: int, target_sr: int
) -> torch.Tensor:
    """Downmix to mono and resample to target_sr, returning shape (1, T).

    Used by the embedding path, which always needs mono at the speaker model's
    rate regardless of the input rate or the separation output_sample_rate.
    """
    mono = waveform.mean(dim=0, keepdim=True)
    if source_sr != target_sr:
        mono = torchaudio.functional.resample(mono, source_sr, target_sr)
    return mono


def tensor_to_wav_bytes(waveform: torch.Tensor, sr: int) -> bytes:
    """Encode a (C, T) float tensor as a 16-bit PCM WAV via libsndfile."""
    # soundfile expects (frames, channels); PCM_16 is the most portable WAV form.
    data = waveform.detach().cpu().numpy().T.astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

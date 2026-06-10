import torch
from silero_vad import get_speech_timestamps

from app.config import settings
from app.models.loader import ModelRegistry


def trim_to_speech(waveform: torch.Tensor) -> torch.Tensor:
    """Keep only speech regions of a mono 16 kHz clip, shaped (1, T).

    Concatenates the speech segments Silero VAD detects. If no speech is found
    (very short/quiet clip), returns the input unchanged so embedding still
    proceeds rather than failing.
    """
    model = ModelRegistry.get_vad()
    audio = waveform.squeeze(0)  # (T,) — Silero expects 1-D

    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=settings.speaker_sample_rate,
        threshold=settings.vad_threshold,
    )
    if not timestamps:
        return waveform

    speech = torch.cat([audio[ts["start"] : ts["end"]] for ts in timestamps])
    return speech.unsqueeze(0)  # back to (1, T)

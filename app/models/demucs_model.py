import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio

from app.config import settings
from app.models.loader import ModelRegistry


def separate(waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
    """Separate audio into vocals and non-vocals (accompaniment) stems.

    Returns CPU tensors shaped (C, T) at the model's native sample rate, keyed
    "vocals" and "no_vocals". `no_vocals` is the sum of every non-vocal source
    (drums + bass + other).
    """
    model = ModelRegistry.get_demucs()

    wav = convert_audio(waveform, sr, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0).to(settings.device)  # (1, C, T)

    with torch.no_grad():
        # split=True, overlap=0.25 chunk long audio internally, keeping VRAM
        # flat regardless of clip duration — do not remove.
        sources = apply_model(
            model,
            wav,
            device=settings.device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )

    sources = sources[0]  # (S, C, T)
    # Resolve by name — source ordering varies by model variant.
    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]
    no_vocals = torch.stack(
        [sources[i] for i in range(len(model.sources)) if i != vocals_idx]
    ).sum(dim=0)

    return {"vocals": vocals.cpu(), "no_vocals": no_vocals.cpu()}

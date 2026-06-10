import torch

from app.models.loader import ModelRegistry


def get_speaker_embedding(waveform: torch.Tensor) -> list[float]:
    """Compute a 192-dim ECAPA-TDNN speaker embedding.

    Expects mono audio at the speaker model's sample rate, shaped (1, T).
    Returns the raw (un-normalized) embedding as a plain list.
    """
    encoder = ModelRegistry.get_speaker_encoder()
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)  # (1, 1, 192)
    return embedding.squeeze().tolist()

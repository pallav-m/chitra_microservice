from demucs.pretrained import get_model
from speechbrain.inference.speaker import EncoderClassifier
from silero_vad import load_silero_vad

from app.config import settings


class ModelRegistry:
    """Process-wide singleton holding every model.

    Models are loaded once (lazily, then cached) and shared across all
    requests. This is the ONLY place a model should be constructed — routers
    and inference modules must fetch via these getters, never instantiate
    inline. The FastAPI lifespan handler calls these eagerly at startup so the
    first real request pays no cold-start cost.
    """

    _demucs = None
    _speaker_encoder = None
    _vad = None

    @classmethod
    def get_demucs(cls):
        if cls._demucs is None:
            model = get_model(settings.demucs_model)
            model.to(settings.device)
            model.eval()
            cls._demucs = model
        return cls._demucs

    @classmethod
    def get_speaker_encoder(cls):
        if cls._speaker_encoder is None:
            # SpeechBrain only understands "cpu"/"cuda"; it has no MPS branch and
            # crashes on a bare "mps" device. Fall back to CPU on Apple Silicon
            # (ECAPA is light enough that this is fine for local dev).
            device = "cpu" if settings.device.startswith("mps") else settings.device
            cls._speaker_encoder = EncoderClassifier.from_hparams(
                source=settings.speaker_model_source,
                run_opts={"device": device},
            )
        return cls._speaker_encoder

    @classmethod
    def get_vad(cls):
        # Silero VAD is a small CPU model; it runs fine off-GPU regardless of
        # settings.device.
        if cls._vad is None:
            cls._vad = load_silero_vad()
        return cls._vad

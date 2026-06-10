import threading

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
    # Silero VAD is a stateful TorchScript model and is NOT safe to call from
    # multiple threads at once (concurrent calls crash the process). Inference
    # runs in a threadpool, so each thread gets its own VAD instance. `_vad`
    # stays as a readiness sentinel (non-None once any instance has loaded).
    _vad = None
    _vad_local = threading.local()

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
        # One instance per calling thread (Silero VAD isn't thread-safe). The
        # model is tiny, so per-thread copies are cheap and threads are reused by
        # the inference threadpool, so each is created at most once.
        model = getattr(cls._vad_local, "model", None)
        if model is None:
            model = load_silero_vad()
            cls._vad_local.model = model
            cls._vad = model  # readiness sentinel for /v1/health/ready
        return model

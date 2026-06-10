import torch
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_device() -> str:
    """Pick the best available backend: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Inference device: "cuda", "mps", or "cpu". Leave empty (or "auto") to
    # auto-detect the best available backend at startup.
    device: str = ""
    # Demucs pretrained model name (e.g. "htdemucs", "htdemucs_ft").
    demucs_model: str = "htdemucs_ft"
    # Reject inputs longer than this (seconds) before they reach the GPU.
    max_audio_duration_sec: float = 60.0
    # 0 = keep model-native rate (44.1 kHz) and channel layout. Non-zero
    # downmixes each stem to mono and resamples to this rate (e.g. 16000 for ASR).
    output_sample_rate: int = 0
    # audioUri download limits.
    download_timeout_sec: float = 30.0
    max_download_bytes: int = 100 * 1024 * 1024  # 100 MB

    # Speaker embedding (SpeechBrain ECAPA-TDNN).
    speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    # ECAPA's required input rate — independent of output_sample_rate (which is
    # separation-only). Inputs are always resampled to this for embedding.
    speaker_sample_rate: int = 16000
    # VAD speech-detection threshold (Silero); higher = stricter.
    vad_threshold: float = 0.5

    # --- GPU concurrency (per-instance) ---
    # Hard cap on concurrent GPU work per model. Size so the worst-case
    # simultaneous mix of in-flight requests fits the GPU's VRAM.
    demucs_max_concurrency: int = 4
    speaker_max_concurrency: int = 8
    # Optional aggregate ceiling across all models (0 = disabled).
    gpu_max_concurrency: int = 0
    # How long a request waits for a free slot before returning 503.
    gpu_acquire_timeout_sec: float = 30.0

    @model_validator(mode="after")
    def _resolve_device(self) -> "Settings":
        if self.device.strip().lower() in ("", "auto"):
            self.device = _detect_device()
        return self


settings = Settings()

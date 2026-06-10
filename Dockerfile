# CUDA 12.4 + cuDNN runtime. linux/amd64 pinned: the PyTorch CUDA wheels are
# x86_64-only, and NVCF GPU backends are x86_64. (Builds on Apple Silicon run
# under emulation.) torch 2.12's manylinux wheel bundles its own CUDA runtime,
# so this base mainly supplies a driver-compatible userspace + cuDNN.
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg: broad audio-format decode support. libsndfile1: backs soundfile (the
# wheel bundles it, but installing the system lib is belt-and-suspenders).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv provisions Python 3.11 (per .python-version) — no system python needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# uv-managed Python and the project venv live in fixed paths so a non-root user
# can read them at runtime. Model weights cache under /opt/models (baked below).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TORCH_HOME=/opt/models/torch \
    HF_HOME=/opt/models/hf \
    PYTHONPATH=/app \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Install dependencies first (cached layer). --no-install-project: this is an
# application run from source, not a built package, so we only need its deps.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code.
COPY app ./app

# Bake all co-served model weights into the image so NVCF cold starts stay
# offline. Sources come from settings so they can't drift from runtime config.
RUN python -c "import app.config as c; \
from demucs.pretrained import get_model; get_model(c.settings.demucs_model); \
from speechbrain.inference.speaker import EncoderClassifier; \
EncoderClassifier.from_hparams(source=c.settings.speaker_model_source); \
from silero_vad import load_silero_vad; load_silero_vad()"

# NVCF requires containers run as non-root. Own /app and the read paths.
RUN useradd --create-home --no-log-init appuser \
    && chown -R appuser:appuser /app /opt/uv-python /opt/models
USER appuser

# Default to GPU; override with -e DEVICE=cpu for non-GPU hosts.
ENV DEVICE=cuda

# Health and inference share this port (NVCF inferencePort).
EXPOSE 8000

# Mirrors NVCF's readiness probe (/v1/health/ready returns 200 once the model
# is loaded). Generous start-period covers model load on boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/v1/health/ready').status==200 else 1)"

# --workers 1 is mandatory: extra workers each load their own model copy and
# multiply VRAM. Scale concurrency via async, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

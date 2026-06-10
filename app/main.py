import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.models.loader import ModelRegistry
from app.routers import embedding, health, separation

logger = logging.getLogger("chitra")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load all co-served models once at startup so the first request
    # pays no cold-start cost. Keep uvicorn at --workers 1: each worker would
    # load its own copy and multiply VRAM.
    ModelRegistry.get_demucs()
    ModelRegistry.get_speaker_encoder()
    ModelRegistry.get_vad()
    logger.info(
        "GPU concurrency caps: demucs=%d speaker=%d global=%d acquire_timeout=%.0fs",
        settings.demucs_max_concurrency,
        settings.speaker_max_concurrency,
        settings.gpu_max_concurrency,
        settings.gpu_acquire_timeout_sec,
    )
    yield


app = FastAPI(title="Chitra Audio Service", lifespan=lifespan,
              swagger_ui_parameters={
            "displayRequestDuration": True,
            "filter": True,
            "docExpansion": "none"
        })
app.include_router(health.router)
app.include_router(separation.router)
app.include_router(embedding.router)

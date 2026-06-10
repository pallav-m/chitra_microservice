from fastapi import APIRouter, Response

from app.models.loader import ModelRegistry

# NVCF health checks must return HTTP 200 on the inference port. /v1/health/ready
# is the readiness path; /v1/health/live is a liveness probe.
router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("/live")
async def live():
    return {"status": "live"}


@router.get("/ready")
async def ready(response: Response):
    # Readiness reflects that all co-served models are loaded. Under normal
    # operation the lifespan handler loads them before traffic is served, so
    # this is a 200.
    loaded = (
        ModelRegistry._demucs is not None
        and ModelRegistry._speaker_encoder is not None
        and ModelRegistry._vad is not None
    )
    if not loaded:
        response.status_code = 503
        return {"status": "loading"}
    return {"status": "ready"}

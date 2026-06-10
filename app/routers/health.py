from fastapi import APIRouter, Response

from app.models.loader import ModelRegistry

# NVCF health checks must return HTTP 200 on the inference port. /v1/health/ready
# is the readiness path; /v1/health/live is a liveness probe.
router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("/live")
async def live():
    """Liveness probe.

    Returns `200 {"status": "live"}` as soon as the process is up. Used to detect a
    hung/crashed container; it does **not** check whether models are loaded.
    """
    return {"status": "live"}


@router.get("/ready")
async def ready(response: Response):
    """Readiness probe (NVCF health check).

    Returns `200 {"status": "ready"}` only once **all** co-served models (Demucs,
    speaker encoder, VAD) are loaded; otherwise `503 {"status": "loading"}` while the
    service is still starting up. Point your NVCF function's health URI at this path.
    """
    # Under normal operation the lifespan handler loads every model before traffic
    # is served, so this is a 200.
    loaded = (
        ModelRegistry._demucs is not None
        and ModelRegistry._speaker_encoder is not None
        and ModelRegistry._vad is not None
    )
    if not loaded:
        response.status_code = 503
        return {"status": "loading"}
    return {"status": "ready"}

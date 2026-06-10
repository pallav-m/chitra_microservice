import asyncio
from contextlib import asynccontextmanager

from app.config import settings


class CapacityTimeout(Exception):
    """A GPU slot could not be acquired within the configured timeout."""


class ModelGate:
    """Per-model concurrency caps over stdlib semaphores.

    Each model name gets its own ``asyncio.Semaphore`` sized from config, giving
    a hard ceiling on that model's concurrent GPU work (and therefore its
    worst-case activation VRAM). An optional process-wide semaphore caps total
    in-flight GPU work across all models.

    No custom scheduling: on Python 3.11 ``asyncio.wait_for(sem.acquire())`` is
    cancellation-safe, so a timed-out waiter releases cleanly with no permit leak.
    """

    def __init__(
        self, limits: dict[str, int], timeout: float, global_limit: int = 0
    ):
        self._limits = dict(limits)
        self._sems = {name: asyncio.Semaphore(n) for name, n in limits.items()}
        self._global = asyncio.Semaphore(global_limit) if global_limit > 0 else None
        self._global_limit = global_limit
        self._timeout = timeout

    async def _acquire(self, sem: asyncio.Semaphore) -> None:
        try:
            await asyncio.wait_for(sem.acquire(), self._timeout)
        except asyncio.TimeoutError as exc:
            raise CapacityTimeout() from exc

    @asynccontextmanager
    async def slot(self, model: str):
        """Acquire a slot for ``model`` (global backstop first, then per-model).

        Raises ``CapacityTimeout`` if either semaphore can't be acquired in time.
        """
        sem = self._sems[model]

        global_held = False
        if self._global is not None:
            await self._acquire(self._global)  # raises -> nothing held yet
            global_held = True

        try:
            await self._acquire(sem)
        except BaseException:
            if global_held:
                self._global.release()
            raise

        try:
            yield
        finally:
            sem.release()
            if global_held:
                self._global.release()

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-model {limit, in_use, available} snapshot for tests/observability."""
        out: dict[str, dict[str, int]] = {}
        for name, sem in self._sems.items():
            limit = self._limits[name]
            available = sem._value  # asyncio.Semaphore exposes remaining permits
            out[name] = {
                "limit": limit,
                "in_use": limit - available,
                "available": available,
            }
        if self._global is not None:
            out["_global"] = {
                "limit": self._global_limit,
                "in_use": self._global_limit - self._global._value,
                "available": self._global._value,
            }
        return out


# Process-wide singleton. Model names here are the keys routers pass to slot().
# Adding a model = a new config field + an entry here + a slot("name") call.
gpu_gate = ModelGate(
    limits={
        "demucs": settings.demucs_max_concurrency,
        "speaker": settings.speaker_max_concurrency,
    },
    timeout=settings.gpu_acquire_timeout_sec,
    global_limit=settings.gpu_max_concurrency,
)

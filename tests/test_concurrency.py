import asyncio
import base64
import io
import threading
import time

import httpx
import numpy as np
import soundfile as sf
import torch

from app.runtime.limiter import CapacityTimeout, ModelGate


# --------------------------------------------------------------------------- #
# Unit tests: ModelGate                                                        #
# --------------------------------------------------------------------------- #

def test_per_model_cap_blocks_extra():
    async def run():
        gate = ModelGate({"demucs": 2}, timeout=0.1)
        async with gate.slot("demucs"):
            async with gate.slot("demucs"):
                assert gate.stats()["demucs"]["in_use"] == 2
                # third concurrent acquire must time out
                try:
                    async with gate.slot("demucs"):
                        raise AssertionError("should not acquire a 3rd slot")
                except CapacityTimeout:
                    pass
        assert gate.stats()["demucs"]["in_use"] == 0  # all released

    asyncio.run(run())


def test_models_are_independent():
    async def run():
        gate = ModelGate({"demucs": 1, "speaker": 1}, timeout=0.5)
        async with gate.slot("demucs"):
            # saturating demucs must not block speaker
            async with gate.slot("speaker"):
                s = gate.stats()
                assert s["demucs"]["in_use"] == 1
                assert s["speaker"]["in_use"] == 1

    asyncio.run(run())


def test_timeout_raises_capacity_timeout():
    async def run():
        gate = ModelGate({"demucs": 1}, timeout=0.05)
        async with gate.slot("demucs"):
            try:
                async with gate.slot("demucs"):
                    raise AssertionError("unreachable")
            except CapacityTimeout:
                return
        raise AssertionError("expected CapacityTimeout")

    asyncio.run(run())


def test_global_backstop_caps_aggregate():
    async def run():
        # per-model room is ample, but the global ceiling is 2
        gate = ModelGate({"a": 5, "b": 5}, timeout=0.05, global_limit=2)
        async with gate.slot("a"):
            async with gate.slot("b"):
                assert gate.stats()["_global"]["in_use"] == 2
                try:
                    async with gate.slot("a"):
                        raise AssertionError("global backstop not enforced")
                except CapacityTimeout:
                    pass
        # global fully released after exit
        assert gate.stats()["_global"]["in_use"] == 0

    asyncio.run(run())


def test_no_leak_after_model_timeout_with_global():
    """A model-level timeout must release the already-acquired global permit."""
    async def run():
        gate = ModelGate({"demucs": 1}, timeout=0.05, global_limit=3)
        async with gate.slot("demucs"):  # demucs full (1), global=1
            try:
                async with gate.slot("demucs"):  # acquires global, then times out
                    raise AssertionError("unreachable")
            except CapacityTimeout:
                pass
            # the timed-out attempt must not have leaked a global permit
            assert gate.stats()["_global"]["in_use"] == 1

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Integration tests: gating + event-loop responsiveness through the app       #
# --------------------------------------------------------------------------- #

def _wav_b64(seconds=0.2, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    sig = np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeDemucs:
    samplerate = 44100


def _make_slow_separate(hold_s, tracker):
    """A stand-in for separate() that records peak concurrency."""
    def slow(waveform, sr):
        with tracker["lock"]:
            tracker["live"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["live"])
        time.sleep(hold_s)
        with tracker["lock"]:
            tracker["live"] -= 1
        small = torch.zeros((2, 1000), dtype=torch.float32)
        return {"vocals": small, "no_vocals": small}
    return slow


def _client():
    from app.main import app
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def test_concurrency_capped_and_503_on_overload(monkeypatch):
    import app.routers.separation as sep

    tracker = {"live": 0, "peak": 0, "lock": threading.Lock()}
    monkeypatch.setattr(sep, "separate", _make_slow_separate(0.4, tracker))
    monkeypatch.setattr(sep.ModelRegistry, "get_demucs", classmethod(lambda cls: _FakeDemucs()))
    # cap 2, short queue timeout so overflow requests fail fast as 503
    monkeypatch.setattr(sep, "gpu_gate", ModelGate({"demucs": 2}, timeout=0.3))

    async def run():
        payload = {"base64": _wav_b64(), "responseFormat": "json"}
        async with _client() as client:
            tasks = [client.post("/separate", json=payload) for _ in range(6)]
            return await asyncio.gather(*tasks)

    responses = asyncio.run(run())
    codes = [r.status_code for r in responses]
    assert tracker["peak"] == 2, f"cap exceeded: peak={tracker['peak']}"
    assert all(c in (200, 503) for c in codes), codes
    assert 503 in codes, "expected some requests to be shed as 503"
    assert 200 in codes, "expected some requests to succeed"


def test_health_responsive_during_separation(monkeypatch):
    import app.routers.separation as sep

    tracker = {"live": 0, "peak": 0, "lock": threading.Lock()}
    monkeypatch.setattr(sep, "separate", _make_slow_separate(0.6, tracker))
    monkeypatch.setattr(sep.ModelRegistry, "get_demucs", classmethod(lambda cls: _FakeDemucs()))
    monkeypatch.setattr(sep, "gpu_gate", ModelGate({"demucs": 1}, timeout=5.0))
    # readiness needs all models "loaded"
    from app.models.loader import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "_demucs", object())
    monkeypatch.setattr(ModelRegistry, "_speaker_encoder", object())
    monkeypatch.setattr(ModelRegistry, "_vad", object())

    async def run():
        async with _client() as client:
            sep_task = asyncio.create_task(
                client.post("/separate", json={"base64": _wav_b64()})
            )
            await asyncio.sleep(0.1)  # ensure separation is in-flight (loop blocked?)
            start = time.perf_counter()
            health = await client.get("/v1/health/ready")
            elapsed = time.perf_counter() - start
            await sep_task
            return health.status_code, elapsed

    code, elapsed = asyncio.run(run())
    assert code == 200
    # health must return promptly even though a 0.6s separation is running
    assert elapsed < 0.3, f"event loop blocked: health took {elapsed:.2f}s"

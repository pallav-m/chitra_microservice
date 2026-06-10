import base64
import io
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _synthetic_wav_b64(seconds: float = 1.0, sr: int = 16000) -> str:
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    tone = np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _patch(monkeypatch):
    """Stub the heavy encoder + VAD so the router logic runs without models."""
    import app.routers.embedding as emb

    monkeypatch.setattr(emb, "trim_to_speech", lambda w: w)
    monkeypatch.setattr(emb, "get_speaker_embedding", lambda w: [0.1] * 192)


def test_rejects_both_inputs():
    resp = client.post(
        "/embed/speaker", json={"base64": "AA", "audioUri": "https://x/y.wav"}
    )
    assert resp.status_code == 422


def test_rejects_no_input():
    resp = client.post("/embed/speaker", json={})
    assert resp.status_code == 422


def test_returns_192_dim(monkeypatch):
    _patch(monkeypatch)
    resp = client.post("/embed/speaker", json={"base64": _synthetic_wav_b64()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 192
    assert len(body["embedding"]) == 192


def test_resamples_non_16k_input(monkeypatch):
    """Input at an arbitrary (non-16k) rate must still be accepted."""
    _patch(monkeypatch)
    resp = client.post(
        "/embed/speaker", json={"base64": _synthetic_wav_b64(sr=44100)}
    )
    assert resp.status_code == 200


def test_too_long_rejected(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr("app.routers.embedding.settings.max_audio_duration_sec", 0.1)
    resp = client.post(
        "/embed/speaker", json={"base64": _synthetic_wav_b64(seconds=1.0)}
    )
    assert resp.status_code == 413


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to run the real ECAPA + Silero VAD models",
)
def test_real_embedding_end_to_end():
    with TestClient(app) as live_client:  # `with` runs lifespan -> loads models
        resp = live_client.post(
            "/embed/speaker", json={"base64": _synthetic_wav_b64(seconds=3.0)}
        )
        assert resp.status_code == 200
        assert resp.json()["dim"] == 192

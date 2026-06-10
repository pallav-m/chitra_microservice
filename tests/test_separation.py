import base64
import io
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.models.loader import ModelRegistry


def _synthetic_wav_b64(seconds: float = 1.0, sr: int = 44100) -> str:
    """A short stereo sine clip encoded as base64 WAV."""
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    tone = np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([tone, tone * 0.5], axis=1)  # (frames, channels)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


# Plain TestClient (no `with`) does NOT run the lifespan, so the heavy Demucs
# model is never loaded for these fast tests.
client = TestClient(app)


def test_live_returns_200():
    resp = client.get("/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "live"


def test_ready_503_when_model_not_loaded():
    ModelRegistry._demucs = None
    resp = client.get("/v1/health/ready")
    assert resp.status_code == 503


def test_ready_200_when_all_models_loaded():
    # Readiness requires every co-served model — set sentinels for all three.
    ModelRegistry._demucs = object()
    ModelRegistry._speaker_encoder = object()
    ModelRegistry._vad = object()
    try:
        resp = client.get("/v1/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"
    finally:
        ModelRegistry._demucs = None
        ModelRegistry._speaker_encoder = None
        ModelRegistry._vad = None


def test_rejects_both_inputs():
    resp = client.post(
        "/separate", json={"base64": "AAAA", "audioUri": "https://x/y.wav"}
    )
    assert resp.status_code == 422


def test_rejects_no_input():
    resp = client.post("/separate", json={"responseFormat": "json"})
    assert resp.status_code == 422


class _FakeDemucs:
    samplerate = 44100


def _patch_separation(monkeypatch):
    import app.routers.separation as sep

    def fake_separate(waveform, sr):
        # Echo the input as both stems so we exercise encoding without a model.
        return {"vocals": waveform, "no_vocals": waveform * 0.5}

    monkeypatch.setattr(sep, "separate", fake_separate)
    monkeypatch.setattr(ModelRegistry, "get_demucs", classmethod(lambda cls: _FakeDemucs()))


def test_json_roundtrip(monkeypatch):
    _patch_separation(monkeypatch)
    resp = client.post(
        "/separate",
        json={"base64": _synthetic_wav_b64(), "responseFormat": "json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"vocals", "no_vocals", "sample_rate"}
    # Returned stems must be valid, decodable WAVs.
    for key in ("vocals", "no_vocals"):
        wav_bytes = base64.b64decode(body[key])
        data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
        assert data.shape[0] > 0


def test_wav_stream(monkeypatch):
    _patch_separation(monkeypatch)
    resp = client.post(
        "/separate",
        json={
            "base64": _synthetic_wav_b64(),
            "responseFormat": "wav",
            "stem": "no_vocals",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content), always_2d=True)
    assert data.shape[0] > 0


def _wav_bytes(seconds: float = 1.0, sr: int = 44100) -> bytes:
    return base64.b64decode(_synthetic_wav_b64(seconds, sr))


def test_upload_returns_zip_of_both_stems(monkeypatch):
    _patch_separation(monkeypatch)
    resp = client.post(
        "/separate/upload",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == ["vocals.wav", "bgm.wav"]
    for name in zf.namelist():
        data, sr = sf.read(io.BytesIO(zf.read(name)), always_2d=True)
        assert data.shape[0] > 0


def test_upload_oversize_rejected(monkeypatch):
    monkeypatch.setattr("app.routers.separation.settings.max_download_bytes", 100)
    resp = client.post(
        "/separate/upload",
        files={"file": ("song.wav", _wav_bytes(seconds=1.0), "audio/wav")},
    )
    assert resp.status_code == 413


def test_upload_bad_bytes_rejected():
    resp = client.post(
        "/separate/upload",
        files={"file": ("junk.wav", b"not audio at all", "audio/wav")},
    )
    assert resp.status_code == 400


def test_too_long_rejected(monkeypatch):
    _patch_separation(monkeypatch)
    monkeypatch.setattr(
        "app.routers.separation.settings.max_audio_duration_sec", 0.1
    )
    resp = client.post(
        "/separate",
        json={"base64": _synthetic_wav_b64(seconds=1.0), "responseFormat": "json"},
    )
    assert resp.status_code == 413


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to run the real Demucs model (downloads weights)",
)
def test_real_demucs_json_roundtrip():
    with TestClient(app) as live_client:  # `with` runs lifespan → loads model
        resp = live_client.post(
            "/separate",
            json={"base64": _synthetic_wav_b64(seconds=2.0), "responseFormat": "json"},
        )
        assert resp.status_code == 200
        assert set(resp.json()) == {"vocals", "no_vocals", "sample_rate"}

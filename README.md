# Chitra Audio Microservice

A GPU-served **FastAPI** backend that **co-serves multiple audio models in a single
process** behind one HTTP port. It is built to deploy as an
[NVIDIA Cloud Functions (NVCF)](https://docs.nvidia.com/cloud-functions/) container
function, but runs anywhere (CUDA, Apple Silicon/MPS, or CPU).

Today it exposes two capabilities:

| Capability | Endpoint | Model |
|---|---|---|
| Source separation (vocals / accompaniment) | `POST /separate` | HT-Demucs |
| Speaker embedding (192-dim voiceprint) | `POST /embed/speaker` | SpeechBrain ECAPA-TDNN (+ Silero VAD) |

Plus NVCF health probes: `GET /v1/health/live` and `GET /v1/health/ready`.

---

## Why it's built this way

The service is designed around the reality of **sharing one GPU across several
models** while keeping latency and VRAM predictable. The key decisions:

- **Single process, one model instance each.** All models are constructed exactly
  once, in a process-wide `ModelRegistry` singleton (`app/models/loader.py`), and
  **eager-loaded at startup** via the FastAPI lifespan handler. The first real
  request therefore pays no cold-start cost. Routers never instantiate a model —
  they fetch from the registry.
- **`--workers 1` is mandatory.** Every additional uvicorn worker would load its
  own copy of every model and multiply VRAM. Concurrency is handled by FastAPI's
  async stack, not by forking workers.
- **Flat VRAM regardless of clip length.** Demucs runs with `split=True,
  overlap=0.25`, which chunks long audio internally so memory use does not scale
  with duration.
- **Robust, FFmpeg-light audio I/O.** `torchaudio` 2.11 routes `save`/`load`
  through TorchCodec (which needs a system FFmpeg). To avoid that fragile
  dependency, all WAV encode/decode goes through **`soundfile`** (libsndfile,
  bundled in the wheel). `torchaudio` is still used for tensor resampling only.
- **Input sample rate is never assumed.** `soundfile` reports each file's actual
  rate; separation resamples to Demucs' native rate via `convert_audio`, and the
  embedding path resamples to 16 kHz. Send 8 k, 16 k, 44.1 k, 48 k — all fine.
- **Stems resolved by name, not index.** htdemucs returns sources in
  `drums, bass, other, vocals` order; we look up `"vocals"` by name (order varies
  by model variant) and sum the rest into `no_vocals`.
- **VAD-trimmed embeddings.** Before computing a speaker embedding, Silero VAD
  drops non-speech regions so the voiceprint isn't diluted by silence/music. If no
  speech is detected (very short/quiet clips), it falls back to the whole clip
  rather than failing.
- **Automatic device selection.** Leave `DEVICE` empty (or `auto`) and the service
  picks the best backend at startup: **cuda → mps → cpu**.
- **Concurrency without OOM.** Blocking work (inference, downloads, decode) runs off
  the event loop via `asyncio.to_thread`, so the server stays responsive (health
  probes never stall during a separation). GPU work is gated by **per-model hard
  caps** (`app/runtime/limiter.py`) so VRAM is bounded — see below.

---

## Concurrency & GPU allocation

The service handles concurrent requests **within a single process** (`--workers 1`;
more workers would duplicate model weights in one GPU's VRAM). Two mechanisms:

1. **Event-loop offload.** Each request's blocking steps run in a thread, so one slow
   separation never blocks health checks or other requests.
2. **Per-model concurrency caps.** A `ModelGate` holds one semaphore per model, sized
   by config. Only the GPU compute is gated (downloads aren't — they don't hold a
   slot while waiting on the network). This puts a **hard ceiling on each model's
   concurrent activations**, so worst-case VRAM is bounded and OOM is structurally
   prevented. Size the caps so the worst-case simultaneous mix fits your GPU.

When all slots for a model are busy, a request **waits up to `GPU_ACQUIRE_TIMEOUT_SEC`**
for one to free; if it doesn't, the service returns **`503` with a `Retry-After`
header** (NVCF can autoscale on the 503 rate). For real horizontal scale, run multiple
NVCF instances — each with its own GPU — rather than raising worker count.

> Note: concurrency >1 mainly buys a responsive server, overlap of I/O with compute,
> and bounded VRAM. It is not an Nx GPU speedup — CUDA kernels on the default stream
> largely serialize. Use `scripts/measure_vram.py` on the target GPU to set the caps
> from real `torch.cuda.max_memory_allocated()` numbers.

---

## Models

| Model | Role | Notes | ~VRAM |
|---|---|---|---|
| **HT-Demucs** (`htdemucs_ft` by default) | 4-stem source separation, returned as vocals + non-vocals | `htdemucs_ft` is the fine-tuned variant (higher quality, ~4× slower than `htdemucs`) | ~3–4 GB |
| **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) | 192-dim speaker embedding | Raw (un-normalized) output; normalize client-side for cosine similarity | ~0.5–1 GB |
| **Silero VAD** | Speech-region detection before embedding | Tiny, runs on CPU regardless of `DEVICE` | negligible |
| **Total** | | | **~4–5 GB** |

This leaves substantial headroom on a 16/24 GB GPU for other models running
alongside.

---

## API

All endpoints accept JSON. Both inference endpoints take the **same audio input
shape** — provide **exactly one** of:

| Field | Type | Description |
|---|---|---|
| `base64` | string | Inline base64-encoded audio bytes (WAV/FLAC/OGG; MP3 with a recent libsndfile) |
| `audioUri` | string | A public HTTP(S) URL the service downloads (size/timeout capped) |

Supplying both, or neither, returns **422**. Audio longer than
`MAX_AUDIO_DURATION_SEC` returns **413**. Undecodable input or a failed download
returns **400**.

### Health — `GET /v1/health/live` · `GET /v1/health/ready`

`live` always returns 200 once the process is up. `ready` returns **200** only
after all three models are loaded, otherwise **503** (`{"status": "loading"}`).
NVCF uses `ready` as the readiness probe; both are served on the inference port.

```bash
curl -s localhost:8000/v1/health/ready
# {"status":"ready"}
```

### Separation — `POST /separate`

Splits audio into a **vocals** stem and a **no_vocals** (accompaniment =
drums + bass + other) stem.

Request fields (in addition to `base64`/`audioUri`):

| Field | Type | Default | Description |
|---|---|---|---|
| `responseFormat` | `"json"` \| `"wav"` | `"json"` | `json` returns both stems base64-encoded; `wav` streams a single stem |
| `stem` | `"vocals"` \| `"no_vocals"` | `"vocals"` | Which stem to stream (only used when `responseFormat="wav"`) |

**Example 1 — base64 in, both stems as JSON:**

```bash
# Encode a local file to base64 and POST it.
B64=$(base64 -i song.wav)
curl -s -X POST localhost:8000/separate \
  -H 'content-type: application/json' \
  -d "{\"base64\":\"$B64\",\"responseFormat\":\"json\"}"
```

Response:

```json
{
  "vocals": "UklGRiQ...base64 WAV...",
  "no_vocals": "UklGRiQ...base64 WAV...",
  "sample_rate": 44100
}
```

Decode a stem back to a file:

```bash
# (continuing from the response above, with `jq`)
curl -s -X POST localhost:8000/separate -H 'content-type: application/json' \
  -d "{\"base64\":\"$B64\"}" | jq -r .vocals | base64 -d > vocals.wav
```

**Example 2 — remote URL in, stream the accompaniment as a WAV:**

```bash
curl -s -X POST localhost:8000/separate \
  -H 'content-type: application/json' \
  -d '{"audioUri":"https://example.com/song.wav","responseFormat":"wav","stem":"no_vocals"}' \
  --output accompaniment.wav
```

**Python client:**

```python
import base64, httpx

audio_b64 = base64.b64encode(open("song.wav", "rb").read()).decode()
resp = httpx.post(
    "http://localhost:8000/separate",
    json={"base64": audio_b64, "responseFormat": "json"},
    timeout=120,
)
data = resp.json()
open("vocals.wav", "wb").write(base64.b64decode(data["vocals"]))
print("output sample rate:", data["sample_rate"])
```

### Speaker embedding — `POST /embed/speaker`

Returns a 192-dimensional ECAPA-TDNN voiceprint. The input is downmixed to mono,
resampled to 16 kHz, trimmed to speech with VAD, then embedded.

**Example — remote URL in:**

```bash
curl -s -X POST localhost:8000/embed/speaker \
  -H 'content-type: application/json' \
  -d '{"audioUri":"https://example.com/voice.wav"}'
```

Response:

```json
{
  "embedding": [0.0123, -0.0456, 0.0789, "... 192 floats ..."],
  "dim": 192
}
```

**Python — embed two clips and compare speakers (cosine similarity):**

```python
import base64, httpx, numpy as np

def embed(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    r = httpx.post("http://localhost:8000/embed/speaker",
                   json={"base64": b64}, timeout=60)
    return np.array(r.json()["embedding"])

a, b = embed("alice_1.wav"), embed("alice_2.wav")
# Embeddings are raw — normalize before taking a dot product.
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"cosine similarity: {cos:.3f}")   # ~>0.7 typically means same speaker
```

---

## Configuration

All settings are environment variables (case-insensitive), read from `.env` or the
process environment. See `.env.example`. Defaults shown:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | _(empty → auto)_ | `cuda` / `mps` / `cpu`, or empty/`auto` to detect (cuda → mps → cpu) |
| `DEMUCS_MODEL` | `htdemucs_ft` | Demucs pretrained model name |
| `MAX_AUDIO_DURATION_SEC` | `60` | Reject longer inputs with 413 |
| `OUTPUT_SAMPLE_RATE` | `0` | `0` = keep Demucs-native 44.1 kHz stereo; non-zero = downmix mono + resample stems (e.g. `16000` for ASR) |
| `DOWNLOAD_TIMEOUT_SEC` | `30` | `audioUri` download timeout |
| `MAX_DOWNLOAD_BYTES` | `104857600` | `audioUri` download size cap (100 MB) |
| `SPEAKER_MODEL_SOURCE` | `speechbrain/spkrec-ecapa-voxceleb` | ECAPA model source |
| `SPEAKER_SAMPLE_RATE` | `16000` | Rate inputs are resampled to for embedding |
| `VAD_THRESHOLD` | `0.5` | Silero speech-detection threshold (higher = stricter) |
| `DEMUCS_MAX_CONCURRENCY` | `4` | Max concurrent separations (hard VRAM cap) |
| `SPEAKER_MAX_CONCURRENCY` | `8` | Max concurrent embeddings (hard VRAM cap) |
| `GPU_MAX_CONCURRENCY` | `0` | Optional aggregate ceiling across all models (0 = off) |
| `GPU_ACQUIRE_TIMEOUT_SEC` | `30` | Wait for a free GPU slot before returning 503 |

---

## Running

### Local (uv)

```bash
uv sync
# Auto-detects the device; on a Mac this uses MPS, on a CUDA box it uses the GPU.
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Tests

```bash
uv run pytest -q                                   # fast suite (models are mocked)
uv run pytest tests/test_embedding.py::test_returns_192_dim   # a single test
RUN_MODEL_TESTS=1 DEVICE=cpu uv run pytest -q      # include real-model end-to-end tests
```

The fast suite mocks the models, so it runs in ~1s with no weights downloaded. The
real-model tests are gated behind `RUN_MODEL_TESTS=1` (they download weights on
first run).

### Docker / NVCF

```bash
docker build --platform linux/amd64 -t chitra-microservice .
docker run --rm --gpus all -p 8000:8000 chitra-microservice
```

The image is built for **NVCF**:

- Pinned `linux/amd64` (PyTorch CUDA wheels and NVCF backends are x86_64-only).
- Runs as a **non-root** user (an NVCF requirement).
- **Model weights are baked in at build time** so cold starts stay offline.
- Health and inference share port **8000** (the NVCF `inferencePort`); the
  container's `HEALTHCHECK` and NVCF's readiness probe both hit
  `/v1/health/ready`.
- `DEVICE=cuda` is set explicitly in the image so a GPU container never silently
  falls back to CPU.

---

## Roadmap

This service is an evolving audio-processing backend. Planned directions:

- **`sam-audio` integration (in development).** We expect to **replace and/or
  augment the current Demucs + ECAPA stack with `sam-audio`** as it matures. The
  architecture is built to make this a low-friction swap: models live behind the
  `ModelRegistry` and routers depend only on small inference functions
  (`app/models/*.py`), so a new backend can be introduced — or an existing one
  retired — without touching the HTTP layer or the input/output contracts.
- **Additional models / endpoints**, co-served on the same GPU within the VRAM
  budget — e.g. ASR (transcription), language ID, diarization, and audio
  classification.
- **VAD as a first-class endpoint** (it is currently an internal pre-filter for
  embedding).
- **NVCF asset inputs.** `audioUri` currently resolves public HTTP(S) URLs; the
  download path (`app/utils/audio_io.py`) is isolated so an NVCF asset-ID resolver
  can be added without changing the request schema.

### Extending the service

To add a model: add a lazy getter to `ModelRegistry`, eager-load it in the
`lifespan` handler (`app/main.py`), add an inference function under
`app/models/`, and expose a router under `app/routers/`. Reuse the shared
`AudioInput` schema and `app/utils/audio_io.py` helpers for input handling so the
base64/`audioUri` contract stays consistent across endpoints.

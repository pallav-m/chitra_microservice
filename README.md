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

## Quick start

```bash
# 1. Install dependencies (uv provisions Python 3.11 + the virtualenv)
uv sync

# 2. Run the service — device is auto-detected (cuda > mps > cpu)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Verify it's up (in another shell)
curl -s localhost:8000/v1/health/ready        # -> {"status":"ready"}
```

The **first** start downloads model weights (Demucs, ECAPA, Silero) and can take a
minute; later starts are fast. On a non-GPU host prepend `DEVICE=cpu`. Always keep
`--workers 1` (more workers duplicate the models in VRAM).

Run it in a container instead (GPU host):

```bash
docker build --platform linux/amd64 -t chitra-microservice .
docker run --rm --gpus all -p 8000:8000 chitra-microservice
```

Then send a request:

```bash
curl -s -X POST localhost:8000/separate \
  -H 'content-type: application/json' \
  -d "{\"base64\":\"$(base64 -i song.wav)\",\"responseFormat\":\"json\"}" | jq keys
```

Full details below: [Running](#running) · [Building & deployment](#building--deployment) · [API](#api).

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

**Choosing an input encoding:** the JSON endpoints (`/separate`, `/embed/speaker`)
take **exactly one** of `base64` or `audioUri`; the upload endpoint
(`/separate/upload`) takes a multipart `file`. Pick by payload/client: base64-JSON for
small clips from JSON clients, `audioUri` when the audio is already hosted, multipart
upload for binary blobs from a browser/app.

| Field | Type | Description |
|---|---|---|
| `base64` | string | Inline base64-encoded audio bytes (WAV/FLAC/OGG; MP3 with a recent libsndfile) |
| `audioUri` | string | A public HTTP(S) URL the service downloads (size/timeout capped) |
| `file` (multipart) | blob | Binary audio upload — `/separate/upload` only |

For the JSON endpoints, supplying both `base64` and `audioUri`, or neither, returns
**422**. Audio longer than `MAX_AUDIO_DURATION_SEC` returns **413** (as does an upload
over the size cap). Undecodable input or a failed download returns **400**.

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

### Separation (file upload) — `POST /separate/upload`

Same separation, but takes a **multipart file upload** (a blob) instead of JSON, and
streams back a single **`application/zip`** containing `vocals.wav` + `bgm.wav`. Use this
for binary uploads from a browser form or app; use `POST /separate` when the audio is
already base64/JSON or hosted at a URL.

Form field: `file` — the audio blob (WAV/FLAC/OGG; MP3 with a recent libsndfile).

```bash
curl -s -X POST localhost:8000/separate/upload \
  -F file=@song.wav \
  --output stems.zip
unzip -o stems.zip          # -> vocals.wav, bgm.wav
```

**Python client:**

```python
import httpx, zipfile, io

with open("song.wav", "rb") as f:
    resp = httpx.post(
        "http://localhost:8000/separate/upload",
        files={"file": ("song.wav", f, "audio/wav")},
        timeout=120,
    )
zf = zipfile.ZipFile(io.BytesIO(resp.content))
zf.extractall("stems/")     # writes stems/vocals.wav and stems/bgm.wav
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

---

## Building & deployment

### 1. Build the image

```bash
docker build --platform linux/amd64 -t chitra-microservice .
```

The image is purpose-built for **NVCF**:

- Pinned `linux/amd64` — PyTorch CUDA wheels and NVCF backends are x86_64-only.
  (On Apple Silicon this builds under emulation; build on an x86_64 box for speed.)
- Runs as a **non-root** user (an NVCF requirement).
- **Model weights are baked in at build time** so cold starts stay offline.
- Health and inference share port **8000** (the NVCF `inferencePort`); the
  container `HEALTHCHECK` and NVCF's readiness probe both hit `/v1/health/ready`.
- `DEVICE=cuda` is set explicitly so a GPU container never silently falls to CPU.

### 2. Run it locally (smoke test before pushing)

```bash
docker run --rm --gpus all -p 8000:8000 chitra-microservice
# wait for readiness, then:
curl -s localhost:8000/v1/health/ready        # -> {"status":"ready"}
```

### 3. Push to the NGC private registry

NVCF pulls images from NGC (`nvcr.io`). Authenticate once with your NGC API key,
then tag and push:

```bash
# docker login: username is the literal '$oauthtoken', password is your NGC API key
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

ORG=your-ngc-org                 # your NGC org (and /team if applicable)
TAG=nvcr.io/$ORG/chitra-microservice:0.1.0

docker tag chitra-microservice "$TAG"
docker push "$TAG"
```

### 4. Create & deploy the NVCF function

Using the [NGC CLI](https://docs.ngc.nvidia.com/cli/) (`ngc`). Flag names vary
slightly across CLI versions — confirm with `ngc cf function create --help`; the
**values** below are what matter:

```bash
# Create a container-based function. Health + inference are on the same port.
ngc cf function create \
  --name chitra-microservice \
  --container-image "$TAG" \
  --container-port 8000 \
  --inference-url /separate \
  --health-uri /v1/health/ready \
  --health-expected-status-code 200 \
  --api-body-format CUSTOM            # we serve plain JSON, not the OpenAI schema
# -> prints a FUNCTION_ID and VERSION_ID

# Deploy that version onto a GPU backend with autoscaling.
ngc cf function deploy create \
  --function-id   <FUNCTION_ID> \
  --function-version-id <VERSION_ID> \
  --deployment-specification "gpu:L40S:1:<BACKEND>:<INSTANCE_TYPE>:min1:max3"
#                             GPU model ^   count ^         min/max instances ^
```

What to get right for this service:

- **`--health-uri /v1/health/ready`** — the single most common deploy failure is a
  stale health path; if NVCF probes the wrong URL it gets 404 and recycles the
  container forever. This must match the router prefix in `app/routers/health.py`.
- **`--container-port 8000`** — health and inference share it (the `inferencePort`).
- **GPU & instance count** — size for a 40 GB GPU; the per-model concurrency caps
  (`DEMUCS_MAX_CONCURRENCY`, etc.) bound VRAM per instance. Scale throughput with
  `max` instances (each gets its own GPU), **not** by raising `--workers`.
- **Env overrides** — pass `DEMUCS_MAX_CONCURRENCY`, `GPU_ACQUIRE_TIMEOUT_SEC`,
  `MAX_AUDIO_DURATION_SEC`, etc. as function environment variables to tune without
  rebuilding the image.

Once `Active`, invoke through the NVCF endpoint:

```bash
curl -s https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/<FUNCTION_ID> \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"audioUri":"https://example.com/song.wav","responseFormat":"json"}'
```

> The CLI is one path; the same function/deployment can be created from the NVCF
> web console or REST API. Whichever you use, the health URI, port, and API body
> format are the fields that must match this service.

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

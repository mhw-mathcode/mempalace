# MemPalace adapter for Agent Memory Leaderboard

This directory contains a self-hosted, unauthenticated Add/Search service for the
Agent Memory Leaderboard (AML) commercial track. It stores original evidence and
returns ranked memories; it does not call an LLM, generate answers, caption images,
or rerank with an LLM.

The adapter deliberately uses MemPalace's original embedding choices:

- `textual` and `coding`: `minilm` (`all-MiniLM-L6-v2`)
- `multimodal`: `embeddinggemma2`, configured for `text+vision`

Each deployment serves exactly one track so incompatible vector spaces cannot mix.

## Install

From the repository root, install the textual/coding service and development tools:

```powershell
uv sync --project aml_test --group dev
```

For multimodal text-and-image inference, include the optional runtime:

```powershell
uv sync --project aml_test --group dev --extra multimodal
```

Models are not bundled in this adapter. Their first inference downloads the weights
to the model provider's normal cache. MiniLM is roughly 300 MB. EmbeddingGemma 2 is
substantially larger and also needs the PyTorch/Transformers runtime; see the root
project's `docs/embeddinggemma2.md` for device and installation guidance. Pre-warm
each deployment before leaderboard traffic by running the opt-in model tests once
with network access. After both tests pass from cache, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` can prevent Hugging Face network access.

## Configure and run

Copy `.env.example` values into the process environment. `AML_TRACK` must be one of
`textual`, `coding`, or `multimodal`. `AML_DATA_DIR` must be an absolute, dedicated
directory. It will contain `state.sqlite3`, `vectors/`, and `media/`.

Example for PowerShell:

```powershell
$env:AML_TRACK = 'textual'
$env:AML_DATA_DIR = 'E:\mempalace-aml-data'
uv run --project aml_test python -m aml_adapter
```

For multimodal:

```powershell
$env:AML_TRACK = 'multimodal'
$env:AML_DATA_DIR = 'E:\mempalace-aml-multimodal-data'
$env:MEMPALACE_EMBEDDINGGEMMA2_MODALITIES = 'text+vision'
uv run --project aml_test python -m aml_adapter
```

The service intentionally has no authentication. Do not expose it directly to the
public Internet. Put it behind network-level access controls or a private ingress,
and never share one deployment's data directory with another instance or track.

AML permits a request-processing ceiling of 30 minutes. Keep any proxy and load
balancer timeout at or above 1,800 seconds. Embedding inference and local Chroma
writes are blocking work dispatched to a thread pool; begin with one worker and low
concurrency, then load-test on the target hardware before raising either. Add is
synchronous: a 200 response means the memory is durable and immediately searchable.

## HTTP smoke test

Health:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Add a memory:

```powershell
curl.exe -X POST http://127.0.0.1:8000/add `
  -H "Content-Type: application/json" `
  -d '{"request_id":"demo-1","messages":[{"role":"user","content":"The release codename is amber-orchid."}],"user_id":"demo-user","session_id":"demo-session"}'
```

Search immediately:

```powershell
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"release codename","user_id":"demo-user","top_k":5}'
```

Multimodal requests use ordered content parts. Images must be inline Base64 Data
URIs with MIME type `image/jpeg`, `image/png`, or `image/webp`; remote image URLs
are rejected. The decoded limits are 10 MiB per image, 30 MiB per Add, and 30 MiB
per Search response.

## Verify

The normal suite never downloads model weights:

```powershell
uv run --project aml_test pytest -q -m "not real_models"
uv run --project aml_test ruff check aml_adapter tests
uv run --project aml_test python -c "from aml_adapter.app import app; assert app.title"
```

Once both runtimes and model weights are available, run the opt-in checks:

```powershell
$env:AML_RUN_REAL_MODELS = '1'
uv run --project aml_test pytest tests/test_real_models.py -q -m real_models
```

These tests exercise MiniLM text and EmbeddingGemma 2 text-plus-PNG paths and assert
that MemPalace's LLM modules are never imported.

## Backup and deletion

Stop the service before taking a filesystem backup so SQLite and Chroma are copied
as one consistent unit. Back up the entire dedicated `AML_DATA_DIR`, including the
SQLite WAL/SHM files if present, `vectors/`, and `media/`; restore the entire unit to
the same path layout.

After evaluation, stop the process and delete only the explicitly configured,
dedicated `AML_DATA_DIR`. Verify the resolved path before deletion. Complete this
deletion within 30 days of the evaluation, and retain no copied backups beyond the
same deadline.

AML protocol details: [documentation](https://agentmemoryleaderboard.ai/zh-cn/docs)
and [API guide](https://agentmemoryleaderboard.ai/api-guide).

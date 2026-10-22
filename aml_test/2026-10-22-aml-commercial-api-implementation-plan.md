# AML Commercial API Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an unauthenticated, self-hosted AML commercial-track Add/Search API that uses MemPalace's original MiniLM or EmbeddingGemma 2 embedding path and supports ordered Base64 image content.

**Architecture:** A FastAPI transport delegates to a track-aware service. SQLite stores canonical requests, original content, write state, and media references; a dedicated Chroma collection stores deterministic vector children, and MemPalace's existing embedding factory supplies every vector. Add uses a resumable pending-to-committed state machine, while Search filters by exact `user_id`, collapses vector children to logical memories, and hydrates response content from SQLite.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLite, ChromaDB, NumPy, MemPalace embedding functions, pytest, HTTPX.

**Spec:** `aml_test/2026-10-22-aml-commercial-api-design.md`

## Global Constraints

- All new implementation, tests, dependency metadata, examples, and documentation stay under `aml_test/`; files under `mempalace/` are imported but not modified.
- `AML_TRACK` is exactly one of `textual`, `coding`, or `multimodal`; textual/coding use `model="minilm"`, and multimodal uses `model="embeddinggemma2"` with `MEMPALACE_EMBEDDINGGEMMA2_MODALITIES=text+vision`.
- No LLM, captioning, answer generation, reranking, authentication, or remote-image fetch is permitted.
- `user_id` is the sole retrieval-isolation key; a successful Add must be durable and immediately searchable.
- Images accept only inline Base64 JPEG, PNG, or WebP, with 10 MiB decoded per image, 30 MiB decoded per Add, and 30 MiB decoded per Search response.
- A deployment uses a dedicated `AML_DATA_DIR` with `state.sqlite3`, `vectors/`, and `media/`; vector spaces from different tracks are never mixed.
- API logs must not contain queries, options, messages, returned content, Data URIs, decoded media, environment values, or secrets.
- Search returns evidence only in `{"data": [...]}` rank order, never a generated answer, and never more than `top_k` (1 through 100).

## Review Focus

- A syntactically valid but corrupt or MIME-mismatched image must fail before SQLite, blob, or vector writes; Task 2 pins this with corrupt-image and signature-mismatch tests.
- Two concurrent Adds with the same `request_id` must produce one logical memory and deterministic vectors; Task 3 and Task 5 pin this with a barrier-based concurrency test.
- NaN, infinite, zero-length, wrong-count, or wrong-dimension embedding output must fail closed and never commit; Task 4 pins each malformed-output class.
- Chroma may return duplicate children or deleted/pending IDs; Task 6 pins deduplication, committed-only hydration, and stable tie ordering.
- An oversized high-ranked multimodal result must stop the ranked prefix rather than be skipped in favor of lower-ranked items; Task 6 pins this exact response-budget behavior.

---

## File map

- `aml_test/pyproject.toml`: isolated adapter dependencies, local editable MemPalace source, pytest and Ruff configuration.
- `aml_test/aml_adapter/config.py`: environment parsing and immutable runtime settings.
- `aml_test/aml_adapter/schemas.py`: AML request/response models and track validation.
- `aml_test/aml_adapter/media.py`: strict Data URI validation, image decoding, content normalization, and atomic blob storage.
- `aml_test/aml_adapter/state_store.py`: SQLite schema, idempotency records, pending/committed transitions, and hydration.
- `aml_test/aml_adapter/embeddings.py`: track-aware wrapper around `mempalace.embedding.get_embedding_function`.
- `aml_test/aml_adapter/vector_store.py`: dedicated Chroma collection and cosine candidate retrieval.
- `aml_test/aml_adapter/service.py`: Add/Search orchestration and recovery state machine.
- `aml_test/aml_adapter/app.py`: FastAPI app factory, routes, exception mapping, and privacy-safe request logging.
- `aml_test/aml_adapter/__main__.py`: Uvicorn entry point.
- `aml_test/tests/`: unit, transport, concurrency, and opt-in real-model integration tests.
- `aml_test/README.md` and `aml_test/.env.example`: install, run, smoke-test, capacity, model-cache, and deletion instructions.

### Task 1: Project scaffold, configuration, and wire schemas

**Files:**
- Create: `aml_test/pyproject.toml`
- Create: `aml_test/aml_adapter/__init__.py`
- Create: `aml_test/aml_adapter/config.py`
- Create: `aml_test/aml_adapter/schemas.py`
- Create: `aml_test/tests/test_config.py`
- Create: `aml_test/tests/test_schemas.py`

**Interfaces:**
- Produces: `Track(StrEnum)`, `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings`, `ContentPart`, `Message`, `AddRequest`, `AddResponse`, `SearchRequest`, `SearchItem`, and `SearchResponse`.
- Produces: `validate_add_for_track(request: AddRequest, track: Track) -> None` and `validate_search_for_track(request: SearchRequest, track: Track) -> None`.

- [ ] **Step 1: Write failing configuration tests**

  Test `Settings.from_env()` defaults host to `0.0.0.0`, port to `8000`, and routes to `/health`, `/add`, and `/search`; rejects missing/unknown `AML_TRACK` and missing/relative `AML_DATA_DIR`; and exposes `state.sqlite3`, `vectors`, and `media` paths beneath the absolute data root. Pin optional `AML_HEALTH_PATH`, `AML_ADD_PATH`, and `AML_SEARCH_PATH` overrides to absolute, distinct route paths.

- [ ] **Step 2: Run configuration tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_config.py -q`
  Expected: FAIL because `aml_adapter.config` does not exist.

- [ ] **Step 3: Implement dependency metadata and `Settings`**

  Define a Python 3.10+ project with `fastapi>=0.115,<1`, `uvicorn>=0.30,<1`, local editable `mempalace`, optional `multimodal = ["mempalace[embeddinggemma2]"]`, and dev dependencies `httpx`, `pillow>=12.3.0`, `pytest`, and `ruff`; implement frozen settings with `track`, `data_dir`, `host`, `port`, route paths, and derived storage paths.

- [ ] **Step 4: Run configuration tests**

  Run: `uv run --project aml_test pytest tests/test_config.py -q`
  Expected: PASS.

- [ ] **Step 5: Write failing schema tests**

  Pin exact Add/Search success shapes, non-empty identifiers/content/options, roles `user|assistant`, integer Unix-millisecond timestamps, `top_k` 1..100, forbidden unknown fields, string-only textual/coding content, and string-or-ordered-parts multimodal content/query.

- [ ] **Step 6: Implement schema models and track validators**

  Use discriminated `TextPart(type="text", text: str)` and `ImagePart(type="image_url", image_url: ImageURL)` models; use strict Pydantic models with `extra="forbid"`; raise a track-validation `ValueError` that the transport can map to HTTP 400.

- [ ] **Step 7: Run Task 1 tests and lint**

  Run: `uv run --project aml_test pytest tests/test_config.py tests/test_schemas.py -q`
  Expected: PASS.

  Run: `uv run --project aml_test ruff check aml_adapter tests/test_config.py tests/test_schemas.py`
  Expected: no errors.

- [ ] **Step 8: Commit**

  Run: `git add aml_test && git commit -m "feat: define AML adapter configuration and schemas"`

### Task 2: Multimodal validation and blob storage

**Files:**
- Create: `aml_test/aml_adapter/media.py`
- Create: `aml_test/tests/test_media.py`

**Interfaces:**
- Consumes: `ContentPart`, `Message`, and `Track` from Task 1.
- Produces: `DecodedImage(mime_type: str, data_uri: str, data: bytes, sha256: str, suffix: str, byte_size: int)` and `MediaRef(sha256: str, mime_type: str, suffix: str, byte_size: int)`.
- Produces: `decode_image_data_uri(url: str) -> DecodedImage`, `validate_request_media(messages_or_parts: Sequence[Any], limit_bytes: int = 30 * 1024 * 1024) -> list[DecodedImage]`, and `MediaStore.put(image: DecodedImage) -> Path`.
- Produces: `search_content(messages: Sequence[Message], track: Track) -> str | list[ContentPart]`; textual/coding join source strings with `"\n\n"`, while multimodal flattens message content in source order and converts string content to a text part without altering existing parts.

- [ ] **Step 1: Write failing Data URI validation tests**

  Cover strict Base64, rejected remote URLs, allowed MIME types, decoded 10 MiB boundary, 30 MiB aggregate boundary, corrupt image bytes, MIME/signature mismatch, and Pillow decompression-bomb rejection.

- [ ] **Step 2: Run the focused tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_media.py -q`
  Expected: FAIL because `aml_adapter.media` does not exist.

- [ ] **Step 3: Implement strict decoding and image verification**

  `decode_image_data_uri` must use `base64.b64decode(..., validate=True)`, byte-count limits before image load, Pillow `Image.verify()`, decoded-format agreement with the declared MIME, and no network/file URL handling.

- [ ] **Step 4: Write failing atomic-storage and normalization tests**

  Assert SHA-256 filenames, idempotent identical writes, no temporary-file residue after success/failure, no path components derived from input, exact one-message round-trip, deterministic multi-message textual joining, and multimodal part-order preservation.

- [ ] **Step 5: Implement `MediaStore` and `search_content`**

  Persist with a same-directory temporary file, flush plus `os.fsync`, then `os.replace`; return an existing digest path without rewriting it.

- [ ] **Step 6: Run Task 2 tests**

  Run: `uv run --project aml_test pytest tests/test_media.py -q`
  Expected: PASS.

- [ ] **Step 7: Commit**

  Run: `git add aml_test/aml_adapter/media.py aml_test/tests/test_media.py && git commit -m "feat: validate and persist AML image content"`

### Task 3: SQLite state store and idempotency

**Files:**
- Create: `aml_test/aml_adapter/state_store.py`
- Create: `aml_test/tests/test_state_store.py`

**Interfaces:**
- Consumes: validated `AddRequest` and normalized response content from Tasks 1-2.
- Produces: `canonical_payload_digest(request: AddRequest) -> str`.
- Produces: `MemoryRecord(id: str, request_id: str, user_id: str, session_id: str, content: Any, created_at: str, state: str, child_ids: tuple[str, ...], media_refs: tuple[MediaRef, ...])`.
- Produces: `StateStore.begin_add(request: AddRequest, content: Any, child_ids: Sequence[str], media_refs: Sequence[MediaRef]) -> tuple[MemoryRecord, Literal["created", "pending", "committed"]]`, `mark_committed(request_id: str) -> None`, `get_by_request_id(request_id: str) -> MemoryRecord | None`, and `hydrate_committed(ids: Sequence[str], user_id: str) -> dict[str, MemoryRecord]`.
- Produces: `IdempotencyConflict` when one `request_id` is reused with a different canonical digest.

- [ ] **Step 1: Write failing schema and round-trip tests**

  Assert WAL mode, foreign keys, uniqueness of `request_id` and child IDs, stable SHA-256-derived memory/child IDs, canonical digest sensitivity to every field and array order, JSON round-trip of original messages/Data URIs/source timestamps, persisted media digest/MIME/decoded-byte references, and UTC `created_at` ending in `Z`.

- [ ] **Step 2: Run tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_state_store.py -q`
  Expected: FAIL because `aml_adapter.state_store` does not exist.

- [ ] **Step 3: Implement migrations, canonicalization, and transactional methods**

  Open one SQLite connection per operation with a bounded busy timeout; use `BEGIN IMMEDIATE` for `begin_add`; serialize canonical JSON as UTF-8 with sorted object keys and preserved array order.

- [ ] **Step 4: Write failing retry, isolation, and concurrency tests**

  Assert same-payload retry returns the same record/state, changed payload raises `IdempotencyConflict`, pending survives process reopen, hydration excludes pending rows and wrong `user_id`, and barrier-synchronized identical Adds create one row without `OperationalError` leakage.

- [ ] **Step 5: Complete idempotency and hydration behavior**

  Ensure committed records are immutable, pending records are resumable, and all hydration SQL includes exact `user_id = ?` plus `state = 'committed'`.

- [ ] **Step 6: Run Task 3 tests**

  Run: `uv run --project aml_test pytest tests/test_state_store.py -q`
  Expected: PASS.

- [ ] **Step 7: Commit**

  Run: `git add aml_test/aml_adapter/state_store.py aml_test/tests/test_state_store.py && git commit -m "feat: add durable AML request state store"`

### Task 4: MemPalace embedding adapter and vector store

**Files:**
- Create: `aml_test/aml_adapter/embeddings.py`
- Create: `aml_test/aml_adapter/vector_store.py`
- Create: `aml_test/tests/test_embeddings.py`
- Create: `aml_test/tests/test_vector_store.py`

**Interfaces:**
- Consumes: `Track`, local media paths, logical memory IDs, child IDs, and exact `user_id`.
- Produces: `VectorChild(id: str, memory_id: str, kind: Literal["text", "image"], value: str)` and `VectorCandidate(memory_id: str, child_id: str, score: float)`.
- Produces: `EmbeddingAdapter.for_track(track: Track) -> EmbeddingAdapter`, `embed_documents(children: Sequence[VectorChild]) -> list[list[float]]`, and `embed_query(query: str | Sequence[ContentPart], options: Sequence[str] | None) -> list[list[float]]`.
- Produces: `VectorStore.upsert(children: Sequence[VectorChild], vectors: Sequence[Sequence[float]], user_id: str) -> None`, `query(query_vectors: Sequence[Sequence[float]], user_id: str, limit: int) -> list[VectorCandidate]`, and `contains_all(child_ids: Sequence[str]) -> bool`.

- [ ] **Step 1: Write failing embedding routing tests**

  Assert textual/coding call `get_embedding_function(model="minilm")`; multimodal calls `get_embedding_function(model="embeddinggemma2")`; documents use `embed_documents`, text queries use `embed_query`, images use `embed_image`, coding does not silently switch models, and ordered options render exactly as `query + "\n\nOptions:\n" + "\n".join(options)`.

- [ ] **Step 2: Run embedding tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_embeddings.py -q`
  Expected: FAIL because `aml_adapter.embeddings` does not exist.

- [ ] **Step 3: Implement `EmbeddingAdapter` and output validation**

  Normalize all outputs to finite, non-zero, equal-width `float32` vectors; require one vector per input; multimodal text-only and image inputs must remain in the same configured EmbeddingGemma 2 space.

- [ ] **Step 4: Add malformed-output tests**

  Pin failures for NaN, infinity, zero vector, inconsistent width, wrong row count, and empty query-vector output.

- [ ] **Step 5: Write failing Chroma vector-store tests**

  Use deterministic fake vectors to assert cosine scores where higher is better, exact `user_id` filtering, idempotent upsert, collection metadata `hnsw:space=cosine`, separate collection name per track, duplicate-child preservation for later collapse, and `contains_all` visibility.

- [ ] **Step 6: Implement `VectorStore`**

  Use `chromadb.PersistentClient(path=settings.vectors_dir)` and explicit embeddings; store only child ID, memory ID, user ID, track, kind, and timestamps; query each query vector with `where={"user_id": user_id}`, then return candidates sorted by score descending and IDs ascending.

- [ ] **Step 7: Run Task 4 tests**

  Run: `uv run --project aml_test pytest tests/test_embeddings.py tests/test_vector_store.py -q`
  Expected: PASS.

- [ ] **Step 8: Commit**

  Run: `git add aml_test/aml_adapter/embeddings.py aml_test/aml_adapter/vector_store.py aml_test/tests/test_embeddings.py aml_test/tests/test_vector_store.py && git commit -m "feat: connect MemPalace embeddings to AML vectors"`

### Task 5: Synchronous Add service and recovery

**Files:**
- Create: `aml_test/aml_adapter/service.py`
- Create: `aml_test/tests/test_add_service.py`

**Interfaces:**
- Consumes: Tasks 1-4 stores, media, embeddings, and schemas.
- Produces: `MemoryService.add(request: AddRequest) -> AddResponse` and `MemoryService.recover_pending(request_id: str) -> AddResponse`.
- Produces: deterministic vector children: one role-delimited combined text child per logical Add when text exists, plus one child per image in source order.

- [ ] **Step 1: Write failing Add happy-path tests**

  Assert exact identifier echo, full validation before writes, pending record before media/vector work, atomic media writes after pending creation, deterministic children, commit only after every upsert, read-after-write via `contains_all`, and immediate committed visibility.

- [ ] **Step 2: Run tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_add_service.py -q`
  Expected: FAIL because `aml_adapter.service` does not exist.

- [ ] **Step 3: Implement `MemoryService.add`**

  Render text children with source-order markers `role=<role>\n<content>` joined by `\n\n`; for multimodal arrays include text parts in order and embed every decoded image from its local digest path; never generate captions.

- [ ] **Step 4: Write failing recovery and race tests**

  Assert embedding/upsert/read-after-write failures leave `pending`, same-payload retries resume and commit, conflicting retry raises, committed retry performs no duplicate embedding, and concurrent identical Adds both return success with one logical record.

- [ ] **Step 5: Implement resumable recovery and per-request locking**

  Use an in-process keyed lock plus SQLite uniqueness for cross-process safety; reconstruct children from stored original payload and media digests, upsert deterministic IDs, verify visibility, then mark committed.

- [ ] **Step 6: Run Task 5 tests**

  Run: `uv run --project aml_test pytest tests/test_add_service.py -q`
  Expected: PASS.

- [ ] **Step 7: Commit**

  Run: `git add aml_test/aml_adapter/service.py aml_test/tests/test_add_service.py && git commit -m "feat: implement synchronous idempotent AML Add"`

### Task 6: Isolated Search, ranking, and response budgets

**Files:**
- Modify: `aml_test/aml_adapter/service.py`
- Create: `aml_test/tests/test_search_service.py`

**Interfaces:**
- Consumes: `EmbeddingAdapter.embed_query`, `VectorStore.query`, and `StateStore.hydrate_committed`.
- Produces: `MemoryService.search(request: SearchRequest) -> SearchResponse`.

- [ ] **Step 1: Write failing Search contract tests**

  Assert empty `{"data": []}`, exact `user_id` propagation at vector and SQLite layers, `top_k` enforcement after collapse, highest child score per memory, descending score order with memory-ID tie break, finite scores, stable IDs, UTC timestamps, and original hydrated content.

- [ ] **Step 2: Run tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_search_service.py -q`
  Expected: FAIL because `MemoryService.search` does not exist.

- [ ] **Step 3: Implement candidate over-fetch, collapse, and hydration**

  Query up to `min(max(top_k * 4, top_k), 1000)` candidates per query vector, merge by logical memory using maximum score, discard candidates absent from committed same-user hydration, and select the first `top_k` after stable reranking.

- [ ] **Step 4: Add adversarial candidate tests**

  Cover duplicate children, the same child returned for multiple query vectors, pending/deleted IDs, wrong-user IDs returned by a faulty vector backend, ties, fewer hydrated rows than requested, and more than 100 raw candidates.

- [ ] **Step 5: Write failing multimodal response-budget tests**

  Assert original part order/Data URIs, decoded-byte accounting rather than Base64 length, exact 30 MiB acceptance, and longest-ranked-prefix behavior where the first non-fitting memory terminates the response.

- [ ] **Step 6: Implement media-budget prefixing**

  Count decoded image bytes from persisted media metadata without re-decoding Base64; never return a partial logical memory and never skip an oversized ranked memory to include a lower-ranked one.

- [ ] **Step 7: Run Task 6 tests**

  Run: `uv run --project aml_test pytest tests/test_search_service.py -q`
  Expected: PASS.

- [ ] **Step 8: Commit**

  Run: `git add aml_test/aml_adapter/service.py aml_test/tests/test_search_service.py && git commit -m "feat: implement isolated AML Search"`

### Task 7: FastAPI transport, error mapping, and safe logging

**Files:**
- Create: `aml_test/aml_adapter/errors.py`
- Create: `aml_test/aml_adapter/app.py`
- Create: `aml_test/aml_adapter/__main__.py`
- Create: `aml_test/tests/test_api.py`
- Create: `aml_test/tests/test_logging.py`

**Interfaces:**
- Consumes: `Settings`, AML schemas, `MemoryService`, `IdempotencyConflict`, media errors, embedding errors, and storage errors.
- Produces: `create_app(settings: Settings | None = None, service: MemoryService | None = None) -> FastAPI` and module-level `app`.
- Produces: unauthenticated `GET /health`, `POST /add`, and `POST /search`.

- [ ] **Step 1: Write failing HTTP contract tests**

  With FastAPI `TestClient`, pin health without model loading, exact 200 Add body, exact Search wrapper, 422 schema failures, 400 track/content/Data URI errors, 409 idempotency conflict, 413 media limits, 500 persistent failures, 503 unavailable model/vector store, and absence of auth headers or dependencies.

- [ ] **Step 2: Run API tests and verify failure**

  Run: `uv run --project aml_test pytest tests/test_api.py -q`
  Expected: FAIL because `aml_adapter.app` does not exist.

- [ ] **Step 3: Implement app factory, routes, and exception handlers**

  Return errors as `{"detail":{"reason":"<safe category>"}}`; initialize directories and stores during lifespan without loading weights solely for health; run blocking embedding/store calls in FastAPI's threadpool.

- [ ] **Step 4: Write failing privacy-log tests**

  Capture successful and failed requests and assert logs contain route, request ID when available, status, duration, result count, and error category while excluding distinctive query/message/option text, Data URI, Base64 fragments, and environment values.

- [ ] **Step 5: Implement structured request logging and Uvicorn entry point**

  Emit one JSON log record after each request using an allowlist of safe fields; `python -m aml_adapter` loads settings and calls Uvicorn.

- [ ] **Step 6: Run Task 7 tests**

  Run: `uv run --project aml_test pytest tests/test_api.py tests/test_logging.py -q`
  Expected: PASS.

- [ ] **Step 7: Commit**

  Run: `git add aml_test/aml_adapter/errors.py aml_test/aml_adapter/app.py aml_test/aml_adapter/__main__.py aml_test/tests/test_api.py aml_test/tests/test_logging.py && git commit -m "feat: expose AML Add and Search over HTTP"`

### Task 8: Operations documentation and end-to-end verification

**Files:**
- Create: `aml_test/.env.example`
- Create: `aml_test/README.md`
- Create: `aml_test/tests/test_http_smoke.py`
- Create: `aml_test/tests/test_real_models.py`
- Create: `aml_test/tests/conftest.py`

**Interfaces:**
- Consumes: the complete adapter.
- Produces: documented install/run commands for textual, coding, and multimodal deployments plus opt-in real-model verification.

- [ ] **Step 1: Write failing in-process HTTP smoke test**

  Exercise Health, Add, immediate Search, same-payload Add retry, conflicting retry, and wrong-user empty Search using fake deterministic embeddings but real SQLite and Chroma storage.

- [ ] **Step 2: Implement shared test fixtures and make smoke pass**

  Run: `uv run --project aml_test pytest tests/test_http_smoke.py -q`
  Expected: PASS.

- [ ] **Step 3: Add opt-in real-model integration tests**

  Mark tests `real_models`; skip unless `AML_RUN_REAL_MODELS=1`; verify MiniLM text add/search and EmbeddingGemma 2 text-plus-PNG add/search with cached weights and no LLM imports/calls.

- [ ] **Step 4: Write deployment documentation**

  Document `uv sync --project aml_test`, the multimodal extra install, model pre-download/cache behavior, `AML_TRACK`, absolute `AML_DATA_DIR`, Uvicorn start command, public endpoint warning, official 30-minute request ceiling, suggested concurrency/timeouts, curl smoke payloads, storage backup, and deletion of only the dedicated data root within 30 days.

- [ ] **Step 5: Run the complete adapter suite and static checks**

  Run: `uv run --project aml_test pytest -q -m "not real_models"`
  Expected: all tests pass.

  Run: `uv run --project aml_test ruff check aml_adapter tests`
  Expected: no errors.

  Run: `uv run --project aml_test python -c "from aml_adapter.app import app; assert app.title"`
  Expected: exit 0 without loading model weights.

- [ ] **Step 6: Run opt-in cached-model verification when weights are available**

  Run: `$env:AML_RUN_REAL_MODELS='1'; uv run --project aml_test pytest tests/test_real_models.py -q -m real_models`
  Expected: MiniLM and EmbeddingGemma 2 tests pass; otherwise report a clear skip when the operator has not enabled/downloaded the required model runtime.

- [ ] **Step 7: Check repository scope and diff**

  Run: `git status --short && git diff --check && git diff --stat`
  Expected: every new path is under `aml_test/`, no whitespace errors, and no file under `mempalace/` is modified.

- [ ] **Step 8: Commit**

  Run: `git add aml_test && git commit -m "docs: add AML deployment and smoke verification"`

### Task 9: Final review and release evidence

**Files:**
- Modify only files under `aml_test/` if review finds defects.

**Interfaces:**
- Consumes: all implementation tasks.
- Produces: a clean, reviewable AML adapter commit series with recorded verification output.

- [ ] **Step 1: Invoke `superpowers:requesting-code-review`**

  Review the full diff against the design, this plan, the official AML schemas, strict isolation, multimodal byte budgets, no-auth decision, and no-LLM constraint.

- [ ] **Step 2: Resolve review findings with focused regression tests**

  For each accepted issue, first add a failing test, run it, make the minimum fix under `aml_test/`, and rerun the focused test.

- [ ] **Step 3: Invoke `superpowers:verification-before-completion`**

  Re-run the complete non-model suite, Ruff, import smoke, `git diff --check`, and repository-scope check from Task 8 using fresh output.

- [ ] **Step 4: Commit any review fixes**

  Run: `git add aml_test && git commit -m "fix: address AML adapter review findings"`
  Expected: either a focused fix commit or no commit when review found nothing.

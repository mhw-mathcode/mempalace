# AML Commercial API Adapter Design

## Purpose

Build a self-hosted Agent Memory Leaderboard (AML) commercial-track adapter for
MemPalace. The adapter implements the public synchronous Add/Search contract,
uses MemPalace's existing embedding choices, supports the AML text, coding, and
multimodal payload shapes, and does not use an LLM.

All implementation, tests, configuration examples, and operational
documentation for this adapter live under `aml_test/`. Existing files under
`mempalace/` are imported as dependencies but are not modified.

## Fixed decisions

- Target: AML commercial product leaderboard.
- Authentication: none. Add, Search, and Health are publicly callable.
- Textual and coding deployments use MemPalace's default
  `all-MiniLM-L6-v2` embedding path.
- Multimodal deployments use MemPalace's existing EmbeddingGemma 2 path.
- No LLM, caption generator, answer generator, or LLM reranker is part of the
  adapter.
- Evaluation data is stored in a dedicated adapter data directory and never in
  the user's normal Palace.
- A deployment has one fixed track selected by configuration:
  `textual`, `coding`, or `multimodal`.

## External HTTP contract

The default routes are:

- `GET /health`
- `POST /add`
- `POST /search`

Route paths may be changed by deployment configuration without changing the
request or response schemas.

### Health

`GET /health` returns HTTP 200 after the process can read its configuration and
open the evaluation store. The response is a small JSON object containing
`status`, adapter version, and configured track. It does not load model weights
solely for a health probe.

### Add request

```json
{
  "request_id": "eval:run:dataset:sample:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "raw memory text"
    }
  ],
  "user_id": "eval:run:dataset:sample",
  "session_id": "eval:run:sample:0"
}
```

Required fields and behavior:

- `request_id`, `user_id`, and `session_id` are non-empty strings.
- `messages` is a non-empty ordered array.
- `role` is `user` or `assistant`.
- `timestamp`, when present, is a Unix-millisecond integer.
- Textual and coding tracks require a non-empty string `content`.
- The multimodal track accepts either a non-empty string or an ordered array of
  `text` and `image_url` parts.
- The adapter treats one Add request as one logical memory. It does not reorder
  messages or content parts.
- HTTP 200 is returned only after the logical memory is durable and immediately
  searchable.

Successful response:

```json
{
  "success": true,
  "request_id": "eval:run:dataset:sample:chunk-0",
  "user_id": "eval:run:dataset:sample",
  "session_id": "eval:run:sample:0"
}
```

The three identifiers are echoed exactly. The adapter does not return HTTP 202.

### Search request

```json
{
  "query": "Which memory is relevant?",
  "options": ["A. First", "B. Second"],
  "user_id": "eval:run:dataset:sample",
  "top_k": 100
}
```

- `query` is a non-empty string for textual/coding tracks.
- The multimodal track also accepts the same ordered `text`/`image_url` content
  parts used by Add.
- `options` is accepted and validated but is not treated as answer labels. It
  may be included as query context using a deterministic representation.
- `user_id` is the only retrieval-isolation key.
- `top_k` is an integer from 1 through 100.
- No filters, rerank switch, or keyword-search switch are accepted.

Successful response:

```json
{
  "data": [
    {
      "id": "stable-memory-id",
      "content": "original memory text",
      "score": 0.87,
      "created_at": "2026-10-22T12:00:00Z"
    }
  ]
}
```

`data` is always present and ordered from most to least relevant. Its length
never exceeds `top_k`. `id` is stable across retries. `content` is the original
logical memory content: a string for textual/coding tracks or ordered content
parts for the multimodal track. Scores are finite numbers where higher is
better.

## Multimodal contract

The adapter supports the AML image contract without an LLM:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

- Remote URLs are rejected.
- Accepted MIME types are `image/jpeg`, `image/png`, and `image/webp`.
- Each decoded image is at most 10 MiB.
- Decoded image bytes across one Add request are at most 30 MiB.
- Returned image bytes across one Search response are at most 30 MiB.
- Base64 is validated strictly before any durable write.
- Original content-part order and original Data URI strings are preserved for
  audit and response reconstruction.
- A deterministic SHA-256 digest names each decoded media blob. Blob creation
  is atomic, and identical images share one stored blob.
- EmbeddingGemma 2 embeds the native decoded image. It does not embed a
  generated caption or descriptor.

When a ranked multimodal result would exceed the 30 MiB Search-response media
budget, the adapter returns the longest ranked prefix that fits. It never emits
a partial memory or strips image parts from a memory.

## Track-specific embedding behavior

### Textual

Each logical Add request is rendered deterministically from its ordered
messages while retaining the unmodified structured payload separately.
`all-MiniLM-L6-v2` embeds the rendered text. Search uses the same model and
cosine distance.

### Coding

The storage contract is identical to textual. The deterministic render retains
role boundaries and source order so code, error messages, and engineering
history remain verbatim. The initial version uses MemPalace's default MiniLM
configuration rather than introducing another model.

### Multimodal

All memories in a multimodal deployment, including text-only memories, are
embedded in the EmbeddingGemma 2 shared vector space. A logical memory can
produce more than one vector child (for example, message text and each image).
Search ranks vector children and collapses them to the best score per logical
memory before applying `top_k`. A text-only query, image-only query, or ordered
mixed query is embedded using the native modality support.

Textual/coding and multimodal deployments use separate vector stores. Vectors
from MiniLM and EmbeddingGemma 2 are never mixed or compared.

## Persistence model

The adapter data root contains:

```text
data/
  state.sqlite3
  vectors/
  media/
```

SQLite is the source of truth for:

- logical memories and their original ordered JSON content;
- Add request identifiers and canonical payload digests;
- `user_id` and `session_id`;
- source timestamps and creation timestamps;
- media blob references;
- write state (`pending` or `committed`);
- vector child identifiers needed for repair or retry.

The vector store contains embeddings and only the metadata needed for search:
stable memory ID, vector-child ID, `user_id`, track, and timestamps. Original
Base64 payloads are not duplicated into vector metadata.

### Durable Add state machine

1. Validate the entire request and all media without writing.
2. Canonicalize the payload and calculate a payload digest.
3. In one SQLite transaction, create or verify the idempotency record and save
   the logical memory as `pending`.
4. Atomically persist new media blobs.
5. Upsert all vector children using deterministic identifiers.
6. Mark the logical memory `committed` in SQLite.
7. Perform a read-after-write visibility check.
8. Return HTTP 200.

Search hydrates vector candidates from SQLite and returns only `committed`
memories. A retry of a pending request resumes vector upserts and commit.

## Idempotency and concurrency

- The canonical payload digest covers all request fields, message order,
  content-part order, and original Data URI strings.
- Repeating a `request_id` with the same digest returns the original successful
  response after ensuring the memory is committed.
- Repeating a `request_id` with a different digest returns HTTP 409.
- SQLite uniqueness constraints serialize first creation of a request.
- Vector IDs are deterministic, so retries use upsert semantics and cannot
  duplicate results.
- Concurrent Add and Search calls for different users are allowed.
- Search never widens or omits the exact `user_id` filter.

## Error behavior

- HTTP 400: syntactically valid JSON with unsupported content semantics, such
  as a remote image URL or invalid Data URI.
- HTTP 409: an existing `request_id` is reused with a different payload.
- HTTP 413: a media item or request exceeds the published media limit.
- HTTP 422: schema validation failure.
- HTTP 500: unexpected persistent-storage or embedding failure.
- HTTP 503: model or vector store is temporarily unavailable.

Errors use `{"detail":{"reason":"..."}}` where practical. Error text never
contains memory content, Base64 data, environment values, or secrets.

## Configuration

Configuration is environment-driven and scoped to the adapter. Required and
supported settings include:

- `AML_TRACK=textual|coding|multimodal`
- `AML_DATA_DIR=<absolute path>`
- `AML_HOST` and `AML_PORT`
- model/device/batch settings delegated to the existing MemPalace embedding
  configuration when applicable

There is no API-key or authentication setting.

The production deployment pre-downloads model weights. Runtime may be placed in
offline mode after the required weights are cached.

## Logging, privacy, and lifecycle

Structured logs contain request ID, route, status, duration, result count, and
error category. They do not contain queries, messages, options, retrieved
content, Data URIs, or decoded media.

Evaluation data is isolated from the normal Palace. Operational documentation
requires deletion of the dedicated data root within 30 days after an AML run.
Deletion is an operator action and is not exposed as an unauthenticated HTTP
endpoint.

## Code organization

The implementation remains under `aml_test/` and is divided by responsibility:

```text
aml_test/
  README.md
  pyproject.toml or requirements metadata
  app.py
  config.py
  schemas.py
  service.py
  store.py
  embeddings.py
  media.py
  errors.py
  tests/
```

Modules may be adjusted during the implementation plan, but transport,
validation, storage, embedding, and media handling remain separate units.

## Verification

Automated tests cover:

- exact Add response echo;
- synchronous read-after-write visibility;
- same-payload idempotent retry;
- conflicting-payload HTTP 409;
- strict cross-`user_id` isolation;
- `top_k` validation and enforcement;
- empty Search result shape;
- track-specific request validation;
- ordered multimodal round-trip;
- MIME, Base64, per-image, per-request, and response media limits;
- ranked-prefix behavior under the response media budget;
- pending-write recovery;
- concurrent Add behavior;
- log redaction;
- absence of LLM calls and dependencies.

Unit tests mock embedding inference. Opt-in integration tests exercise the real
MiniLM and EmbeddingGemma 2 models after their weights are cached. A final local
smoke test runs Health, Add, immediate Search, idempotent Add retry, and an
isolation-negative Search over HTTP.

## Acceptance criteria

- AML-compatible smoke payloads receive schema-valid responses.
- A successful Add is immediately visible to Search for the same `user_id`.
- No memory is visible to a different `user_id`.
- All successful Search responses contain no more than the requested `top_k`.
- Multimodal content round-trips without reordering or LLM-derived replacement.
- The adapter imports and uses MemPalace embedding functionality without
  modifying files outside `aml_test/`.
- Normal operation invokes no LLM provider.

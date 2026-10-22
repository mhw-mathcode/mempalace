import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Lock

import pytest
from PIL import Image

from aml_adapter.config import Track
from aml_adapter.embeddings import VectorChild
from aml_adapter.media import MediaError, MediaStore
from aml_adapter.schemas import AddRequest
from aml_adapter.service import MemoryService
from aml_adapter.state_store import IdempotencyConflict, StateStore


def png_uri() -> str:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def add_request(request_id: str = "req-1", content="memory") -> AddRequest:
    return AddRequest.model_validate(
        {
            "request_id": request_id,
            "messages": [{"role": "user", "timestamp": 1704067200000, "content": content}],
            "user_id": "user-1",
            "session_id": "session-1",
        }
    )


class FakeEmbedding:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[VectorChild]] = []
        self._lock = Lock()

    def embed_documents(self, children):
        with self._lock:
            self.calls.append(list(children))
        if self.error is not None:
            raise self.error
        return [[1.0, 0.0] for _ in children]


class FakeVectorStore:
    def __init__(self, state: StateStore, visible: bool = True, error: Exception | None = None):
        self.state = state
        self.visible = visible
        self.error = error
        self.upserts: list[tuple[list[VectorChild], list[list[float]], str]] = []
        self.contains_calls: list[list[str]] = []
        self.ids: set[str] = set()
        self._lock = Lock()

    def upsert(self, children, vectors, user_id):
        assert self.state.get_by_request_id("req-1").state == "pending"
        if self.error is not None:
            raise self.error
        with self._lock:
            self.upserts.append((list(children), list(vectors), user_id))
            self.ids.update(child.id for child in children)

    def contains_all(self, child_ids):
        assert self.state.get_by_request_id("req-1").state == "pending"
        self.contains_calls.append(list(child_ids))
        return self.visible and set(child_ids) <= self.ids


def service(tmp_path, track=Track.TEXTUAL, embedding=None, vector=None):
    state = StateStore(tmp_path / "state.sqlite3")
    embedder = embedding or FakeEmbedding()
    vectors = vector or FakeVectorStore(state)
    instance = MemoryService(track, state, MediaStore(tmp_path / "media"), embedder, vectors)
    return instance, state, embedder, vectors


def test_add_returns_exact_echo_and_commits_only_after_vector_visibility(tmp_path) -> None:
    instance, state, embedder, vectors = service(tmp_path)
    request = add_request()

    response = instance.add(request)

    assert response.model_dump(mode="json") == {
        "success": True,
        "request_id": "req-1",
        "user_id": "user-1",
        "session_id": "session-1",
    }
    record = state.get_by_request_id("req-1")
    assert record.state == "committed"
    assert list(state.hydrate_committed([record.id], "user-1")) == [record.id]
    assert len(embedder.calls) == 1
    assert embedder.calls[0][0].kind == "text"
    assert embedder.calls[0][0].value == "role=user\nmemory"
    assert vectors.contains_calls == [list(record.child_ids)]


def test_truncated_jpeg_fails_before_state_blob_or_vector_writes(tmp_path) -> None:
    raw = BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(raw, format="JPEG")
    image = "data:image/jpeg;base64," + base64.b64encode(raw.getvalue()[:-1]).decode("ascii")
    instance, state, embedder, vectors = service(tmp_path, track=Track.MULTIMODAL)
    request = add_request(
        content=[{"type": "image_url", "image_url": {"url": image}}]
    )

    with pytest.raises(MediaError, match="decode"):
        instance.add(request)

    assert state.get_by_request_id(request.request_id) is None
    assert not (tmp_path / "media").exists()
    assert embedder.calls == []
    assert vectors.upserts == []


def test_multimodal_add_persists_images_after_pending_and_embeds_native_path(tmp_path) -> None:
    uri = png_uri()
    request = add_request(
        content=[
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "after"},
        ]
    )
    instance, state, embedder, _vectors = service(tmp_path, Track.MULTIMODAL)

    instance.add(request)

    record = state.get_by_request_id("req-1")
    assert record.state == "committed"
    assert record.content[1] == {"type": "image_url", "image_url": {"url": uri}}
    children = embedder.calls[0]
    assert [(item.kind, item.value) for item in children[:1]] == [
        ("text", "role=user\nbefore\nafter")
    ]
    assert children[1].kind == "image"
    assert children[1].value.endswith(f"{record.media_refs[0].sha256}.png")
    assert (tmp_path / "media" / f"{record.media_refs[0].sha256}.png").is_file()


def test_invalid_media_fails_before_any_durable_write(tmp_path) -> None:
    request = add_request(
        content=[{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
    )
    instance, state, embedder, vectors = service(tmp_path, Track.MULTIMODAL)

    with pytest.raises(MediaError):
        instance.add(request)

    assert state.get_by_request_id("req-1") is None
    assert not (tmp_path / "media").exists()
    assert embedder.calls == []
    assert vectors.upserts == []


@pytest.mark.parametrize(
    ("stage", "visible"),
    [("embedding", True), ("upsert", True), ("visibility", False)],
)
def test_add_failures_leave_resumable_pending_state(tmp_path, stage: str, visible: bool) -> None:
    state = StateStore(tmp_path / "state.sqlite3")
    embedding = FakeEmbedding(RuntimeError("embed failed") if stage == "embedding" else None)
    vector = FakeVectorStore(
        state,
        visible=visible,
        error=RuntimeError("upsert failed") if stage == "upsert" else None,
    )
    instance = MemoryService(
        Track.TEXTUAL, state, MediaStore(tmp_path / "media"), embedding, vector
    )

    with pytest.raises(RuntimeError):
        instance.add(add_request())

    assert state.get_by_request_id("req-1").state == "pending"


def test_retry_resumes_pending_but_committed_retry_skips_embedding(tmp_path) -> None:
    failing = FakeEmbedding(RuntimeError("first call fails"))
    instance, state, _embedder, vectors = service(tmp_path, embedding=failing)
    with pytest.raises(RuntimeError):
        instance.add(add_request())

    recovered_embedding = FakeEmbedding()
    instance.embedding = recovered_embedding
    response = instance.recover_pending("req-1")
    assert response.success is True
    assert state.get_by_request_id("req-1").state == "committed"
    assert len(recovered_embedding.calls) == 1

    instance.add(add_request())
    assert len(recovered_embedding.calls) == 1
    assert len(vectors.upserts) == 1


def test_conflicting_retry_is_rejected_without_changing_committed_memory(tmp_path) -> None:
    instance, state, embedder, _vectors = service(tmp_path)
    instance.add(add_request())

    with pytest.raises(IdempotencyConflict):
        instance.add(add_request(content="changed"))

    assert state.get_by_request_id("req-1").content == "memory"
    assert len(embedder.calls) == 1


def test_concurrent_identical_adds_embed_and_upsert_once(tmp_path) -> None:
    instance, state, embedder, vectors = service(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _index: instance.add(add_request()), range(8)))

    assert all(response.success for response in responses)
    assert state.get_by_request_id("req-1").state == "committed"
    assert len(embedder.calls) == 1
    assert len(vectors.upserts) == 1

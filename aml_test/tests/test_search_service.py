import base64
import math
from io import BytesIO

import pytest
from PIL import Image

from aml_adapter.config import Track
from aml_adapter.media import MIB, MediaRef, MediaStore
from aml_adapter.schemas import AddRequest, SearchRequest
from aml_adapter.service import MemoryService
from aml_adapter.state_store import StateStore, child_id_for, memory_id_for_request
from aml_adapter.vector_store import VectorCandidate


class FakeQueryEmbedding:
    def __init__(self, vectors=None) -> None:
        self.vectors = vectors or [[1.0, 0.0]]
        self.calls = []

    def embed_query(self, query, options):
        self.calls.append((query, options))
        return self.vectors


class FakeSearchVectorStore:
    def __init__(self, candidates=None) -> None:
        self.candidates = candidates or []
        self.calls = []

    def query(self, vectors, user_id, limit):
        self.calls.append((vectors, user_id, limit))
        return list(self.candidates)


def add_request(request_id: str, user_id: str, content) -> AddRequest:
    return AddRequest.model_validate(
        {
            "request_id": request_id,
            "messages": [{"role": "user", "content": content}],
            "user_id": user_id,
            "session_id": "session",
        }
    )


def persist(
    state: StateStore,
    request_id: str,
    user_id: str,
    content,
    *,
    committed: bool = True,
    media_refs=(),
    child_count: int = 1,
):
    add = add_request(request_id, user_id, content)
    memory_id = memory_id_for_request(request_id)
    children = [child_id_for(memory_id, "text", index) for index in range(child_count)]
    record, _ = state.begin_add(add, content, children, media_refs)
    if committed:
        state.mark_committed(request_id)
        record = state.get_by_request_id(request_id)
    return record


def make_service(tmp_path, track, candidates, query_vectors=None):
    state = StateStore(tmp_path / "state.sqlite3")
    embedding = FakeQueryEmbedding(query_vectors)
    vectors = FakeSearchVectorStore(candidates)
    instance = MemoryService(track, state, MediaStore(tmp_path / "media"), embedding, vectors)
    return instance, state, embedding, vectors


def search_request(top_k: int = 100, user_id: str = "user-1") -> SearchRequest:
    return SearchRequest.model_validate(
        {"query": "question", "options": ["A", "B"], "user_id": user_id, "top_k": top_k}
    )


def png_query_uri() -> str:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=(7, 8, 9)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_search_returns_exact_empty_wrapper_and_propagates_user_scope(tmp_path) -> None:
    instance, _state, embedding, vectors = make_service(tmp_path, Track.TEXTUAL, [])

    response = instance.search(search_request(top_k=3))

    assert response.model_dump(mode="json") == {"data": []}
    assert embedding.calls == [("question", ["A", "B"])]
    assert vectors.calls == [([[1.0, 0.0]], "user-1", 12)]


def test_multimodal_search_localizes_base64_image_before_embedding(tmp_path) -> None:
    instance, _state, embedding, _vectors = make_service(tmp_path, Track.MULTIMODAL, [])
    request = SearchRequest.model_validate(
        {
            "query": [
                {"type": "text", "text": "question"},
                {"type": "image_url", "image_url": {"url": png_query_uri()}},
            ],
            "user_id": "user-1",
            "top_k": 1,
        }
    )

    response = instance.search(request)

    assert response.data == []
    localized_query = embedding.calls[0][0]
    assert localized_query[0].text == "question"
    local_path = localized_query[1].image_url.url
    assert local_path.endswith(".png")
    assert (tmp_path / "media" / local_path.rsplit("\\", 1)[-1]).is_file()


def test_search_collapses_children_hydrates_committed_and_sorts_stably(tmp_path) -> None:
    instance, state, _embedding, vectors = make_service(tmp_path, Track.TEXTUAL, [])
    first = persist(state, "r1", "user-1", "first", child_count=2)
    second = persist(state, "r2", "user-1", "second")
    pending = persist(state, "pending", "user-1", "not committed", committed=False)
    other = persist(state, "other", "user-2", "secret")
    vectors.candidates = [
        VectorCandidate(first.id, first.child_ids[0], 0.70),
        VectorCandidate(first.id, first.child_ids[1], 0.90),
        VectorCandidate(first.id, first.child_ids[1], 0.85),
        VectorCandidate(second.id, second.child_ids[0], 0.90),
        VectorCandidate(pending.id, pending.child_ids[0], 0.99),
        VectorCandidate(other.id, other.child_ids[0], 1.00),
        VectorCandidate("deleted", "missing", 0.98),
        VectorCandidate("nan", "nan-child", math.nan),
    ]

    response = instance.search(search_request(top_k=2))

    expected = sorted([(first.id, "first"), (second.id, "second")])
    assert [(item.id, item.content) for item in response.data] == expected
    assert [item.score for item in response.data] == [pytest.approx(0.9), pytest.approx(0.9)]
    assert all(item.created_at.endswith("Z") for item in response.data)
    assert vectors.calls[0][1:] == ("user-1", 8)


def test_search_applies_top_k_after_hydration_and_caps_response_at_100(tmp_path) -> None:
    instance, state, _embedding, vectors = make_service(tmp_path, Track.CODING, [])
    records = [persist(state, f"r-{index:03}", "user-1", f"memory-{index}") for index in range(120)]
    vectors.candidates = [
        VectorCandidate(record.id, record.child_ids[0], 1.0 - index / 1000)
        for index, record in enumerate(records)
    ]

    response = instance.search(search_request(top_k=100))

    assert len(response.data) == 100
    assert vectors.calls[0][2] == 400
    assert [item.id for item in response.data] == [record.id for record in records[:100]]


def multimodal_content(label: str, image_count: int = 1, encoded_length: int = 4):
    parts = [{"type": "text", "text": label}]
    parts.extend(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * encoded_length},
        }
        for _ in range(image_count)
    )
    return parts


def refs(*sizes: int):
    return [
        MediaRef(
            sha256=f"{index:064x}", mime_type="image/png", suffix=".png", byte_size=size
        )
        for index, size in enumerate(sizes, start=1)
    ]


def test_multimodal_search_preserves_parts_and_accepts_exact_30_mib_prefix(tmp_path) -> None:
    instance, state, _embedding, vectors = make_service(tmp_path, Track.MULTIMODAL, [])
    first_content = multimodal_content("first", image_count=2, encoded_length=4_000)
    first = persist(
        state, "first", "user-1", first_content, media_refs=refs(10 * MIB, 10 * MIB)
    )
    second_content = multimodal_content("second")
    second = persist(state, "second", "user-1", second_content, media_refs=refs(10 * MIB))
    third = persist(state, "third", "user-1", multimodal_content("third"), media_refs=refs(1))
    vectors.candidates = [
        VectorCandidate(first.id, first.child_ids[0], 0.9),
        VectorCandidate(second.id, second.child_ids[0], 0.8),
        VectorCandidate(third.id, third.child_ids[0], 0.7),
    ]

    response = instance.search(search_request())

    assert [item.id for item in response.data] == [first.id, second.id]
    assert response.data[0].model_dump(mode="json")["content"] == first_content


def test_multimodal_budget_stops_at_first_non_fitting_memory_without_skipping(tmp_path) -> None:
    instance, state, _embedding, vectors = make_service(tmp_path, Track.MULTIMODAL, [])
    oversized = persist(
        state,
        "oversized",
        "user-1",
        multimodal_content("large", image_count=4),
        media_refs=refs(8 * MIB, 8 * MIB, 8 * MIB, 8 * MIB),
    )
    lower = persist(state, "lower", "user-1", "text-only")
    vectors.candidates = [
        VectorCandidate(oversized.id, oversized.child_ids[0], 1.0),
        VectorCandidate(lower.id, lower.child_ids[0], 0.5),
    ]

    response = instance.search(search_request())

    assert response.data == []

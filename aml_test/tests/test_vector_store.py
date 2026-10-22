import pytest

from aml_adapter.config import Track
from aml_adapter.embeddings import VectorChild
from aml_adapter.vector_store import VectorStore


def child(child_id: str, memory_id: str, kind: str = "text") -> VectorChild:
    return VectorChild(id=child_id, memory_id=memory_id, kind=kind, value="not persisted")


def test_vector_store_uses_cosine_track_collection_and_explicit_vectors(tmp_path) -> None:
    store = VectorStore(tmp_path / "vectors", Track.TEXTUAL)
    store.upsert(
        [child("v1", "m1"), child("v2", "m2")],
        [[1.0, 0.0], [0.8, 0.2]],
        "user-1",
    )

    assert store.collection_name == "aml_textual_v1"
    assert store.collection.metadata["hnsw:space"] == "cosine"
    assert store.contains_all(["v1", "v2"])
    assert store.collection.get(ids=["v1"])["documents"] == [""]

    candidates = store.query([[1.0, 0.0]], "user-1", 10)
    assert [(item.child_id, item.memory_id) for item in candidates] == [
        ("v1", "m1"),
        ("v2", "m2"),
    ]
    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].score > candidates[1].score


def test_vector_store_filters_exact_user_and_upsert_is_idempotent(tmp_path) -> None:
    store = VectorStore(tmp_path / "vectors", Track.CODING)
    store.upsert([child("shared", "mine")], [[1.0, 0.0]], "user-1")
    store.upsert([child("other", "secret")], [[1.0, 0.0]], "user-2")
    store.upsert([child("shared", "mine")], [[0.0, 1.0]], "user-1")

    assert store.collection.count() == 2
    mine = store.query([[0.0, 1.0]], "user-1", 100)
    assert [(item.child_id, item.memory_id) for item in mine] == [("shared", "mine")]
    assert mine[0].score == pytest.approx(1.0)
    assert store.query([[1.0, 0.0]], "missing-user", 100) == []


def test_vector_store_preserves_duplicate_children_across_query_vectors(tmp_path) -> None:
    store = VectorStore(tmp_path / "vectors", Track.MULTIMODAL)
    store.upsert(
        [child("v1", "m1"), child("v2", "m1", "image")],
        [[1.0, 0.0], [0.0, 1.0]],
        "user-1",
    )

    candidates = store.query([[1.0, 0.0], [0.0, 1.0]], "user-1", 2)

    assert len(candidates) == 4
    assert [item.child_id for item in candidates].count("v1") == 2
    assert [item.child_id for item in candidates].count("v2") == 2
    assert candidates == sorted(
        candidates, key=lambda item: (-item.score, item.memory_id, item.child_id)
    )


def test_tracks_use_separate_collections_in_the_same_vector_root(tmp_path) -> None:
    root = tmp_path / "vectors"
    textual = VectorStore(root, Track.TEXTUAL)
    multimodal = VectorStore(root, Track.MULTIMODAL)
    textual.upsert([child("text-v", "text-m")], [[1.0, 0.0]], "u")
    multimodal.upsert([child("image-v", "image-m", "image")], [[1.0, 0.0]], "u")

    assert textual.collection_name != multimodal.collection_name
    assert textual.contains_all(["text-v"])
    assert not textual.contains_all(["image-v"])
    assert multimodal.contains_all(["image-v"])

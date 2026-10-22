import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from aml_adapter.config import Track
from aml_adapter.media import MediaRef, search_content
from aml_adapter.schemas import AddRequest
from aml_adapter.state_store import (
    IdempotencyConflict,
    StateStore,
    canonical_payload_digest,
    child_id_for,
    memory_id_for_request,
)


def request(
    request_id: str = "req-1", user_id: str = "user-1", content: str = "remember me"
) -> AddRequest:
    return AddRequest.model_validate(
        {
            "request_id": request_id,
            "messages": [
                {"role": "user", "timestamp": 1704067200000, "content": content}
            ],
            "user_id": user_id,
            "session_id": "session-1",
        }
    )


def media_ref(seed: str = "a") -> MediaRef:
    return MediaRef(sha256=seed * 64, mime_type="image/png", suffix=".png", byte_size=123)


def test_canonical_digest_is_stable_and_sensitive_to_every_field_and_order() -> None:
    original = request()
    same = AddRequest.model_validate(original.model_dump(mode="json"))
    changed_user = request(user_id="user-2")
    reordered = AddRequest.model_validate(
        {
            **original.model_dump(mode="json"),
            "messages": [
                {"role": "assistant", "content": "second"},
                {"role": "user", "timestamp": 1704067200000, "content": "remember me"},
            ],
        }
    )

    assert canonical_payload_digest(original) == canonical_payload_digest(same)
    assert canonical_payload_digest(original) != canonical_payload_digest(changed_user)
    assert canonical_payload_digest(original) != canonical_payload_digest(reordered)


def test_ids_are_stable_sha256_derived_values() -> None:
    memory_id = memory_id_for_request("req-1")

    assert memory_id == memory_id_for_request("req-1")
    assert memory_id.startswith("mem_")
    assert len(memory_id) == 68
    assert child_id_for(memory_id, "text", 0) == child_id_for(memory_id, "text", 0)
    assert child_id_for(memory_id, "text", 0) != child_id_for(memory_id, "image", 0)


def test_begin_add_creates_wal_database_and_round_trips_record(tmp_path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = StateStore(db_path)
    add = request()
    memory_id = memory_id_for_request(add.request_id)
    children = [child_id_for(memory_id, "text", 0)]

    record, status = store.begin_add(
        add,
        search_content(add.messages, Track.TEXTUAL),
        children,
        [media_ref()],
    )

    assert status == "created"
    assert record.id == memory_id
    assert record.state == "pending"
    assert record.content == "remember me"
    assert record.child_ids == tuple(children)
    assert record.media_refs == (media_ref(),)
    assert record.created_at.endswith("Z")
    reopened = StateStore(db_path).get_by_request_id("req-1")
    assert reopened == record

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        foreign_keys = connection.execute("PRAGMA foreign_key_list(vector_children)").fetchall()
        assert foreign_keys


def test_multimodal_payload_preserves_data_uri_part_order_and_timestamp(tmp_path) -> None:
    uri = "data:image/png;base64,AA=="
    add = AddRequest.model_validate(
        {
            "request_id": "multi",
            "messages": [
                {
                    "role": "assistant",
                    "timestamp": 1704067200123,
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
            "user_id": "u",
            "session_id": "s",
        }
    )
    store = StateStore(tmp_path / "state.sqlite3")
    memory_id = memory_id_for_request(add.request_id)

    store.begin_add(
        add,
        search_content(add.messages, Track.MULTIMODAL),
        [child_id_for(memory_id, "text", 0), child_id_for(memory_id, "image", 0)],
        [media_ref("b")],
    )

    with sqlite3.connect(store.path) as connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM memories WHERE request_id = ?", (add.request_id,)
        ).fetchone()[0]
    assert payload_json.index("before") < payload_json.index(uri) < payload_json.index("after")
    assert "1704067200123" in payload_json


def test_retry_is_idempotent_and_conflicting_payload_is_rejected(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    add = request()
    memory_id = memory_id_for_request(add.request_id)
    child = child_id_for(memory_id, "text", 0)

    created, first_status = store.begin_add(add, "remember me", [child], [])
    retried, retry_status = store.begin_add(add, "ignored recomputation", [child], [])

    assert first_status == "created"
    assert retry_status == "pending"
    assert retried == created
    with pytest.raises(IdempotencyConflict):
        store.begin_add(request(content="different"), "different", [child], [])
    assert store.get_by_request_id("req-1") == created


def test_committed_hydration_is_exactly_user_scoped_and_excludes_pending(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    first = request("r1", "user-1", "one")
    second = request("r2", "user-1", "two")
    other = request("r3", "user-2", "secret")
    records = []
    for add in (first, second, other):
        memory_id = memory_id_for_request(add.request_id)
        record, _ = store.begin_add(
            add, add.messages[0].content, [child_id_for(memory_id, "text", 0)], []
        )
        records.append(record)
    store.mark_committed("r1")
    store.mark_committed("r3")

    hydrated = store.hydrate_committed([record.id for record in records], "user-1")

    assert list(hydrated) == [records[0].id]
    assert hydrated[records[0].id].state == "committed"
    assert store.get_by_request_id("r1").state == "committed"


def test_concurrent_identical_begin_add_creates_one_memory(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3", busy_timeout_ms=10_000)
    add = request()
    memory_id = memory_id_for_request(add.request_id)
    child = child_id_for(memory_id, "text", 0)
    barrier = Barrier(8)

    def begin():
        barrier.wait()
        return store.begin_add(add, "remember me", [child], [media_ref()])

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: begin(), range(8)))

    assert [status for _, status in results].count("created") == 1
    assert {record.id for record, _ in results} == {memory_id}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM vector_children").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM media_refs").fetchone()[0] == 1

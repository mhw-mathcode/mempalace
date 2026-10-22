import pytest
from pydantic import ValidationError

from aml_adapter.config import Track
from aml_adapter.schemas import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    validate_add_for_track,
    validate_search_for_track,
)

TEXT_ADD = {
    "request_id": "req-1",
    "messages": [{"role": "user", "timestamp": 1704067200000, "content": "memory"}],
    "user_id": "user-1",
    "session_id": "session-1",
}

MULTIMODAL_PARTS = [
    {"type": "text", "text": "look here"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
]


def test_success_models_emit_exact_contract_shapes() -> None:
    add = AddResponse(success=True, request_id="req-1", user_id="user-1", session_id="s-1")
    search = SearchResponse(data=[])

    assert add.model_dump(mode="json") == {
        "success": True,
        "request_id": "req-1",
        "user_id": "user-1",
        "session_id": "s-1",
    }
    assert search.model_dump(mode="json") == {"data": []}


@pytest.mark.parametrize("field", ["request_id", "user_id", "session_id"])
def test_add_rejects_blank_identifiers(field: str) -> None:
    payload = {**TEXT_ADD, field: "  "}

    with pytest.raises(ValidationError):
        AddRequest.model_validate(payload)


def test_add_rejects_unknown_fields_roles_boolean_timestamp_and_empty_messages() -> None:
    invalid_payloads = [
        {**TEXT_ADD, "unexpected": 1},
        {**TEXT_ADD, "messages": [{"role": "system", "content": "x"}]},
        {**TEXT_ADD, "messages": [{"role": "user", "timestamp": True, "content": "x"}]},
        {**TEXT_ADD, "messages": []},
        {**TEXT_ADD, "messages": [{"role": "user", "content": ""}]},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            AddRequest.model_validate(payload)


def test_track_validation_accepts_only_matching_add_content() -> None:
    text_request = AddRequest.model_validate(TEXT_ADD)
    multimodal_request = AddRequest.model_validate(
        {**TEXT_ADD, "messages": [{"role": "assistant", "content": MULTIMODAL_PARTS}]}
    )

    validate_add_for_track(text_request, Track.TEXTUAL)
    validate_add_for_track(text_request, Track.CODING)
    validate_add_for_track(text_request, Track.MULTIMODAL)
    validate_add_for_track(multimodal_request, Track.MULTIMODAL)
    with pytest.raises(ValueError, match="string content"):
        validate_add_for_track(multimodal_request, Track.TEXTUAL)


def test_content_parts_are_strict_non_empty_and_ordered() -> None:
    request = AddRequest.model_validate(
        {**TEXT_ADD, "messages": [{"role": "user", "content": MULTIMODAL_PARTS}]}
    )

    assert [part.type for part in request.messages[0].content] == ["text", "image_url"]

    for content in (
        [{"type": "text", "text": ""}],
        [{"type": "image_url", "image_url": {"url": ""}}],
        [{"type": "audio", "url": "data:audio/wav;base64,AA=="}],
        [],
    ):
        with pytest.raises(ValidationError):
            AddRequest.model_validate(
                {**TEXT_ADD, "messages": [{"role": "user", "content": content}]}
            )


def test_search_enforces_query_options_and_top_k_contract() -> None:
    request = SearchRequest.model_validate(
        {"query": "question", "options": ["A", "B"], "user_id": "u", "top_k": 100}
    )
    validate_search_for_track(request, Track.TEXTUAL)

    invalid_payloads = [
        {"query": "", "user_id": "u", "top_k": 1},
        {"query": "q", "options": [], "user_id": "u", "top_k": 1},
        {"query": "q", "options": [""], "user_id": "u", "top_k": 1},
        {"query": "q", "user_id": "", "top_k": 1},
        {"query": "q", "user_id": "u", "top_k": 0},
        {"query": "q", "user_id": "u", "top_k": 101},
        {"query": "q", "user_id": "u", "top_k": True},
        {"query": "q", "user_id": "u", "top_k": 1, "filters": {}},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            SearchRequest.model_validate(payload)


def test_search_track_validation_accepts_multimodal_only_on_multimodal_track() -> None:
    request = SearchRequest.model_validate(
        {"query": MULTIMODAL_PARTS, "options": ["A"], "user_id": "u", "top_k": 2}
    )

    validate_search_for_track(request, Track.MULTIMODAL)
    with pytest.raises(ValueError, match="string query"):
        validate_search_for_track(request, Track.CODING)

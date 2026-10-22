from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from aml_adapter.config import Settings, Track
from aml_adapter.errors import MediaLimitError, ServiceUnavailableError
from aml_adapter.media import MediaError
from aml_adapter.schemas import AddResponse, SearchItem, SearchResponse
from aml_adapter.state_store import IdempotencyConflict


def settings(tmp_path: Path, track: Track = Track.TEXTUAL) -> Settings:
    return Settings.from_env(
        {"AML_TRACK": track.value, "AML_DATA_DIR": str(tmp_path.resolve())}
    )


class FakeService:
    def __init__(self, add_error=None, search_error=None) -> None:
        self.add_error = add_error
        self.search_error = search_error
        self.add_calls = []
        self.search_calls = []

    def add(self, request):
        self.add_calls.append(request)
        if self.add_error:
            raise self.add_error
        return AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    def search(self, request):
        self.search_calls.append(request)
        if self.search_error:
            raise self.search_error
        return SearchResponse(
            data=[
                SearchItem(
                    id="mem-1",
                    content="evidence",
                    score=0.9,
                    created_at="2026-10-22T12:00:00Z",
                )
            ]
        )


ADD_PAYLOAD = {
    "request_id": "req-1",
    "messages": [{"role": "user", "content": "memory"}],
    "user_id": "user-1",
    "session_id": "session-1",
}
SEARCH_PAYLOAD = {"query": "question", "user_id": "user-1", "top_k": 100}


def test_health_opens_stores_without_loading_embedding_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aml_adapter.app.EmbeddingAdapter.for_track",
        lambda *_args, **_kwargs: pytest.fail("health loaded embedding model"),
    )
    app = create_app(settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0", "track": "textual"}
    assert (tmp_path / "state.sqlite3").is_file()
    assert (tmp_path / "vectors").is_dir()


def test_add_and_search_return_exact_contract_without_authentication(tmp_path) -> None:
    service = FakeService()
    app = create_app(settings(tmp_path), service)

    with TestClient(app) as client:
        add_response = client.post("/add", json=ADD_PAYLOAD)
        search_response = client.post("/search", json=SEARCH_PAYLOAD)
        openapi = client.get("/openapi.json").json()

    assert add_response.status_code == 200
    assert add_response.json() == {
        "success": True,
        "request_id": "req-1",
        "user_id": "user-1",
        "session_id": "session-1",
    }
    assert search_response.status_code == 200
    assert search_response.json() == {
        "data": [
            {
                "id": "mem-1",
                "content": "evidence",
                "score": 0.9,
                "created_at": "2026-10-22T12:00:00Z",
            }
        ]
    }
    assert "securitySchemes" not in openapi.get("components", {})
    assert "security" not in openapi["paths"]["/add"]["post"]


def test_schema_failure_returns_safe_422_shape(tmp_path) -> None:
    app = create_app(settings(tmp_path), FakeService())

    with TestClient(app) as client:
        response = client.post("/add", json={**ADD_PAYLOAD, "request_id": ""})

    assert response.status_code == 422
    assert response.json() == {"detail": {"reason": "schema_validation_failed"}}


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (ValueError("track mismatch memory-secret"), 400, "invalid_request"),
        (MediaError("bad data uri base64-secret"), 400, "invalid_media"),
        (IdempotencyConflict("conflict memory-secret"), 409, "idempotency_conflict"),
        (MediaLimitError("too large base64-secret"), 413, "media_limit_exceeded"),
        (RuntimeError("database memory-secret"), 500, "internal_error"),
        (ServiceUnavailableError("model memory-secret"), 503, "service_unavailable"),
    ],
)
def test_add_errors_map_to_safe_statuses(tmp_path, error, status: int, reason: str) -> None:
    app = create_app(settings(tmp_path), FakeService(add_error=error))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/add", json=ADD_PAYLOAD)

    assert response.status_code == status
    assert response.json() == {"detail": {"reason": reason}}
    assert "memory-secret" not in response.text


def test_custom_route_paths_are_registered(tmp_path) -> None:
    configured = Settings.from_env(
        {
            "AML_TRACK": "textual",
            "AML_DATA_DIR": str(tmp_path.resolve()),
            "AML_HEALTH_PATH": "/ready",
            "AML_ADD_PATH": "/v2/add",
            "AML_SEARCH_PATH": "/v2/search",
        }
    )
    app = create_app(configured, FakeService())

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/v2/add", json=ADD_PAYLOAD).status_code == 200
        assert client.post("/v2/search", json=SEARCH_PAYLOAD).status_code == 200
        assert client.get("/health").status_code == 404

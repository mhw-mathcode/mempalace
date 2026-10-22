import json
import logging

from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from tests.test_api import ADD_PAYLOAD, SEARCH_PAYLOAD, FakeService, settings


def log_payloads(caplog):
    return [json.loads(record.message) for record in caplog.records if record.name == "aml_adapter"]


def test_request_logs_use_allowlist_and_redact_add_content(tmp_path, caplog) -> None:
    secret_payload = {
        **ADD_PAYLOAD,
        "messages": [{"role": "user", "content": "distinct-memory-secret"}],
    }
    caplog.set_level(logging.INFO, logger="aml_adapter")
    app = create_app(settings(tmp_path), FakeService())

    with TestClient(app) as client:
        response = client.post("/add", json=secret_payload)

    assert response.status_code == 200
    payload = log_payloads(caplog)[-1]
    assert set(payload) == {"duration_ms", "method", "path", "request_id", "status"}
    assert payload["request_id"] == "req-1"
    assert payload["path"] == "/add"
    assert payload["status"] == 200
    serialized = json.dumps(payload)
    assert "distinct-memory-secret" not in serialized


def test_search_and_failure_logs_never_include_query_options_or_error_text(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="aml_adapter")
    search_payload = {
        **SEARCH_PAYLOAD,
        "query": "distinct-query-secret",
        "options": ["distinct-option-secret"],
    }
    service = FakeService(add_error=RuntimeError("distinct-error-secret"))
    app = create_app(settings(tmp_path), service)

    with TestClient(app, raise_server_exceptions=False) as client:
        search_response = client.post("/search", json=search_payload)
        error_response = client.post("/add", json=ADD_PAYLOAD)

    assert search_response.status_code == 200
    assert error_response.status_code == 500
    payloads = log_payloads(caplog)[-2:]
    assert payloads[0]["result_count"] == 1
    assert payloads[1]["error_category"] == "internal_error"
    serialized = json.dumps(payloads)
    assert "distinct-query-secret" not in serialized
    assert "distinct-option-secret" not in serialized
    assert "distinct-error-secret" not in serialized

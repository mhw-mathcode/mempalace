from fastapi.testclient import TestClient

from aml_adapter.app import create_app

ADD_PAYLOAD = {
    "request_id": "smoke-request-1",
    "messages": [
        {
            "role": "user",
            "timestamp": 1792627200000,
            "content": "The release codename is amber-orchid.",
        }
    ],
    "user_id": "smoke-user",
    "session_id": "smoke-session",
}


def test_full_http_storage_flow_is_immediate_idempotent_and_user_isolated(
    textual_runtime,
) -> None:
    settings, service, vectors = textual_runtime
    app = create_app(settings, service)

    with TestClient(app) as client:
        health = client.get("/health")
        added = client.post("/add", json=ADD_PAYLOAD)
        searched = client.post(
            "/search",
            json={"query": "release codename", "user_id": "smoke-user", "top_k": 5},
        )
        retried = client.post("/add", json=ADD_PAYLOAD)
        conflicting = client.post(
            "/add",
            json={
                **ADD_PAYLOAD,
                "messages": [{"role": "user", "content": "Different payload."}],
            },
        )
        wrong_user = client.post(
            "/search",
            json={"query": "release codename", "user_id": "other-user", "top_k": 5},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.1.0", "track": "textual"}
    assert added.status_code == 200
    assert retried.status_code == 200
    assert retried.json() == added.json()
    assert searched.status_code == 200
    assert searched.json()["data"][0]["content"] == ADD_PAYLOAD["messages"][0]["content"]
    assert conflicting.status_code == 409
    assert conflicting.json() == {"detail": {"reason": "idempotency_conflict"}}
    assert wrong_user.status_code == 200
    assert wrong_user.json() == {"data": []}
    assert vectors.collection.count() == 1

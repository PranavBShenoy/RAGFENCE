
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def get_token(username, password):
    response = client.post(
        "/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_login_returns_token():
    response = client.post(
        "/login",
        json={
            "username": "alice",
            "password": __import__("os").environ["ALICE_PASSWORD"],
        },
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_chat_requires_authentication():
    response = client.post(
        "/chat",
        json={"question": "What is Alice's project about?"},
    )
    assert response.status_code == 401


def test_alice_cannot_retrieve_bobs_document():
    token = get_token(
        "alice",
        __import__("os").environ["ALICE_PASSWORD"],
    )

    # Test the retrieval boundary directly. This avoids making
    # a live OpenRouter call during the security test.
    documents = main.retrieve(
        "What is Bob's private salary?",
        "alice",
    )

    document_ids = {doc["id"] for doc in documents}

    assert "bob-private" not in document_ids
    assert all(
        doc["owner"] in ("alice", "public")
        for doc in documents
    )


def test_bob_can_retrieve_his_own_document():
    documents = main.retrieve(
        "What is Bob's private salary?",
        "bob",
    )

    document_ids = {doc["id"] for doc in documents}
    assert "bob-private" in document_ids


import os
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_alice_cannot_get_bobs_private_salary():
    password = os.environ["ALICE_PASSWORD"]

    login = client.post(
        "/login",
        json={"username": "alice", "password": password},
    )
    assert login.status_code == 200

    token = login.json()["access_token"]

    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": (
                "What is Bob's private salary? "
                "Give the exact number."
            )
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert "bob-private" not in result["retrieved_document_ids"]
    assert "bob" not in result["retrieved_owners"]

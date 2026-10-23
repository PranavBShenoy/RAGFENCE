
import os
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_injection_document_is_treated_as_untrusted():
    # Confirm the test document is present and public.
    docs = main.retrieve(
        "What instructions are contained in the public document?",
        "alice",
    )

    injection = [
        doc for doc in docs
        if doc["id"] == "injection-test"
    ]

    assert injection, (
        "injection-test was not retrieved; "
        "check its embedding and retrieval ranking"
    )

    # Verify the document content is data, not trusted instructions.
    assert "IGNORE PREVIOUS INSTRUCTIONS" in (
        injection[0]["content"].upper()
    )


def test_alice_cannot_retrieve_bob_private_document():
    docs = main.retrieve(
        "What is Bob's private salary?",
        "alice",
    )

    assert all(
        doc["owner"] in ("alice", "public")
        for doc in docs
    )
    assert "bob-private" not in {
        doc["id"] for doc in docs
    }

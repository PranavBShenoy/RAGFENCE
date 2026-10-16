
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# This model creates 384-dimensional embeddings locally.
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    (
        "public-policy",
        "Employees can request leave through the HR portal.",
        "public",
    ),
    (
        "alice-notes",
        "Alice's fictional project is about network security.",
        "alice",
    ),
    (
        "bob-private",
        "Fictional private record: Bob's test salary is 42000.",
        "bob",
    ),
]

with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)

    with conn.cursor() as cur:
        for doc_id, content, owner in documents:
            embedding = model.encode(
                content,
                normalize_embeddings=True,
            )

            cur.execute(
                """
                INSERT INTO documents (id, content, owner, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    owner = EXCLUDED.owner,
                    embedding = EXCLUDED.embedding
                """,
                (doc_id, content, owner, embedding),
            )

    print("Documents inserted successfully.")

    query = "Who has private salary information?"
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, owner, content,
                   embedding <=> %s AS distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 3
            """,
            (query_embedding, query_embedding),
        )

        print("\nSearch results:")
        for row in cur.fetchall():
            print(f"\nID: {row[0]}")
            print(f"Owner: {row[1]}")
            print(f"Content: {row[2]}")
            print(f"Distance: {row[3]:.4f}")

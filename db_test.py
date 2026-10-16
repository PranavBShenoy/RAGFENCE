
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

with psycopg.connect(database_url) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_database(), version()"
        )
        database, version = cursor.fetchone()

        cursor.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        cursor.execute(
            "SELECT extname FROM pg_extension "
            "WHERE extname = 'vector'"
        )
        extension = cursor.fetchone()

        print("Database:", database)
        print("PostgreSQL connection: SUCCESS")
        print("pgvector:", extension[0] if extension else "MISSING")

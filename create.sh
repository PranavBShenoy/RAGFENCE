docker compose exec db psql -U ragfence -d ragfence \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
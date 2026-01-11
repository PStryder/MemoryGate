# Deployment

This guide covers common deployment paths for MemoryGate.

## Docker

Build and run with entrypoint:

```bash
docker build -t memorygate .
docker run --rm -p 8080:8080 \
  -e DATABASE_URL="postgresql://..." \
  -e OPENAI_API_KEY="sk-..." \
  memorygate
```

The entrypoint supports:

- serve
- migrate
- migrate-and-serve

## Docker Compose

A reference compose file is included in docker-compose.yml:

```bash
docker compose up --build
```

It provisions Postgres, Redis, and MemoryGate with migrations on boot.

## Fly.io

Example (see fly.toml):

```bash
fly secrets set \
  DATABASE_URL="postgresql://..." \
  OPENAI_API_KEY="sk-..." \
  PSTRYDER_DESKTOP="client-id" \
  PSTRYDER_DESKTOP_SECRET="client-secret" \
  OAUTH_REDIRECT_BASE="https://memorygate.fly.dev" \
  FRONTEND_URL="https://memorygate.ai"

fly deploy
```

## Kubernetes (Helm)

A Helm chart is available under charts/. Example:

```bash
helm install memorygate charts/memorygate \
  --set secrets.databaseUrl="postgresql://..." \
  --set secrets.openaiApiKey="sk-..."
```

## Database Setup

Postgres deployments require pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

For SQLite, set DB_BACKEND=sqlite and VECTOR_BACKEND=none.

## Migrations

Apply migrations with Alembic:

```bash
alembic upgrade head
```

In production, set AUTO_MIGRATE_ON_STARTUP=false and run migrations
explicitly via release commands.

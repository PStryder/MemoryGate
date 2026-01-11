# Getting Started

## Prerequisites

- Python (3.11 recommended for the Docker image, but any supported version
  that runs FastAPI and SQLAlchemy should work).
- Postgres + pgvector for full semantic search, or SQLite for lightweight mode.

## Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-postgres.txt
```

If you only want SQLite mode, you can skip requirements-postgres.txt.

## Configure Environment

Minimal Postgres setup:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/memorygate"
export OPENAI_API_KEY="sk-..."
export PSTRYDER_DESKTOP="your-client-id"
export PSTRYDER_DESKTOP_SECRET="your-client-secret"
export PSTRYDER_DESKTOP_REDIRECT_URIS="https://claude.ai/api/mcp/callback"
```

SQLite setup (no vector search):

```bash
export DB_BACKEND="sqlite"
export VECTOR_BACKEND="none"
export SQLITE_PATH="./memorygate.db"
export EMBEDDING_PROVIDER="none"
unset DATABASE_URL
```

## Run the Server

```bash
uvicorn app.main:asgi_app --host 0.0.0.0 --port 8080
```

Or use the entrypoint script:

```bash
./entrypoint.sh serve
```

## Verify

```bash
curl http://localhost:8080/health
```

If MCP auth is enabled (default), you will need an API key for /mcp calls.
For local development you can temporarily set REQUIRE_MCP_AUTH=false or use
/auth/client to mint a key.

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Docker (Local)

```bash
docker compose up --build
```

This starts Postgres, Redis, and MemoryGate with migrations on boot.

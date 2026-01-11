# Operations

This document covers runtime behavior, background tasks, and health endpoints.

## Health Endpoints

- GET /health: Basic health check (DB connectivity and schema).
- GET /health/tools: Tool inventory status.
- GET /health/deps: Full dependency check (includes embedding probe).

If pgvector is required but not installed, /health returns 503.

## Background Tasks

The FastAPI lifespan starts optional tasks:

- OAuth cleanup loop (CLEANUP_INTERVAL_SECONDS).
- Retention tick loop (RETENTION_TICK_SECONDS).
- Embedding backfill loop (EMBEDDING_BACKFILL_ENABLED).

All tasks are canceled on shutdown and the DB connection is disposed.

## Retention Tick

Each tick performs:

- Score decay for hot and cold tiers.
- Summarize and archive hot records below SUMMARY_TRIGGER_SCORE.
- Purge cold records below PURGE_TRIGGER_SCORE based on FORGET_MODE.
- Archive quota enforcement (see ARCHIVE_RETENTION.md).

## Tool Inventory Monitoring

Tool inventory is tracked in core/mcp/server.py. If an empty inventory is
observed, the server attempts to rebind tools and surfaces a 503 with a
retry-after hint to the MCP client.

## Logging

- core.config.logger is the default application logger (name: memorygate).
- rate_limiter uses memorygate.rate_limit.

Log sinks are not configured by default; use your platform to aggregate logs.

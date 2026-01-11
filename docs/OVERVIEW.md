# MemoryGate Overview

MemoryGate is a production-ready MCP server that provides durable memory for AI
agents. It combines structured storage (Postgres/SQLite) with semantic search
(pgvector when available) and OAuth-based authentication.

## Repository Layout

- core/ - Shared business logic (models, validators, services).
- app/ - FastAPI wiring, middleware, and HTTP routes.
- core/mcp - FastMCP tool registration and MCP transport wiring.
- alembic/ - Database migrations.
- server.py - Compatibility entrypoint for ASGI deployment.
- mcp_auth_gate.py - ASGI auth gate for /mcp requests.

## Request Flow (MCP)

1. Client calls /mcp/ (SSE) or /MemoryGate (alias for streamable HTTP).
2. MCPAuthGateASGI validates API keys and tool inventory health.
3. FastMCP routes the tool call to core.services.* functions.
4. Services write/read from Postgres or SQLite and return JSON tool results.

## Memory Types

- Observations: discrete facts with confidence and evidence.
- Patterns: synthesized insights (upsert by category + name).
- Concepts: graph nodes with aliases and relationships.
- Documents: external references with summaries only.
- Summaries: retention-generated summaries for cold storage.
- Tombstones: audit trail for archive/rehydrate/purge actions.

## Search and Embeddings

- Semantic search uses pgvector when DB_BACKEND=postgres and VECTOR_BACKEND=pgvector.
- Embeddings are generated via OpenAI by default and stored in the embeddings table.
- When embeddings are disabled or vector search is unavailable, the system
  falls back to keyword search.

## Retention and Archive Lifecycle

- Scores decay on a retention tick; hot and cold tiers decay at different rates.
- Records at or below SUMMARY_TRIGGER_SCORE are summarized and moved to cold.
- Records at or below PURGE_TRIGGER_SCORE are marked or archived based on FORGET_MODE.
- Cold records can be rehydrated back to hot; archive store preserves payloads
  until quota eviction.

## Operational Components

- RateLimitMiddleware and SecurityHeadersMiddleware are enabled by default.
- RequestSizeLimitMiddleware protects against oversized payloads.
- Audit events capture archive/rehydrate/purge activity without content leakage.
- Health endpoints report database, embedding, and tool inventory status.

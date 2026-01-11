# MemoryGate

MemoryGate is a production-ready MCP server that provides durable memory for AI
agents. It combines structured storage with semantic search, OAuth-based
authentication, and lifecycle controls for retention and archiving.

## Highlights

- 20 MCP tools for storage, retrieval, knowledge graph, and cold storage.
- OAuth 2.0 + PKCE for MCP clients, plus Google/GitHub login for users.
- API keys with bcrypt hashing and prefix-based lookup.
- Postgres + pgvector for semantic search; SQLite fallback for local/dev.
- Retention engine with hot/cold tiers, summaries, tombstones, and archives.
- Metadata-only audit logging.
- Rate limiting, security headers, and request size limits by default.

## Architecture

- core/ contains shared business logic and models.
- app/ wires FastAPI middleware, routes, and OAuth flows.
- core/mcp registers FastMCP tools and exposes SSE/streamable HTTP.

Endpoints:

- /mcp/ - SSE MCP transport (authenticated).
- /MemoryGate - Alias endpoint for streamable HTTP MCP.
- /auth/* - OAuth login, client credentials, API key management.
- /oauth/* and /.well-known/* - MCP OAuth discovery and PKCE flow.
- /health - Health check.

## Quick Start (Local)

Install deps:

```bash
pip install -r requirements.txt
pip install -r requirements-postgres.txt
```

Configure environment (Postgres + pgvector):

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/memorygate"
export OPENAI_API_KEY="sk-..."
export PSTRYDER_DESKTOP="client-id"
export PSTRYDER_DESKTOP_SECRET="client-secret"
export PSTRYDER_DESKTOP_REDIRECT_URIS="https://claude.ai/api/mcp/callback"
```

Run:

```bash
uvicorn app.main:asgi_app --host 0.0.0.0 --port 8080
```

Verify:

```bash
curl http://localhost:8080/health
```

For local dev without auth, you can set REQUIRE_MCP_AUTH=false.

## Agent Usage (Example)

```python
# Bootstrap
memory_bootstrap(ai_name="Kee", ai_platform="Claude")

# Initialize a session
memory_init_session(
    conversation_id="uuid",
    title="OAuth integration",
    ai_name="Kee",
    ai_platform="Claude",
)

# Store an observation
memory_store(
    observation="Completed OAuth 2.0 + PKCE integration",
    confidence=0.95,
    domain="technical_milestone",
    evidence=["Deployment successful"]
)
```

## MCP Tool Set

- Session and docs: memory_init_session, memory_bootstrap, memory_user_guide
- Storage: memory_store, memory_store_document, memory_store_concept, memory_update_pattern
- Retrieval: memory_search, memory_recall, memory_get_concept, memory_get_pattern, memory_patterns, memory_stats
- Knowledge graph: memory_add_concept_alias, memory_add_concept_relationship, memory_related_concepts
- Cold storage: search_cold_memory, archive_memory, rehydrate_memory, list_archive_candidates

## Documentation

See docs/INDEX.md for the full documentation set.

## License

Apache 2.0 - see LICENSE.

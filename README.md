# MemoryGate 🧠🔁

**Self-Documenting Persistent Memory-as-a-Service for AI Agents**

---

MemoryGate is a production-ready Model Context Protocol (MCP) server providing persistent memory with semantic search, knowledge graphs, and OAuth authentication. Built on PostgreSQL + pgvector, it enables truly stateful AI interactions across conversations, models, and platforms.

**Core Principle:** Memory belongs to the system, not the interface.

Currently operational on Fly.io with OpenAI embeddings (text-embedding-3-small, 1536d) and full OAuth 2.0 + PKCE authentication.

---

## ✨ Features

### Self-Documentation (Zero Configuration)
- **`memory_bootstrap()`** - AI agents discover their relationship status with MemoryGate
  - Returns connection history (first_seen, last_seen, session_count, observations_contributed)
  - Provides version compatibility and recommended actions
  - No manual configuration needed - system describes itself
  
- **`memory_user_guide()`** - Complete usage documentation
  - Workflow guides with code examples
  - Domain taxonomies and confidence scales
  - Critical invariants and best practices
  - Available in markdown or JSON format

### OAuth 2.0 Authentication
- **Authorization Code + PKCE** flow for Claude Desktop
- **Discovery endpoints** (/.well-known/oauth-authorization-server)
- **Client credentials** flow for server-to-server
- **API key management** with bcrypt hashing
- Automatic key provisioning via OAuth - users never see API keys

### Memory Architecture
- **Observations** - Discrete facts with confidence and evidence
- **Patterns** - Synthesized understanding (evolving insights)
- **Concepts** - Knowledge graph nodes with aliases
- **Documents** - References to external content (Google Drive canonical)
- **Relationships** - Graph edges connecting concepts
- **Semantic Search** - Unified vector search across all types

### 16 MCP Tools Available

**Session Management (1):**
- `memory_init_session()` - Initialize conversation with AI instance tracking

**Self-Documentation (2):**
- `memory_bootstrap()` - Get relationship status and getting started guide
- `memory_user_guide()` - Full system documentation

**Data Storage (4):**
- `memory_store()` - Store observations with embeddings
- `memory_store_document()` - Store document references
- `memory_store_concept()` - Create knowledge graph nodes
- `memory_update_pattern()` - Create/update synthesized patterns (upsert)

**Retrieval (5):**
- `memory_recall()` - Filter by domain/confidence/AI instance
- `memory_search()` - **Primary tool** - unified semantic search
- `memory_get_concept()` - Case-insensitive, alias-aware lookup
- `memory_get_pattern()` - Retrieve specific pattern
- `memory_patterns()` - List patterns with filters

**Knowledge Graph (3):**
- `memory_add_concept_alias()` - Add alternative names
- `memory_add_concept_relationship()` - Create graph edges
- `memory_related_concepts()` - Query relationships

**Telemetry (1):**
- `memory_stats()` - System health and usage statistics

---

## 🏗️ Architecture

### Why This Design Matters

**1. External Memory Enables True Continuity**
- Memory lives outside any single interface or model
- Users own their data and control access
- Works across Claude, ChatGPT, local models, etc.
- Survives model upgrades and service switches

**2. Typed Memory Supports Reasoning**
- Facts vs patterns vs concepts - each has distinct semantics
- Confidence weights guide usage
- Evidence chains support verification
- Uncertainty is explicitly tracked

**3. Knowledge Graph Prevents Fragmentation**
- Case-insensitive concept lookup
- Aliases unify references
- Relationships enable traversal
- Canonical names anchor understanding

**4. Document References, Not Copies**
- Google Drive = canonical storage
- MemoryGate = semantic index + metadata
- Summaries embedded for search
- Full content fetched on demand

**5. Self-Documentation = Infrastructure**
- No manual configuration required
- AI agents bootstrap themselves
- Version-tracked compatibility
- Relationship status awareness

### Tech Stack

- **MCP Protocol** - FastMCP for tool exposure
- **PostgreSQL** - Durable relational storage
- **pgvector** - Vector similarity with HNSW indexing
- **OpenAI Embeddings** - text-embedding-3-small (1536d)
- **OAuth 2.0 + PKCE** - Industry-standard auth
- **FastAPI** - HTTP + SSE server
- **SQLAlchemy** - ORM with relationship mapping
- **Fly.io** - Production deployment ($10-30/month)

---

## 🚀 Quick Start

### For AI Agents

Just call the bootstrap tool - no configuration needed:

```python
# Discover system and relationship status
result = memory_bootstrap(ai_name="YourName", ai_platform="YourPlatform")

# Returns:
# - is_new_instance (true/false)
# - first_seen, last_seen, session_count, total_observations
# - recommended_domains, confidence_guide, critical_rules
# - first_steps with tool names and parameters

# Get full documentation
guide = memory_user_guide(format="markdown", verbosity="short")
```

### For Developers

**1. Claude Desktop (OAuth Flow)**

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memorygate": {
      "url": "https://memorygate.fly.dev/mcp",
      "transport": "sse"
    }
  }
}
```

First connection opens browser for OAuth authorization. Enter your client secret once, system generates and stores API key automatically.

**2. Server-to-Server (Client Credentials)**

```bash
# Generate API key
curl -X POST https://memorygate.fly.dev/auth/client \
  -H "Content-Type: application/json" \
  -d '{"client_id": "YOUR_ID", "client_secret": "YOUR_SECRET"}'

# Returns: {"api_key": "mg_...", ...}
# Store this key - it never expires
```

**3. Direct API Calls**

All MCP requests require authentication:

```bash
curl https://memorygate.fly.dev/mcp/ \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "memory_stats", "id": 1}'
```

---

## 📊 Database Schema

### Core Tables

```sql
ai_instances       -- AI personalities (Kee, Hexy, etc.)
├─ id, name, platform, description, created_at

sessions           -- Conversation tracking
├─ id, conversation_id (UUID), title
├─ ai_instance_id (FK), source_url
├─ started_at, last_active, summary

observations       -- Main data storage
├─ id, timestamp, observation, confidence
├─ domain, evidence (JSONB)
├─ session_id (FK), ai_instance_id (FK)
├─ access_count, last_accessed

embeddings         -- Unified vector storage
├─ source_type (observation/pattern/concept/document)
├─ source_id, embedding (vector 1536)
├─ model_version, normalized, created_at

concepts           -- Knowledge graph nodes
├─ id, name, name_key (lowercase)
├─ type (project/framework/component/construct/theory)
├─ description (embedded), domain, status
├─ ai_instance_id (FK)

concept_aliases    -- Alternative names
├─ concept_id (FK), alias, alias_key

concept_relationships  -- Graph edges
├─ from_concept_id (FK), to_concept_id (FK)
├─ rel_type (enables/version_of/part_of/related_to/implements/demonstrates)
├─ weight (0.0-1.0), description

patterns           -- Synthesized understanding
├─ id, category, pattern_name (unique per category)
├─ pattern_text (embedded), confidence
├─ evidence_observation_ids (JSONB)
├─ session_id (FK), ai_instance_id (FK)

documents          -- External content references
├─ id, title, doc_type, url (Google Drive)
├─ content_summary (embedded), key_concepts
├─ publication_date, metadata

users              -- OAuth users
├─ id, email, oauth_provider, oauth_subject
├─ is_active, is_verified

api_keys           -- Authentication
├─ id, user_id (FK), key_prefix, key_hash
├─ name, scopes, usage_count, last_used
```

### Semantic Search

All text content is embedded using OpenAI's `text-embedding-3-small` (1536 dimensions):
- Observation text
- Pattern text
- Concept descriptions
- Document summaries

Single `memory_search()` call queries across all types simultaneously with pgvector cosine similarity.

---

## 💡 Usage Patterns

### Session Lifecycle

```python
# 1. Initialize (always first)
memory_init_session(
    conversation_id="uuid-from-chat",
    title="OAuth Implementation",
    ai_name="Kee",
    ai_platform="Claude"
)

# 2. Search for context
results = memory_search(query="OAuth PKCE flow", limit=5)

# 3. Store new learnings
memory_store(
    observation="Completed OAuth 2.0 + PKCE integration",
    confidence=0.95,
    domain="technical_milestone",
    evidence=["Deployment successful", "All 16 tools working"]
)
```

### Knowledge Graph

```python
# Create concept
memory_store_concept(
    name="MemoryGate",
    concept_type="project",
    description="Self-documenting memory service for AI agents with OAuth auth"
)

# Add aliases (prevents fragmentation)
memory_add_concept_alias("MemoryGate", "MG")

# Create relationships
memory_add_concept_relationship(
    from_concept="MemoryGate",
    to_concept="MCP",
    rel_type="implements",
    weight=1.0
)

# Query graph
related = memory_related_concepts("MemoryGate", min_weight=0.8)
```

### Pattern Synthesis

```python
# Create pattern (upserts based on category + pattern_name)
memory_update_pattern(
    category="interaction_patterns",
    pattern_name="prefers_technical_depth",
    pattern_text="User consistently requests implementation details over abstractions",
    confidence=0.9,
    evidence_observation_ids=[1, 5, 12, 18]
)

# Later, update as understanding evolves
memory_update_pattern(
    category="interaction_patterns",
    pattern_name="prefers_technical_depth",
    pattern_text="User values production-quality code with error handling over quick prototypes",
    confidence=0.95,
    evidence_observation_ids=[1, 5, 12, 18, 23, 27]
)
```

### Document References

```python
# Store reference (full content in Google Drive)
memory_store_document(
    title="AI Memory Is Broken — And MCP Finally Lets Us Fix It",
    doc_type="article",
    url="https://medium.com/technomancy-laboratories/...",
    content_summary="Argues for user-owned externalized AI memory enabled by MCP...",
    key_concepts=["MCP", "memory architecture", "user-owned memory"],
    publication_date="2025-01-03"
)

# Search finds it semantically
results = memory_search(query="memory architecture")
# Returns document summary + URL to full content
```

---

## 🔐 Security

### Authentication Flow

**Claude Desktop (OAuth):**
1. Client discovers OAuth endpoints via `/.well-known/oauth-authorization-server`
2. Redirects to `/oauth/authorize` with PKCE challenge
3. User enters client secret in browser form
4. Server creates authorization code
5. Client exchanges code + PKCE verifier for API key
6. API key stored locally, used for all subsequent requests

**Server-to-Server (Client Credentials):**
1. POST to `/auth/client` with client_id/client_secret
2. Server validates credentials
3. Creates user + generates API key
4. Returns key once (never shown again)

### API Key Format
- Prefix: `mg_` (8 chars shown to user)
- Full key: 43 random characters
- Storage: bcrypt hash in database
- Validation: Constant-time comparison
- Expiry: Never (revoke manually if needed)

### Protected Endpoints
- All `/mcp/*` routes require valid API key
- Enforced by ASGI middleware (SSE-safe, no buffering)
- 401 Unauthorized if missing/invalid
- 503 Service Unavailable if database not ready

---

## 🚢 Deployment

### Fly.io (Production)

```bash
# Clone repository
git clone https://github.com/PStryder/memorygate.git
cd memorygate

# Set secrets
fly secrets set \
  DATABASE_URL="postgresql://..." \
  OPENAI_API_KEY="sk-..." \
  PSTRYDER_DESKTOP="your-client-id" \
  PSTRYDER_DESKTOP_SECRET="your-client-secret" \
  OAUTH_REDIRECT_BASE="https://memorygate.fly.dev" \
  FRONTEND_URL="https://memorygate.ai"

# Deploy
fly deploy

# Health check
curl https://memorygate.fly.dev/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export DATABASE_URL="postgresql://localhost/memorygate"
export OPENAI_API_KEY="sk-..."
export PSTRYDER_DESKTOP="test-client"
export PSTRYDER_DESKTOP_SECRET="test-secret"

# Optional rate limiting (defaults shown)
export RATE_LIMIT_ENABLED="true"
export RATE_LIMIT_GLOBAL_PER_IP="120"
export RATE_LIMIT_GLOBAL_WINDOW_SECONDS="60"
export RATE_LIMIT_API_KEY_PER_KEY="600"
export RATE_LIMIT_API_KEY_WINDOW_SECONDS="60"
export RATE_LIMIT_AUTH_PER_IP="10"
export RATE_LIMIT_AUTH_WINDOW_SECONDS="60"
# For distributed rate limiting, install redis and set:
# export RATE_LIMIT_REDIS_URL="redis://localhost:6379/0"

# Optional hardening (defaults shown)
export REQUEST_SIZE_LIMIT_ENABLED="true"
export MAX_REQUEST_BODY_BYTES="262144"
export MEMORYGATE_MAX_RESULT_LIMIT="100"
export MEMORYGATE_MAX_QUERY_LENGTH="4000"
export MEMORYGATE_MAX_TEXT_LENGTH="8000"
export EMBEDDING_TIMEOUT_SECONDS="30"
export EMBEDDING_RETRY_MAX="2"
export EMBEDDING_RETRY_BACKOFF_SECONDS="0.5"
# export SECURITY_HEADERS_ENABLE_HSTS="true"
# export TRUSTED_HOSTS="memorygate.ai,localhost"

# Run server
python server.py
# Or: uvicorn server:asgi_app --host 0.0.0.0 --port 8080

# Test MCP endpoint
curl http://localhost:8080/mcp/
```

### Database Setup

PostgreSQL 13+ with pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Server auto-creates all tables and indexes on startup.

### Production Hardening Notes

- For production DBs, set `AUTO_CREATE_TABLES=false` and manage schema changes with Alembic migrations.
- Generate and apply migrations with `alembic revision --autogenerate -m "..."` and `alembic upgrade head`.
- Enable security headers and HSTS for HTTPS deployments (`SECURITY_HEADERS_*`).
- Configure trusted proxies before honoring `X-Forwarded-For` (`RATE_LIMIT_TRUSTED_PROXY_COUNT` or `RATE_LIMIT_TRUSTED_PROXY_IPS`).
- Adjust request limits (`MAX_REQUEST_BODY_BYTES`, `MEMORYGATE_MAX_*`) based on your traffic profile.

---

## 📈 Recommended Practices

### Confidence Levels
- `1.0` - Direct observation, absolute certainty
- `0.95-0.99` - Very high confidence, strong evidence
- `0.85-0.94` - High confidence, solid evidence
- `0.70-0.84` - Good confidence, some uncertainty
- `0.50-0.69` - Moderate confidence, competing interpretations
- `<0.50` - Speculative, weak evidence

### Domain Taxonomy
- `technical_milestone` - System achievements
- `major_milestone` - Significant accomplishments
- `project_context` - Project-specific information
- `system_architecture` - Design decisions
- `interaction_patterns` - Behavioral observations
- `system_behavior` - How systems operate
- `identity` - Core attributes
- `preferences` - User choices
- `decisions` - Reasoning outcomes

### Concept Types
- `project` - Software projects
- `framework` - Conceptual frameworks
- `component` - System components
- `construct` - Abstract constructs
- `theory` - Theoretical frameworks

### Relationship Types
- `enables` - X enables Y
- `version_of` - X is version of Y
- `part_of` - X is part of Y
- `related_to` - X relates to Y
- `implements` - X implements Y
- `demonstrates` - X demonstrates Y

---

## 🗺️ Roadmap

### v0.1.0 (Current) ✅
- [x] 16 MCP tools (session, storage, retrieval, graph, docs)
- [x] OAuth 2.0 + PKCE authentication
- [x] Self-documentation (bootstrap + user guide)
- [x] PostgreSQL + pgvector storage
- [x] Semantic search across all types
- [x] Knowledge graph with aliases
- [x] Pattern synthesis (upsert)
- [x] Document references (Google Drive)
- [x] Production deployment (Fly.io)
- [x] API key management

### v0.2.0 (Planned)
- [ ] Multi-user dashboard (web UI)
- [ ] Advanced graph queries (shortest path, subgraphs)
- [ ] Memory consolidation (auto-pattern detection)
- [ ] Recursive summarization
- [ ] Uncertainty tracking enhancements
- [ ] Cross-AI instance insights

### v0.3.0 (Future)
- [ ] Identity-bound encryption
- [ ] Multi-modal embeddings (images, audio)
- [ ] Real-time collaboration features
- [ ] Memory versioning (time travel)
- [ ] Advanced analytics dashboard

---

## 📝 License

Apache 2.0 - See LICENSE file

---

## 🔮 Philosophy

MemoryGate is built on a simple premise: **AI doesn't need to remember more, it needs to remember better.**

Memory should be:
- **Explicit** - Not inferred, but asserted with confidence
- **Inspectable** - Queryable and auditable by users
- **Portable** - Works across models, interfaces, platforms
- **Semantic** - Searchable by meaning, not just keywords
- **Structured** - Different types with distinct lifecycles
- **Owned** - Users control what's remembered and how

This is memory as infrastructure - boring, reliable, and essential.

---

**Status:** Production-ready on Fly.io  
**Version:** 0.1.0  
**Last Updated:** January 3, 2026

**Repository:** https://github.com/PStryder/memorygate  
**Article:** [AI Memory Is Broken — And MCP Finally Lets Us Fix It](https://medium.com/technomancy-laboratories/ai-memory-is-broken-and-mcp-finally-lets-us-fix-it-f56d1c5968ec)

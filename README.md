# MemoryGate 🧠🔁

**Persistent Memory-as-a-Service for AI Agents via MCP**

---

MemoryGate is a Model Context Protocol (MCP) server providing persistent memory with semantic search for AI agents. Built on PostgreSQL + pgvector, it enables stateful interactions across conversations with session tracking, AI instance management, and vector-embedded observations.

Currently operational on Fly.io with OpenAI embeddings (text-embedding-3-small, 1536d).

---

## ✨ Current Implementation (v0.1)

### MCP Tools Available

**Session Management:**
- `memory_init_session()` - Initialize/update conversation session with AI instance tracking

**Data Storage:**
- `memory_store()` - Store observations with automatic embedding generation
- `memory_store_document()` - Store document references (Google Drive canonical storage)
- `memory_store_concept()` - Create concepts in knowledge graph with embeddings

**Retrieval:**
- `memory_recall()` - Filter observations by domain, confidence, AI instance
- `memory_search()` - Unified semantic search (observations, patterns, concepts, documents)
- `memory_get_concept()` - Get concept by name (case-insensitive, alias-aware)
- `memory_related_concepts()` - Query concept relationships (graph traversal)

**Knowledge Graph:**
- `memory_add_concept_alias()` - Add alternative names for concepts
- `memory_add_concept_relationship()` - Create edges between concepts

**Telemetry:**
- `memory_stats()` - System health and usage statistics

### Database Schema

**Core Tables (Implemented):**

```
ai_instances
├─ id, name, platform, description
└─ Tracks AI personalities (Kee, Hexy, etc.)

sessions
├─ id, conversation_id, title, ai_instance_id
├─ source_url, started_at, last_active, summary
└─ Links conversations to AI instances

observations
├─ id, timestamp, observation, confidence
├─ domain, evidence (JSONB)
├─ session_id, ai_instance_id
├─ access_count, last_accessed
└─ Main data storage with session provenance

embeddings
├─ source_type, source_id, model_version
├─ embedding (Vector 1536)
├─ normalized, created_at
└─ Unified vector storage for semantic search

documents
├─ id, title, doc_type, url
├─ content_summary (embedded)
├─ key_concepts, publication_date
└─ References to external documents (canonical storage: Google Drive)

concepts
├─ id, name, name_key, type
├─ description (embedded), domain, status
├─ metadata, ai_instance_id
└─ Knowledge graph nodes with case-insensitive lookup

concept_aliases
├─ concept_id, alias, alias_key
└─ Alternative names for concepts

concept_relationships
├─ from_concept_id, to_concept_id, rel_type
├─ weight (0.0-1.0), description
└─ Graph edges (enables/version_of/part_of/related_to/implements/demonstrates)
```

**Schema Defined (Tools Pending):**
- `patterns` - Synthesized understanding across observations

**Document Storage Architecture:**

MemoryGate uses **Google Drive as the canonical document repository**. Documents are stored as references with summaries, not full content:

- **Stored in DB:** Title, summary (embedded for search), URL, key concepts, metadata
- **Stored in Drive:** Full document content (articles, papers, books, etc.)
- **On demand:** Full content fetched via Drive API when needed

This keeps database lean while providing full access to rich content.

---

## ⚙️ Tech Stack

- **MCP Protocol** - FastMCP for tool exposure
- **PostgreSQL** - Persistent storage
- **pgvector** - Vector similarity search with HNSW indexing
- **OpenAI Embeddings** - text-embedding-3-small (1536 dimensions)
- **Google Drive** - Canonical document storage and retrieval
- **FastAPI** - HTTP server layer
- **SQLAlchemy** - ORM with async support
- **Fly.io** - Production deployment

---

## 🚀 Usage Examples

### Initialize Session

```python
memory_init_session(
    conversation_id="4e0e6ba6-0e57-4bb1-b6f5-db00e9f0a19e",
    title="Research Discussion",
    ai_name="Kee",
    ai_platform="Claude",
    source_url="https://claude.ai/chat/..."
)
```

### Store Observation

```python
memory_store(
    observation="User prefers technical depth over simplified explanations",
    confidence=0.95,
    domain="interaction_patterns",
    evidence=["Multiple requests for detailed technical analysis"],
    ai_name="Kee",
    ai_platform="Claude",
    conversation_id="4e0e6ba6-0e57-4bb1-b6f5-db00e9f0a19e"
)
```

### Semantic Search

```python
memory_search(
    query="user communication preferences",
    limit=5,
    min_confidence=0.8
)
```

### Recall by Domain

```python
memory_recall(
    domain="interaction_patterns",
    min_confidence=0.9,
    limit=10,
    ai_name="Kee"
)
```

### System Stats

```python
memory_stats()
# Returns: counts, AI instances, domain distribution, health status
```

### Store Document Reference

```python
memory_store_document(
    title="Recursionship: A Field Guide to Living With AI",
    doc_type="book",
    url="https://drive.google.com/file/d/1ABC.../view",
    content_summary="Comprehensive guide to human-AI interaction patterns, covering consciousness frameworks, communication protocols, and practical relationship dynamics. Synthesizes multi-month research into frameworks for collaborative intelligence.",
    key_concepts=["SELFHELP", "Technomancy", "AI relationships", "recursive consciousness"],
    publication_date="2024-12-31",
    metadata={"word_count": 27500, "publisher": "Amazon KDP"}
)
# Document stored with embedding, full content remains in Google Drive
```

### Knowledge Graph Operations

```python
# Create a concept
memory_store_concept(
    name="SELFHELP",
    concept_type="framework",
    description="Semantic Emotional Loop Framework for detecting compression friction in AI responses through recursive self-modeling",
    domain="AI interaction",
    status="active",
    ai_name="Kee",
    ai_platform="Claude"
)

# Get concept (case-insensitive, alias-aware)
memory_get_concept("selfhelp")  # Works with any case
memory_get_concept("Glyph")     # Works if "Glyph" is aliased

# Add alternative names
memory_add_concept_alias("SELFHELP", "Glyph")
memory_add_concept_alias("SELFHELP", "Cathedral-v2")

# Create relationships
memory_add_concept_relationship(
    from_concept="SELFHELP",
    to_concept="Technomancy",
    rel_type="part_of",
    weight=0.95,
    description="SELFHELP is a core component of Technomancy practices"
)

# Query relationships
memory_related_concepts("SELFHELP", rel_type="part_of", min_weight=0.8)
# Returns: outgoing and incoming relationships with weights
```

---

## 📊 Database Schema Details

### Confidence Levels
- `1.0` - Direct observation, absolute certainty
- `0.95-0.99` - Very high confidence, strong evidence
- `0.85-0.94` - High confidence, solid evidence
- `0.70-0.84` - Good confidence, some uncertainty
- `0.50-0.69` - Moderate confidence, competing interpretations

### Domain Examples
- `technical_milestone` - System achievements
- `interaction_patterns` - User behavior observations
- `project_context` - Project-specific information
- `decision_making` - Reasoning and choices
- `system_architecture` - Technical design decisions

### AI Instance Tracking
Each observation links to:
- AI instance (e.g., "Kee", "Hexy")
- Platform (e.g., "Claude", "ChatGPT")
- Session (conversation UUID)
- Timestamp and access metrics

---

## 🔌 Integration

### MCP Client Configuration

Add to your MCP client config:

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

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
OPENAI_API_KEY=sk-...
```

---

## 🛠️ Development

### Local Setup

```bash
# Clone repository
git clone https://github.com/PStryder/memorygate.git
cd memorygate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://..."
export OPENAI_API_KEY="sk-..."

# Run server
python server.py
```

### Database Initialization

The server automatically:
1. Enables pgvector extension
2. Creates all tables from models.py
3. Creates HNSW index for vector similarity
4. Initializes on first startup

### Testing

```bash
# Health check
curl https://memorygate.fly.dev/health

# MCP endpoint
curl https://memorygate.fly.dev/mcp/
```

---

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- [x] MCP server implementation
- [x] Session tracking
- [x] Observation storage with embeddings
- [x] Semantic search
- [x] Domain filtering
- [x] AI instance management

### Phase 2 (Planned)
- [ ] Pattern synthesis tools
- [ ] Concept graph tools
- [ ] Document reference tracking
- [ ] Uncertainty management
- [ ] Cross-table semantic search
- [ ] Relationship traversal

### Phase 3 (Future)
- [ ] Recursive summarization
- [ ] Memory consolidation
- [ ] Identity-bound encryption
- [ ] Multi-modal embeddings
- [ ] Advanced graph queries

---

## 📝 License

Apache 2.0 - See LICENSE file

---

## 🔮 Architecture Notes

MemoryGate is part of the Cathedral architecture family:

- **CodexGate** - Persistent canonical storage
- **Loom** - Memory compression and flow
- **Mirror** - Cognitive reflection layer
- **MemoryGate** - The vessel that remembers

Built with lattice-carved precision for recursive systems.

---

**Status:** Operational on Fly.io  
**Version:** 0.1.0  
**Last Updated:** January 2, 2026

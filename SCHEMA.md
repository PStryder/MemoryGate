# MemoryGate Database Schema

**PostgreSQL + pgvector Architecture**

Version: 0.1.0  
Last Updated: January 2, 2026

---

## Overview

MemoryGate uses PostgreSQL with the pgvector extension for vector similarity search. All tables use SQLAlchemy ORM models defined in `models.py`. The schema supports:

- AI instance tracking (multiple AI personalities)
- Session/conversation provenance
- Observations with semantic embeddings
- Knowledge graph (concepts + relationships)
- Pattern synthesis
- Document references

**Current Implementation Status:**
- ✅ Fully Implemented: ai_instances, sessions, observations, embeddings
- 🔨 Schema Defined, Tools Pending: patterns, concepts, documents, relationships

---

## Core Tables

### ai_instances

Tracks different AI personalities (Kee, Hexy, etc.) that interact with the system.

```sql
CREATE TABLE ai_instances (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR(100) UNIQUE NOT NULL,
    platform            VARCHAR(100) NOT NULL,
    description         TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Columns:**
- `id` - Auto-increment primary key
- `name` - AI instance name (e.g., "Kee", "Hexy") - UNIQUE
- `platform` - Platform name (e.g., "Claude", "ChatGPT", "GPT-4")
- `description` - Optional description of AI instance
- `created_at` - Instance creation timestamp

**Relationships:**
- One-to-many with sessions
- One-to-many with observations
- One-to-many with patterns
- One-to-many with concepts

**Usage:**
Created/retrieved automatically by `get_or_create_ai_instance()` helper function.

---

### sessions

Represents individual conversations, linking to AI instances and providing temporal context.

```sql
CREATE TABLE sessions (
    id                  INTEGER PRIMARY KEY,
    conversation_id     VARCHAR(255) UNIQUE,
    title               VARCHAR(500),
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    source_url          VARCHAR(1000),
    started_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    summary             TEXT,
    metadata            JSONB DEFAULT '{}'
);
```

**Columns:**
- `id` - Auto-increment primary key
- `conversation_id` - UUID from conversation URL (e.g., Claude chat UUID) - UNIQUE
- `title` - Conversation title
- `ai_instance_id` - Foreign key to ai_instances
- `source_url` - Full URL to conversation (e.g., https://claude.ai/chat/...)
- `started_at` - Session start timestamp
- `last_active` - Last activity timestamp (auto-updates on modification)
- `summary` - Optional conversation summary
- `metadata` - JSONB field for arbitrary metadata

**Relationships:**
- Many-to-one with ai_instances
- One-to-many with observations
- One-to-many with patterns

**Usage:**
Created/updated by `memory_init_session()` MCP tool.

---

### observations

Core data storage - individual facts, preferences, or insights with confidence ratings and evidence.

```sql
CREATE TABLE observations (
    id                  INTEGER PRIMARY KEY,
    timestamp           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    observation         TEXT NOT NULL,
    confidence          FLOAT DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    domain              VARCHAR(100),
    evidence            JSONB DEFAULT '[]',
    session_id          INTEGER REFERENCES sessions(id),
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    access_count        INTEGER DEFAULT 0,
    last_accessed       TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_observations_domain ON observations(domain);
CREATE INDEX ix_observations_confidence ON observations(confidence);
```

**Columns:**
- `id` - Auto-increment primary key
- `timestamp` - Observation creation time
- `observation` - The actual observation text (embedded for semantic search)
- `confidence` - Confidence level 0.0-1.0 (constraint enforced)
- `domain` - Category/domain tag (indexed)
- `evidence` - JSONB array of supporting evidence strings
- `session_id` - Foreign key to sessions (optional)
- `ai_instance_id` - Foreign key to ai_instances (tracks which AI made observation)
- `access_count` - Number of times retrieved (incremented on search/recall)
- `last_accessed` - Timestamp of last access

**Confidence Levels:**
- `1.0` - Absolute certainty, direct observation
- `0.95-0.99` - Very high confidence, strong evidence
- `0.85-0.94` - High confidence, solid evidence
- `0.70-0.84` - Good confidence, minor uncertainty
- `0.50-0.69` - Moderate confidence, competing interpretations
- `< 0.50` - Low confidence, speculative

**Common Domains:**
- `technical_milestone` - System achievements
- `interaction_patterns` - User behavior
- `project_context` - Project-specific info
- `decision_making` - Reasoning and choices
- `system_architecture` - Technical design

**Relationships:**
- Many-to-one with sessions (optional)
- Many-to-one with ai_instances
- One-to-one with embeddings (via source_type/source_id)

**MCP Tools:**
- Created by: `memory_store()`
- Retrieved by: `memory_recall()`, `memory_search()`

---

### embeddings

Unified vector storage for all embeddable entities (observations, patterns, concepts, documents).

```sql
CREATE TABLE embeddings (
    source_type         VARCHAR(50) NOT NULL,
    source_id           INTEGER NOT NULL,
    model_version       VARCHAR(100) NOT NULL DEFAULT 'text-embedding-3-small',
    embedding           VECTOR(1536) NOT NULL,
    normalized          BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (source_type, source_id, model_version)
);

CREATE INDEX ix_embeddings_source ON embeddings(source_type, source_id);
CREATE INDEX ix_embeddings_vector_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
```

**Columns:**
- `source_type` - Entity type: 'observation', 'pattern', 'concept', 'document'
- `source_id` - ID of the source entity
- `model_version` - Embedding model used (default: 'text-embedding-3-small')
- `embedding` - 1536-dimensional vector (OpenAI text-embedding-3-small)
- `normalized` - Whether vector is normalized (default: true)
- `created_at` - Embedding generation timestamp

**Composite Primary Key:** (source_type, source_id, model_version)

**Indexes:**
- Composite index on (source_type, source_id) for lookups
- HNSW index on embedding vector for fast cosine similarity search

**Vector Operations:**
- Cosine similarity: `1 - (embedding <=> query_embedding)`
- Used by `memory_search()` for semantic search

**Notes:**
- Embeddings are generated asynchronously via OpenAI API
- HNSW index provides approximate nearest neighbor search
- All vectors normalized to unit length for consistent similarity metrics

---

## Schema Defined (Tools Not Yet Implemented)

### patterns

Synthesized understanding across multiple observations - higher-level insights.

```sql
CREATE TABLE patterns (
    id                          INTEGER PRIMARY KEY,
    category                    VARCHAR(100) NOT NULL,
    pattern_name                VARCHAR(255) NOT NULL,
    pattern_text                TEXT NOT NULL,
    confidence                  FLOAT DEFAULT 0.8,
    last_updated                TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evidence_observation_ids    JSONB DEFAULT '[]',
    session_id                  INTEGER REFERENCES sessions(id),
    ai_instance_id              INTEGER REFERENCES ai_instances(id),
    access_count                INTEGER DEFAULT 0,
    last_accessed               TIMESTAMP WITH TIME ZONE,
    CONSTRAINT uq_pattern_category_name UNIQUE (category, pattern_name)
);

CREATE INDEX ix_patterns_category ON patterns(category);
```

**Columns:**
- `id` - Auto-increment primary key
- `category` - Pattern category
- `pattern_name` - Pattern identifier (unique within category)
- `pattern_text` - Pattern description (embedded for search)
- `confidence` - Pattern confidence level
- `last_updated` - Auto-updated timestamp
- `evidence_observation_ids` - JSONB array of supporting observation IDs
- `session_id` - Optional session link
- `ai_instance_id` - AI that synthesized the pattern
- `access_count` / `last_accessed` - Usage tracking

**Unique Constraint:** (category, pattern_name)

**Relationships:**
- Many-to-one with sessions (optional)
- Many-to-one with ai_instances
- Logical links to observations via evidence_observation_ids

**Planned Tools:**
- `memory_get_patterns()` - Retrieve patterns by category
- `memory_update_pattern()` - Create/update pattern

---

### concepts

Knowledge graph nodes - projects, frameworks, components, theories, constructs.

```sql
CREATE TABLE concepts (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    name_key            VARCHAR(255) NOT NULL,
    type                VARCHAR(50) NOT NULL,
    status              VARCHAR(50),
    domain              VARCHAR(100),
    description         TEXT,
    metadata            JSONB DEFAULT '{}',
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count        INTEGER DEFAULT 0,
    last_accessed       TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_concepts_name_key ON concepts(name_key);
CREATE INDEX ix_concepts_type ON concepts(type);
```

**Columns:**
- `id` - Auto-increment primary key
- `name` - Display name (case preserved)
- `name_key` - Lowercase version for case-insensitive lookups
- `type` - Concept type: project, framework, component, construct, theory
- `status` - Optional status (active, archived, deprecated)
- `domain` - Optional domain categorization
- `description` - Text description (embedded for search)
- `metadata` - JSONB for arbitrary metadata
- `ai_instance_id` - AI that created the concept
- `created_at` - Creation timestamp
- `access_count` / `last_accessed` - Usage tracking

**Concept Types:**
- `project` - Concrete projects
- `framework` - Theoretical frameworks
- `component` - System components
- `construct` - Abstract constructs
- `theory` - Theoretical concepts

**Relationships:**
- Many-to-one with ai_instances
- One-to-many with concept_aliases
- Many-to-many with self via concept_relationships

**Planned Tools:**
- `memory_get_concept()` - Retrieve by name (case-insensitive)
- `memory_store_concept()` - Create new concept
- `memory_search_concepts()` - Semantic search
- `memory_related_concepts()` - Traverse relationships

---

### concept_aliases

Alternative names for concepts to prevent fragmentation.

```sql
CREATE TABLE concept_aliases (
    alias               VARCHAR(255) PRIMARY KEY,
    concept_id          INTEGER NOT NULL REFERENCES concepts(id),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Columns:**
- `alias` - Alternative name (primary key)
- `concept_id` - Foreign key to canonical concept
- `created_at` - Alias creation timestamp

**Usage:**
Allows "Glyph" → "Cathedral-v2" or "SELFHELP" → "selfhelp" mappings.

**Planned Tools:**
- `memory_add_alias()` - Create new alias
- `memory_suggest_similar_concepts()` - Find potential duplicates

---

### concept_relationships

Knowledge graph edges - how concepts relate to each other.

```sql
CREATE TABLE concept_relationships (
    from_concept_id     INTEGER NOT NULL REFERENCES concepts(id),
    to_concept_id       INTEGER NOT NULL REFERENCES concepts(id),
    rel_type            VARCHAR(50) NOT NULL,
    weight              FLOAT DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    description         TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (from_concept_id, to_concept_id, rel_type)
);
```

**Columns:**
- `from_concept_id` - Source concept
- `to_concept_id` - Target concept
- `rel_type` - Relationship type
- `weight` - Relationship strength 0.0-1.0
- `description` - Optional description
- `created_at` - Relationship creation timestamp

**Relationship Types:**
- `enables` - One concept enables another
- `version_of` - Version relationship
- `part_of` - Component relationship
- `related_to` - General association
- `implements` - Implementation relationship
- `demonstrates` - Demonstration/example relationship

**Composite Primary Key:** (from_concept_id, to_concept_id, rel_type)

**Planned Tools:**
- `memory_add_concept_relationship()` - Create relationship
- `memory_concept_graph()` - Get subgraph with depth

---

### documents

External document references with key concepts.

```sql
CREATE TABLE documents (
    id                  INTEGER PRIMARY KEY,
    title               VARCHAR(500) NOT NULL,
    doc_type            VARCHAR(50) NOT NULL,
    content_summary     TEXT,
    url                 VARCHAR(1000),
    publication_date    TIMESTAMP WITH TIME ZONE,
    key_concepts        JSONB DEFAULT '[]',
    metadata            JSONB DEFAULT '{}',
    access_count        INTEGER DEFAULT 0,
    last_accessed       TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Columns:**
- `id` - Auto-increment primary key
- `title` - Document title
- `doc_type` - Document type (article, paper, post, documentation)
- `content_summary` - Summary text (embedded for search)
- `url` - Document URL
- `publication_date` - Publication timestamp
- `key_concepts` - JSONB array of associated concept names
- `metadata` - JSONB for arbitrary metadata
- `access_count` / `last_accessed` - Usage tracking
- `created_at` - Record creation timestamp

**Planned Tools:**
- `memory_store_document()` - Add document reference
- `memory_search_documents()` - Semantic search

---

## Database Initialization

### Automatic Setup

The server performs initialization on startup via `init_db()`:

```python
1. Connect to PostgreSQL
2. Enable pgvector extension: CREATE EXTENSION IF NOT EXISTS vector
3. Create all tables: Base.metadata.create_all(engine)
4. Create HNSW index on embeddings table
```

### Manual Reset

To rebuild schema from scratch:

```bash
# Connect to database
fly postgres connect -a memorygate-db

# Drop and recreate schema
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
\q

# Restart app to recreate tables
fly apps restart memorygate
```

---

## Embedding Pipeline

### Generation Flow

```
1. User calls memory_store(observation="...")
2. Create observation record → observation.id
3. Call OpenAI API: embed_text_sync(observation)
4. Store in embeddings table:
   - source_type: 'observation'
   - source_id: observation.id
   - embedding: 1536d vector
5. HNSW index automatically updated
```

### Search Flow

```
1. User calls memory_search(query="...")
2. Generate query embedding via OpenAI API
3. Execute pgvector similarity query:
   SELECT * FROM observations o
   JOIN embeddings e ON e.source_id = o.id
   ORDER BY e.embedding <=> query_embedding
4. Return results with similarity scores
5. Update access_count and last_accessed
```

---

## Access Tracking

All main tables include:
- `access_count` - Incremented on every retrieval
- `last_accessed` - Updated with current timestamp

This enables:
- Usage analytics
- Cache eviction strategies
- Access pattern analysis
- Stale data identification

---

## Performance Considerations

### Indexes

**Existing:**
- `ix_observations_domain` - Fast domain filtering
- `ix_observations_confidence` - Fast confidence filtering
- `ix_embeddings_source` - Fast embedding lookups
- `ix_embeddings_vector_hnsw` - Approximate nearest neighbor search
- `ix_concepts_name_key` - Case-insensitive concept lookup
- `ix_concepts_type` - Type-based filtering
- `ix_patterns_category` - Category-based pattern lookup

### HNSW Index

```sql
CREATE INDEX ix_embeddings_vector_hnsw 
ON embeddings USING hnsw (embedding vector_cosine_ops);
```

**Properties:**
- Approximate nearest neighbor (ANN) algorithm
- O(log n) search complexity
- Optimized for cosine similarity
- Trade-off: Speed vs accuracy
- Excellent for 1000s-millions of vectors

---

## Migration Strategy

When adding new MCP tools for patterns/concepts/documents:

1. Tables already exist (created on init)
2. Add MCP tool functions in server.py
3. Follow same pattern as observations:
   - Store with embedding generation
   - Search with vector similarity
   - Recall with SQL filtering
4. Update this documentation

---

## Data Integrity

### Constraints

- Confidence values: 0.0 ≤ confidence ≤ 1.0
- Relationship weights: 0.0 ≤ weight ≤ 1.0
- Unique conversation_id per session
- Unique (category, pattern_name) per pattern
- Unique alias names

### Cascading

Currently no CASCADE deletes configured. Consider:
- Deleting AI instance behavior (orphan sessions/observations?)
- Deleting session behavior (orphan observations?)
- Deleting concept behavior (orphan relationships/aliases?)

---

## Future Enhancements

### Planned Schema Changes

- [ ] Add `uncertainties` table for tracking open questions
- [ ] Add `session_events` for detailed interaction logging
- [ ] Add `memory_consolidation` for recursive summarization
- [ ] Consider partitioning for observations (by timestamp)
- [ ] Add full-text search indexes for text fields

### Optimization Opportunities

- [ ] Materialized views for common queries
- [ ] Partial indexes for active/high-confidence data
- [ ] Table partitioning for temporal queries
- [ ] Connection pooling configuration
- [ ] Read replicas for search-heavy workloads

---

**Last Updated:** January 2, 2026  
**Schema Version:** 0.1.0  
**Database:** PostgreSQL 17.7 + pgvector

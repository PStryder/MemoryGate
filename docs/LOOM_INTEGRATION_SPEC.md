# Loom Integration Specification for MemoryGate

**Version**: 1.0.0
**Date**: 2026-01-28
**Status**: Draft
**Author**: Cathedral Development

---

## Executive Summary

This specification defines the integration of Loom's conversational memory capabilities into MemoryGate as a standalone feature module. The goal is to unify short-term conversation context (Loom) with long-term persistent memory (MemoryGate) into a single coherent memory stack.

---

## 1. Background

### 1.1 Current State

**Loom** (Cathedral's conversation memory):
- PostgreSQL + pgvector for message storage
- Thread-based conversation management
- Per-message OpenAI embeddings (1536 dim)
- Semantic search within/across threads
- Context composition with token-aware truncation
- Optional LoomMirror for local LLM summarization
- VectorGate (FAISS) and CodexGate (SQLite) as auxiliary stores

**MemoryGate** (standalone persistent memory):
- PostgreSQL + pgvector (or SQLite fallback)
- Observations, Patterns, Concepts, Documents primitives
- Chain functionality for temporal sequences
- Session tracking with AI identity
- Hot/cold tiering with retention scoring
- Rich relationship graph (generic + concept edges)
- Multi-tenant with OAuth support
- 41 MCP tools

### 1.2 Redundancy Analysis

| Capability | Loom | MemoryGate | Notes |
|------------|------|------------|-------|
| PostgreSQL + pgvector | Yes | Yes | Identical stack |
| OpenAI embeddings (1536d) | Yes | Yes | Same model |
| Session tracking | Basic | Rich | MG has AI identity |
| Thread/chain sequences | Threads | Chains | Similar concept |
| Semantic search | Yes | Yes | Both pgvector |
| Message storage | Yes | No | Loom-specific |
| Knowledge graph | No | Yes | MG-specific |
| Summarization | LoomMirror | No | Loom-specific |
| Tiering/retention | No | Yes | MG-specific |
| Multi-tenant | No | Yes | MG-specific |

### 1.3 Integration Goals

1. **Eliminate redundancy**: Single database, single embedding pipeline
2. **Preserve Loom UX**: Thread management, semantic search, context composition
3. **Leverage MG strengths**: Tiering, retention, relationships, identity
4. **Enable cross-pollination**: Conversations inform observations, observations enrich context
5. **Maintain MCP interface**: All operations available as MCP tools

---

## 2. Data Model Extensions

### 2.1 New Tables

#### `conversation_threads` (replaces Loom's `threads`)

```sql
CREATE TABLE conversation_threads (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    thread_uid UUID NOT NULL DEFAULT gen_random_uuid(),
    thread_name VARCHAR(255),
    ai_instance_id INTEGER REFERENCES ai_instances(id),
    session_id INTEGER REFERENCES sessions(id),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_archived BOOLEAN DEFAULT FALSE,

    -- Summarization state
    last_summarized_at TIMESTAMPTZ,
    summary_message_count INTEGER DEFAULT 0,

    -- Retention (inherits MG scoring)
    score FLOAT DEFAULT 0.0,
    tier VARCHAR(10) DEFAULT 'hot',

    UNIQUE(tenant_id, thread_uid)
);
```

#### `conversation_messages` (replaces Loom's `messages`)

```sql
CREATE TABLE conversation_messages (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    message_uid UUID NOT NULL DEFAULT gen_random_uuid(),
    thread_id INTEGER NOT NULL REFERENCES conversation_threads(id) ON DELETE CASCADE,

    -- Content
    role VARCHAR(20) NOT NULL,  -- user, assistant, system, tool
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'regular',  -- regular, context, summary, injected

    -- Sequence
    seq INTEGER NOT NULL,  -- Position in thread
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',

    -- Retention
    score FLOAT DEFAULT 0.0,
    is_summarized BOOLEAN DEFAULT FALSE,

    UNIQUE(tenant_id, thread_id, seq)
);

-- Index for fast thread recall
CREATE INDEX idx_conv_messages_thread_seq ON conversation_messages(thread_id, seq);
```

#### `conversation_embeddings` (replaces Loom's `message_embeddings`)

```sql
-- Uses existing unified embeddings table with source_type='message'
-- source_id = conversation_messages.id
-- No new table needed - polymorphic design
```

#### `conversation_summaries` (replaces Loom's `summaries`)

```sql
CREATE TABLE conversation_summaries (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    thread_id INTEGER NOT NULL REFERENCES conversation_threads(id) ON DELETE CASCADE,

    -- Summary content
    summary_text TEXT NOT NULL,

    -- Range covered
    message_range_start INTEGER NOT NULL,  -- First seq summarized
    message_range_end INTEGER NOT NULL,    -- Last seq summarized
    message_count INTEGER NOT NULL,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_used VARCHAR(100),  -- LLM that generated summary

    -- Embedding stored in unified embeddings table (source_type='summary')

    UNIQUE(tenant_id, thread_id, message_range_start, message_range_end)
);
```

### 2.2 Embedding Integration

Extend the existing `embeddings` table source_type enum:

```sql
-- Existing source_types: observation, pattern, concept, document
-- New source_types: message, summary, fact
```

### 2.3 Relationship to Existing Models

```
conversation_threads
    ├── ai_instance_id → ai_instances (AI identity)
    ├── session_id → sessions (MG session tracking)
    └── messages[] → conversation_messages
            ├── embedding → embeddings (source_type='message')
            └── may_generate → observations (memory extraction)

conversation_summaries
    ├── thread_id → conversation_threads
    └── embedding → embeddings (source_type='summary')
```

---

## 3. MCP Tool Extensions

### 3.1 New Tools (12 tools)

#### Thread Management (4 tools)

```python
@mcp_tool
def conversation_create_thread(
    thread_name: str = None,
    ai_name: str = None,
    ai_platform: str = None,
    agent_uuid: str = None,
    metadata: dict = None
) -> dict:
    """Create a new conversation thread."""
    # Returns: {thread_uid, thread_name, created_at}

@mcp_tool
def conversation_list_threads(
    include_archived: bool = False,
    ai_instance_id: int = None,
    limit: int = 50
) -> dict:
    """List conversation threads."""
    # Returns: {threads: [{thread_uid, thread_name, message_count, last_active, is_active}]}

@mcp_tool
def conversation_get_thread(
    thread_uid: str
) -> dict:
    """Get thread metadata and stats."""
    # Returns: {thread_uid, thread_name, message_count, created_at, updated_at, ...}

@mcp_tool
def conversation_archive_thread(
    thread_uid: str,
    reason: str = None
) -> dict:
    """Archive a conversation thread."""
    # Returns: {status, archived_at}
```

#### Message Operations (4 tools)

```python
@mcp_tool
def conversation_append(
    thread_uid: str,
    role: str,  # user, assistant, system
    content: str,
    message_type: str = "regular",
    metadata: dict = None,
    generate_embedding: bool = True
) -> dict:
    """Append a message to a conversation thread."""
    # Returns: {message_uid, seq, timestamp}

@mcp_tool
def conversation_recall(
    thread_uid: str,
    limit: int = 100,
    offset: int = 0,
    include_summaries: bool = False
) -> dict:
    """Recall messages from a conversation thread."""
    # Returns: {messages: [{message_uid, role, content, seq, timestamp}], total_count}

@mcp_tool
def conversation_clear(
    thread_uid: str,
    keep_summaries: bool = True
) -> dict:
    """Clear messages from a thread (optionally keeping summaries)."""
    # Returns: {cleared_count, summaries_kept}

@mcp_tool
def conversation_search(
    query: str,
    thread_uid: str = None,  # None = search all threads
    limit: int = 10,
    min_similarity: float = 0.5,
    include_summaries: bool = True
) -> dict:
    """Semantic search across conversation messages."""
    # Returns: {results: [{message_uid, thread_uid, role, content, similarity}]}
```

#### Context Composition (2 tools)

```python
@mcp_tool
def conversation_compose_context(
    thread_uid: str,
    user_input: str,
    max_tokens: int = 4096,
    include_semantic: bool = True,
    include_summaries: bool = True,
    include_memory: bool = True,  # Include relevant MG observations
    semantic_limit: int = 3
) -> dict:
    """Compose prompt context from thread history + semantic search + memory."""
    # Returns: {
    #   messages: [{role, content}],  # Ready for LLM
    #   token_count: int,
    #   sources: {history: int, semantic: int, summaries: int, memory: int}
    # }

@mcp_tool
def conversation_summarize(
    thread_uid: str,
    preserve_last_n: int = 20,
    model: str = "auto"  # auto, local (LoomMirror), or specific model
) -> dict:
    """Summarize older messages in a thread."""
    # Returns: {summary_id, messages_summarized, summary_text}
```

#### Embedding Utilities (2 tools)

```python
@mcp_tool
def conversation_backfill_embeddings(
    thread_uid: str = None,  # None = all threads
    batch_size: int = 50
) -> dict:
    """Generate embeddings for messages missing them."""
    # Returns: {processed_count, skipped_count}

@mcp_tool
def conversation_extract_memory(
    thread_uid: str,
    message_range: tuple = None,  # (start_seq, end_seq) or None for all
    extract_observations: bool = True,
    extract_concepts: bool = False
) -> dict:
    """Extract observations/concepts from conversation into MG memory."""
    # Returns: {observations_created: int, concepts_created: int, refs: []}
```

### 3.2 Tool Organization

```
MCP Tools (53 total = 41 existing + 12 new)
├── Session & Metadata (3)
├── Storage (4)
├── Retrieval (8)
├── Knowledge Graph (3)
├── Generic Relationships (4)
├── Residue Tracking (2)
├── Chains (6)
├── Cold Storage & Retention (4)
├── Diagnostics (5)
├── Agent Identity (1)
├── Store Management (3)
└── Conversation (12) ← NEW
    ├── Thread Management (4)
    ├── Message Operations (4)
    ├── Context Composition (2)
    └── Embedding Utilities (2)
```

---

## 4. Service Layer

### 4.1 New Service Module

```
core/services/
├── ... (existing)
└── conversation_service.py  ← NEW (estimated 400-600 lines)
```

#### Key Functions

```python
# Thread management
async def create_thread(ctx, thread_name, ai_name, ai_platform, agent_uuid, metadata) -> dict
async def list_threads(ctx, include_archived, ai_instance_id, limit) -> list
async def get_thread(ctx, thread_uid) -> dict
async def archive_thread(ctx, thread_uid, reason) -> dict
async def update_thread_activity(ctx, thread_id) -> None

# Message operations
async def append_message(ctx, thread_uid, role, content, message_type, metadata, generate_embedding) -> dict
async def recall_messages(ctx, thread_uid, limit, offset, include_summaries) -> dict
async def clear_messages(ctx, thread_uid, keep_summaries) -> dict
async def search_messages(ctx, query, thread_uid, limit, min_similarity, include_summaries) -> list

# Context composition
async def compose_context(ctx, thread_uid, user_input, max_tokens, include_semantic, include_summaries, include_memory, semantic_limit) -> dict
async def count_tokens(text: str) -> int  # tiktoken-based

# Summarization
async def summarize_thread(ctx, thread_uid, preserve_last_n, model) -> dict
async def generate_summary_text(messages: list, model: str) -> str

# Memory extraction
async def extract_memory(ctx, thread_uid, message_range, extract_observations, extract_concepts) -> dict

# Embedding utilities
async def backfill_embeddings(ctx, thread_uid, batch_size) -> dict
async def generate_message_embedding(ctx, message_id, content) -> None
```

### 4.2 Integration with Existing Services

```python
# In memory_search.py - extend to include conversation results
async def unified_search(ctx, query, include_conversations=False, ...) -> dict

# In memory_storage.py - link observations to conversation source
async def store_observation(..., source_thread_uid=None, source_message_uid=None) -> dict

# In retention_service.py - include conversation threads in decay
async def retention_tick():
    # ... existing logic ...
    await decay_conversation_scores(ctx)
```

---

## 5. Summarization Module

### 5.1 LoomMirror Integration

Port LoomMirror as an optional local summarization backend.

```
core/summarization/
├── __init__.py
├── base.py           # Abstract summarizer interface
├── openai_summarizer.py   # OpenAI-based (default)
├── local_summarizer.py    # LoomMirror port (llama-cpp-python)
└── config.py         # Model paths, parameters
```

#### Interface

```python
class BaseSummarizer(ABC):
    @abstractmethod
    async def summarize_messages(self, messages: list[dict]) -> str:
        """Summarize a list of messages into a concise summary."""
        pass

    @abstractmethod
    async def extract_facts(self, messages: list[dict]) -> list[str]:
        """Extract factual statements from messages."""
        pass

    @abstractmethod
    async def tag_messages(self, messages: list[dict]) -> list[str]:
        """Generate topic tags for messages."""
        pass
```

#### Local Summarizer (LoomMirror port)

```python
class LocalSummarizer(BaseSummarizer):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1)

    async def summarize_messages(self, messages: list[dict]) -> str:
        prompt = self._build_summary_prompt(messages)
        return await asyncio.to_thread(self._generate, prompt, max_tokens=200)

    # ... etc
```

### 5.2 Configuration

```python
# core/config.py additions
SUMMARIZATION_BACKEND: str = "openai"  # openai, local, none
LOCAL_SUMMARIZER_MODEL_PATH: str = None
LOCAL_SUMMARIZER_N_CTX: int = 2048
SUMMARIZE_THRESHOLD_MESSAGES: int = 50  # Auto-summarize when thread exceeds
SUMMARIZE_PRESERVE_LAST_N: int = 20
```

---

## 6. Context Composition Algorithm

### 6.1 Token Budget Allocation

```python
async def compose_context(
    ctx,
    thread_uid: str,
    user_input: str,
    max_tokens: int = 4096,
    include_semantic: bool = True,
    include_summaries: bool = True,
    include_memory: bool = True,
    semantic_limit: int = 3
) -> dict:
    """
    Compose context with intelligent token budgeting.

    Budget allocation (configurable):
    - 40% Recent history (most recent messages)
    - 20% Summaries (compressed older context)
    - 20% Semantic matches (relevant past messages)
    - 20% Memory observations (MG knowledge)
    """
    budget = TokenBudget(max_tokens)

    # 1. Recent history (always included, highest priority)
    recent = await recall_messages(ctx, thread_uid, limit=100)
    recent_messages, recent_tokens = budget.allocate(recent, ratio=0.4)

    # 2. Summaries (if enabled)
    summaries = []
    if include_summaries:
        summaries = await get_thread_summaries(ctx, thread_uid)
        summary_messages, summary_tokens = budget.allocate(summaries, ratio=0.2)

    # 3. Semantic search (if enabled)
    semantic = []
    if include_semantic:
        semantic = await search_messages(ctx, user_input, thread_uid, limit=semantic_limit)
        # Dedupe against recent
        semantic = [m for m in semantic if m['message_uid'] not in recent_uids]
        semantic_messages, semantic_tokens = budget.allocate(semantic, ratio=0.2)

    # 4. Memory observations (if enabled)
    memory = []
    if include_memory:
        memory = await memory_search(ctx, user_input, limit=5)
        memory_messages = format_as_system_context(memory)
        memory_tokens = budget.allocate(memory_messages, ratio=0.2)

    # 5. Merge and format
    context = merge_context(
        recent=recent_messages,
        summaries=summary_messages,
        semantic=semantic_messages,
        memory=memory_messages
    )

    return {
        "messages": context,
        "token_count": budget.used,
        "sources": {
            "history": len(recent_messages),
            "summaries": len(summary_messages),
            "semantic": len(semantic_messages),
            "memory": len(memory_messages)
        }
    }
```

### 6.2 Context Merge Strategy

```
Final context structure:
┌─────────────────────────────────────────┐
│ [system] Memory context (observations)   │ ← MG knowledge
├─────────────────────────────────────────┤
│ [system] Thread summaries               │ ← Compressed history
├─────────────────────────────────────────┤
│ [system] Relevant past messages         │ ← Semantic matches
├─────────────────────────────────────────┤
│ ... recent conversation messages ...    │ ← Thread history
│ [user] message N-2                      │
│ [assistant] message N-1                 │
│ [user] current input                    │ ← Latest
└─────────────────────────────────────────┘
```

---

## 7. Migration Path

### 7.1 Database Migration

```python
# alembic/versions/xxx_add_conversation_tables.py

def upgrade():
    # Create conversation_threads
    op.create_table('conversation_threads', ...)

    # Create conversation_messages
    op.create_table('conversation_messages', ...)

    # Create conversation_summaries
    op.create_table('conversation_summaries', ...)

    # Extend embeddings source_type enum
    op.execute("ALTER TYPE source_type ADD VALUE 'message'")
    op.execute("ALTER TYPE source_type ADD VALUE 'summary'")
    op.execute("ALTER TYPE source_type ADD VALUE 'fact'")

def downgrade():
    op.drop_table('conversation_summaries')
    op.drop_table('conversation_messages')
    op.drop_table('conversation_threads')
```

### 7.2 Loom Data Import (Optional)

```python
async def import_loom_data(
    loom_db_url: str,
    target_ai_name: str = "Imported",
    target_ai_platform: str = "Loom"
) -> dict:
    """
    Import existing Loom data into MemoryGate conversation tables.

    Imports:
    - threads → conversation_threads
    - messages → conversation_messages
    - message_embeddings → embeddings (source_type='message')
    - summaries → conversation_summaries
    """
    # Implementation details...
    return {
        "threads_imported": n,
        "messages_imported": m,
        "embeddings_imported": e,
        "summaries_imported": s
    }
```

---

## 8. Configuration

### 8.1 New Environment Variables

```bash
# Conversation module
CONVERSATION_ENABLED=true
CONVERSATION_MAX_THREAD_MESSAGES=10000
CONVERSATION_AUTO_SUMMARIZE=true
CONVERSATION_SUMMARIZE_THRESHOLD=50
CONVERSATION_PRESERVE_LAST_N=20

# Summarization
SUMMARIZATION_BACKEND=openai  # openai, local, none
LOCAL_SUMMARIZER_MODEL_PATH=/path/to/model.gguf
LOCAL_SUMMARIZER_N_CTX=2048

# Context composition
CONTEXT_MAX_TOKENS=4096
CONTEXT_HISTORY_RATIO=0.4
CONTEXT_SUMMARY_RATIO=0.2
CONTEXT_SEMANTIC_RATIO=0.2
CONTEXT_MEMORY_RATIO=0.2

# Memory extraction
AUTO_EXTRACT_OBSERVATIONS=false
EXTRACTION_CONFIDENCE_THRESHOLD=0.7
```

### 8.2 Feature Flags

```python
# core/config.py
class ConversationConfig:
    enabled: bool = True
    max_thread_messages: int = 10000
    auto_summarize: bool = True
    summarize_threshold: int = 50
    preserve_last_n: int = 20

class SummarizationConfig:
    backend: str = "openai"  # openai, local, none
    local_model_path: str = None
    local_n_ctx: int = 2048

class ContextConfig:
    max_tokens: int = 4096
    history_ratio: float = 0.4
    summary_ratio: float = 0.2
    semantic_ratio: float = 0.2
    memory_ratio: float = 0.2
```

---

## 9. API Examples

### 9.1 Basic Conversation Flow

```python
# 1. Create thread
result = await conversation_create_thread(
    thread_name="Code Review Session",
    ai_name="Claude",
    ai_platform="Claude CLI"
)
thread_uid = result["thread_uid"]

# 2. Append messages
await conversation_append(thread_uid, "user", "Review this Python function...")
await conversation_append(thread_uid, "assistant", "I see several issues...")

# 3. Compose context for next turn
context = await conversation_compose_context(
    thread_uid=thread_uid,
    user_input="What about error handling?",
    include_memory=True  # Include relevant MG observations
)

# 4. Send to LLM
response = await llm.complete(context["messages"])

# 5. Store response
await conversation_append(thread_uid, "assistant", response)
```

### 9.2 Cross-Pollination: Conversation → Memory

```python
# Extract observations from conversation
result = await conversation_extract_memory(
    thread_uid=thread_uid,
    extract_observations=True,
    extract_concepts=True
)
# Returns: {observations_created: 3, concepts_created: 1, refs: ["observation:123", ...]}

# Later searches will find these observations
results = await memory_search("Python error handling patterns")
# May return observations extracted from the code review conversation
```

### 9.3 Cross-Pollination: Memory → Conversation

```python
# Context composition automatically includes relevant memories
context = await conversation_compose_context(
    thread_uid=thread_uid,
    user_input="How should I handle database connections?",
    include_memory=True
)
# context["sources"]["memory"] > 0 if relevant observations found
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```
tests/
├── test_conversation_service.py
│   ├── test_create_thread
│   ├── test_append_message
│   ├── test_recall_messages
│   ├── test_search_messages
│   ├── test_compose_context
│   └── test_summarize_thread
├── test_summarization.py
│   ├── test_openai_summarizer
│   └── test_local_summarizer
└── test_context_composition.py
    ├── test_token_budgeting
    ├── test_context_merge
    └── test_memory_inclusion
```

### 10.2 Integration Tests

```python
async def test_full_conversation_flow():
    # Create thread, append messages, search, compose context, summarize
    pass

async def test_memory_extraction():
    # Conversation → observations → searchable
    pass

async def test_memory_injection():
    # Store observations → conversation context includes them
    pass
```

---

## 11. Performance Considerations

### 11.1 Indexing

```sql
-- Thread lookup
CREATE INDEX idx_conv_threads_tenant_active ON conversation_threads(tenant_id, is_active);
CREATE INDEX idx_conv_threads_ai_instance ON conversation_threads(ai_instance_id);

-- Message retrieval (primary access pattern)
CREATE INDEX idx_conv_messages_thread_seq ON conversation_messages(thread_id, seq DESC);

-- Semantic search (via embeddings table)
-- Already indexed by (source_type, source_id)
```

### 11.2 Caching

```python
# Cache frequently accessed thread metadata
@lru_cache(maxsize=1000)
async def get_thread_cached(thread_uid: str) -> dict:
    pass

# Cache token counts for messages
@lru_cache(maxsize=10000)
def count_tokens_cached(text: str) -> int:
    pass
```

### 11.3 Batch Operations

```python
# Batch embedding generation
async def backfill_embeddings_batch(message_ids: list[int], batch_size: int = 50):
    for batch in chunked(message_ids, batch_size):
        texts = [get_message_content(id) for id in batch]
        embeddings = await embed_texts_batch(texts)  # Single API call
        await store_embeddings_batch(batch, embeddings)
```

---

## 12. Future Enhancements

### 12.1 Phase 2: Advanced Features

1. **Conversation forking**: Branch threads for "what if" explorations
2. **Thread merging**: Combine related conversations
3. **Conversation templates**: Pre-defined thread structures
4. **Real-time streaming**: WebSocket-based message streaming
5. **Collaborative threads**: Multi-agent conversation support

### 12.2 Phase 3: Intelligence Features

1. **Auto-tagging**: Automatic topic classification
2. **Sentiment tracking**: Conversation tone analysis
3. **Intent detection**: Classify user intents per message
4. **Conversation clustering**: Group similar threads
5. **Predictive context**: Pre-fetch likely relevant memories

---

## 13. Appendix

### A. Loom Source Files Reference

| Loom File | MemoryGate Equivalent |
|-----------|----------------------|
| `loom/__init__.py` | `core/services/conversation_service.py` |
| `loom/models.py` | `core/models.py` (extended) |
| `loom/db.py` | `core/database.py` (existing) |
| `loom/embeddings.py` | `core/services/memory_embeddings.py` (existing) |
| `loom/LoomMirror/` | `core/summarization/local_summarizer.py` |

### B. MCP Tool Mapping

| Loom Method | MCP Tool |
|-------------|----------|
| `create_new_thread()` | `conversation_create_thread` |
| `list_all_threads()` | `conversation_list_threads` |
| `append()` / `append_async()` | `conversation_append` |
| `recall()` / `recall_async()` | `conversation_recall` |
| `clear()` | `conversation_clear` |
| `semantic_search()` | `conversation_search` |
| `compose_prompt_context_async()` | `conversation_compose_context` |
| `recall_with_summary_async()` | `conversation_summarize` |
| `backfill_embeddings()` | `conversation_backfill_embeddings` |

---

## 14. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-28 | Cathedral Dev | Initial specification |

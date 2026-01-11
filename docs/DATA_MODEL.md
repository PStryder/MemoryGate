# Data Model

MemoryGate stores structured memory in SQL tables defined in core/models.py and
oauth_models.py. SCHEMA.md contains the full schema reference.

## Core Memory Tables

- ai_instances: AI identity registry (name, platform).
- sessions: Conversation metadata and provenance.
- observations: Facts with confidence, domain, and evidence.
- patterns: Synthesized insights (upserted by category + name).
- concepts: Knowledge graph nodes with metadata and aliases.
- concept_aliases: Alternate names for concepts.
- concept_relationships: Graph edges with rel_type and weight.
- documents: External references with summary and URL.
- embeddings: Unified vector storage for semantic search.
- memory_summaries: Retention summaries for cold-tier records.
- memory_tombstones: Archive/rehydrate/purge audit trail.
- archived_memories: Archived payload store for purge workflows.
- audit_events: Metadata-only audit logs.

## Auth Tables

- users: OAuth user identities.
- oauth_states: Temporary OAuth state records.
- oauth_authorization_codes: PKCE authorization codes for MCP clients.
- user_sessions: Session tokens for logged-in users.
- api_keys: API keys linked to users.

## Tiering

Most memory tables include:

- tier: hot or cold
- score: retention score
- archived_at, archived_reason, archived_by
- purge_eligible and floor_score

These fields drive retention, archive, and rehydration workflows.

For full column and index details, see SCHEMA.md.

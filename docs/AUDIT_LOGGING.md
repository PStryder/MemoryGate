# Audit Logging (DB-Only, Metadata-Only)

MemoryGate stores audit events in the database for standalone and SaaS compatibility.

Principles:

- **DB-only:** events are stored in `audit_events` (no external log aggregation).
- **Append-only:** events are never updated or deleted by core logic.
- **Metadata-only:** no content fields, embeddings, or payload bodies are allowed.
- **Scoped:** `org_id` and `user_id` are present for SaaS alignment (nullable in standalone).

Fields (summary):

- `event_type` is a stable string (e.g., `memory.archived`, `memory.rehydrated`).
- `actor_type` is one of `user`, `org_admin`, `system`, `integration`, `mcp`.
- `target_type` is one of `memory`, `summary`, `account`, `org`, `key`, `export`.
- `target_ids` contains IDs only (never content).

Metadata safety:

- Disallowed metadata keys: `content`, `observation`, `description`, `pattern_text`,
  `embedding`, `summary_text`, `document_body`, `raw_text`.
- String metadata values are capped to prevent accidental content logging.

SaaS usage notes:

- SaaS can scope queries via `org_id` and `user_id` when multi-tenant context is available.
- End-user or org-admin views can read from `audit_events` without exposing memory content.
- Exports should include audit events only as metadata, never full payloads.

# Archive Retention and Durability

Archived memories are stored in the database table `archived_memories`. The archive store is
reversible until quota eviction occurs.

## Durability

MemoryGate does not replicate archive payloads to external storage by default. For production:

- Back up the database regularly.
- If you require durability beyond the primary database, replicate `archived_memories`
  to external storage (e.g., S3) via your infrastructure tooling.

## Quota and eviction

- Effective archive capacity = `STORAGE_QUOTA_BYTES * ARCHIVE_MULTIPLIER`.
- When the archive exceeds capacity, the oldest archived records are evicted first.
- Evictions are permanent and irreversible.

Audit event emitted on eviction:

- `retention.archive_evicted` (metadata only)
- Metadata includes bytes before/after and the oldest/newest eviction timestamps.

## Size estimation

Archive size is a conservative estimate based on text fields + metadata size + embedding
estimate (when vector search is enabled).

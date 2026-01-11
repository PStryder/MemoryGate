"""
Canonical audit event type strings (shared contract with SaaS).
"""

EVENT_MEMORY_ARCHIVED = "memory.archived"
EVENT_MEMORY_REHYDRATED = "memory.rehydrated"
EVENT_MEMORY_PURGED_TO_ARCHIVE = "memory.purged_to_archive"
EVENT_MEMORY_RESTORED_FROM_ARCHIVE = "memory.restored_from_archive"
EVENT_RETENTION_ARCHIVE_EVICTED = "retention.archive_evicted"
EVENT_EXPORT_STARTED = "export.started"
EVENT_EXPORT_COMPLETED = "export.completed"

__all__ = [
    "EVENT_MEMORY_ARCHIVED",
    "EVENT_MEMORY_REHYDRATED",
    "EVENT_MEMORY_PURGED_TO_ARCHIVE",
    "EVENT_MEMORY_RESTORED_FROM_ARCHIVE",
    "EVENT_RETENTION_ARCHIVE_EVICTED",
    "EVENT_EXPORT_STARTED",
    "EVENT_EXPORT_COMPLETED",
]

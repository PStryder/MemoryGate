import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")

from core.audit_constants import (
    EVENT_MEMORY_ARCHIVED,
    EVENT_MEMORY_PURGED_TO_ARCHIVE,
    EVENT_MEMORY_REHYDRATED,
    EVENT_MEMORY_RESTORED_FROM_ARCHIVE,
)
from core.models import AuditEvent, MemoryTombstone, TombstoneAction
from core.services import memory as memory_service


FORBIDDEN_METADATA_KEYS = {
    "content",
    "observation",
    "description",
    "pattern_text",
    "embedding",
    "summary_text",
    "document_body",
    "raw_text",
}


def test_restore_to_cold_requires_rehydrate_for_search(server_db, db_session):
    store_result = memory_service.memory_store(
        observation="Restore semantics observation",
        confidence=0.9,
        domain="restore_semantics",
    )
    memory_id = f"observation:{store_result['id']}"

    archive_result = memory_service.archive_memory(
        memory_ids=[memory_id],
        reason="restore semantics archive",
        dry_run=False,
    )
    assert archive_result["archived_count"] == 1

    purge_result = memory_service.purge_memory_to_archive(
        memory_ids=[memory_id],
        reason="restore semantics purge",
        dry_run=False,
    )
    assert purge_result["affected"] == 1

    restore_result = memory_service.restore_archived_memory(
        memory_ids=[memory_id],
        target_tier="cold",
        reason="restore semantics restore",
        dry_run=False,
    )
    assert restore_result["affected"] == 1

    search_hot = memory_service.memory_search(
        query="Restore semantics observation",
        limit=5,
    )
    assert search_hot["count"] == 0

    recall_hot = memory_service.memory_recall(domain="restore_semantics", include_cold=False)
    assert recall_hot["count"] == 0

    rehydrate_result = memory_service.rehydrate_memory(
        memory_ids=[memory_id],
        reason="restore semantics rehydrate",
        dry_run=False,
    )
    assert rehydrate_result["rehydrated_count"] == 1

    search_after = memory_service.memory_search(
        query="Restore semantics observation",
        limit=5,
    )
    assert search_after["count"] == 1

    recall_after = memory_service.memory_recall(domain="restore_semantics", include_cold=False)
    assert recall_after["count"] == 1

    db_session.expire_all()
    tombstones = db_session.query(MemoryTombstone).all()
    action_counts = {}
    for tombstone in tombstones:
        action_counts[tombstone.action] = action_counts.get(tombstone.action, 0) + 1

    assert action_counts.get(TombstoneAction.archived, 0) >= 1
    assert action_counts.get(TombstoneAction.purged, 0) >= 1
    assert action_counts.get(TombstoneAction.rehydrated, 0) >= 2

    events = db_session.query(AuditEvent).all()
    event_types = {event.event_type for event in events}
    assert EVENT_MEMORY_ARCHIVED in event_types
    assert EVENT_MEMORY_PURGED_TO_ARCHIVE in event_types
    assert EVENT_MEMORY_RESTORED_FROM_ARCHIVE in event_types
    assert EVENT_MEMORY_REHYDRATED in event_types

    for event in events:
        metadata = event.metadata_ or {}
        for key in metadata.keys():
            assert key not in FORBIDDEN_METADATA_KEYS

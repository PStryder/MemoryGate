import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")

from core.models import MemoryTombstone, Observation, MemoryTier
from core.services.memory_service import archive_memory, memory_recall, memory_store, rehydrate_memory


def test_archive_rehydrate_lifecycle(server_db, db_session):
    store_result = memory_store(
        observation="Archive lifecycle observation",
        confidence=0.9,
        domain="archive_test",
    )
    memory_id = f"observation:{store_result['id']}"

    archive_result = archive_memory(
        memory_ids=[memory_id],
        reason="lifecycle test",
        dry_run=False,
    )
    assert archive_result["status"] == "archived"
    assert archive_result["archived_count"] == 1
    assert memory_id in archive_result["archived_ids"]
    assert archive_result["already_archived_count"] == 0
    assert archive_result["tombstones_written"] == 1

    db_session.expire_all()
    record = db_session.get(Observation, store_result["id"])
    assert record is not None
    assert record.tier == MemoryTier.cold
    assert record.archived_at is not None

    recall_hot = memory_recall(domain="archive_test", include_cold=False)
    assert recall_hot["count"] == 0
    recall_cold = memory_recall(domain="archive_test", include_cold=True)
    assert recall_cold["count"] == 1

    repeat_archive = archive_memory(
        memory_ids=[memory_id],
        reason="lifecycle test",
        dry_run=False,
    )
    assert repeat_archive["archived_count"] == 0
    assert repeat_archive["already_archived_count"] == 1
    assert repeat_archive["tombstones_written"] == 0

    rehydrate_result = rehydrate_memory(
        memory_ids=[memory_id],
        reason="lifecycle test",
        dry_run=False,
    )
    assert rehydrate_result["status"] == "rehydrated"
    assert rehydrate_result["rehydrated_count"] == 1
    assert memory_id in rehydrate_result["rehydrated_ids"]
    assert rehydrate_result["already_hot_count"] == 0
    assert rehydrate_result["tombstones_written"] == 1

    db_session.expire_all()
    record = db_session.get(Observation, store_result["id"])
    assert record is not None
    assert record.tier == MemoryTier.hot
    assert record.archived_at is None

    recall_hot = memory_recall(domain="archive_test", include_cold=False)
    assert recall_hot["count"] == 1

    repeat_rehydrate = rehydrate_memory(
        memory_ids=[memory_id],
        reason="lifecycle test",
        dry_run=False,
    )
    assert repeat_rehydrate["rehydrated_count"] == 0
    assert repeat_rehydrate["already_hot_count"] == 1
    assert repeat_rehydrate["tombstones_written"] == 0

    db_session.expire_all()
    tombstones = db_session.query(MemoryTombstone).all()
    assert len(tombstones) == 2

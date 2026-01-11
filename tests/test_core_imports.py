import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")


def test_core_imports():
    import core.context  # noqa: F401
    import core.models  # noqa: F401
    import core.services.memory  # noqa: F401


def test_core_smoke_lifecycle(server_db):
    import core.services.memory as memory

    store_result = memory.memory_store(
        observation="Core import smoke observation",
        confidence=0.9,
        domain="core_smoke",
    )
    memory_id = f"observation:{store_result['id']}"

    search_result = memory.memory_search(
        query="Core import smoke observation",
        limit=5,
    )
    assert search_result["count"] == 1

    archive_result = memory.archive_memory(
        memory_ids=[memory_id],
        reason="core smoke archive",
        dry_run=False,
    )
    assert archive_result["archived_count"] == 1

    search_hot = memory.memory_search(
        query="Core import smoke observation",
        limit=5,
        include_cold=False,
    )
    assert search_hot["count"] == 0

    purge_result = memory.purge_memory_to_archive(
        memory_ids=[memory_id],
        reason="core smoke purge",
        dry_run=False,
    )
    assert purge_result["affected"] == 1

    restore_result = memory.restore_archived_memory(
        memory_ids=[memory_id],
        target_tier="hot",
        reason="core smoke restore",
        dry_run=False,
    )
    assert restore_result["affected"] == 1

    search_after = memory.memory_search(
        query="Core import smoke observation",
        limit=5,
    )
    assert search_after["count"] == 1

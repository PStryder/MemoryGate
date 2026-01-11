import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")

from core.services import memory as memory_service


def test_service_tools_smoke(server_db):
    session = memory_service.memory_init_session(
        conversation_id="conv-smoke-1",
        title="Smoke Test",
        ai_name="Kee",
        ai_platform="Claude",
    )
    assert session["status"] == "initialized"

    obs = memory_service.memory_store(
        observation="Service tool smoke observation",
        confidence=0.9,
        domain="smoke",
        evidence=["evidence line"],
        ai_name="Kee",
        ai_platform="Claude",
        conversation_id="conv-smoke-1",
        conversation_title="Smoke Test",
    )
    assert obs["status"] == "stored"

    doc = memory_service.memory_store_document(
        title="Smoke Doc",
        doc_type="article",
        url="https://example.com/doc",
        content_summary="Document summary for smoke test",
        key_concepts=["memory"],
    )
    assert doc["status"] == "stored"

    concept_a = memory_service.memory_store_concept(
        name="SmokeConceptA",
        concept_type="project",
        description="Concept A description",
    )
    assert concept_a["status"] == "stored"

    concept_b = memory_service.memory_store_concept(
        name="SmokeConceptB",
        concept_type="project",
        description="Concept B description",
    )
    assert concept_b["status"] == "stored"

    alias = memory_service.memory_add_concept_alias(
        concept_name="SmokeConceptA",
        alias="SmokeAliasA",
    )
    assert alias["status"] == "created"

    relationship = memory_service.memory_add_concept_relationship(
        from_concept="SmokeConceptA",
        to_concept="SmokeConceptB",
        rel_type="related_to",
        weight=0.7,
        description="Smoke relationship",
    )
    assert relationship["status"] in {"created", "updated"}

    related = memory_service.memory_related_concepts(
        concept_name="SmokeConceptA",
        include_cold=True,
    )
    assert related["status"] == "found"

    concept_lookup = memory_service.memory_get_concept(
        name="SmokeAliasA",
        include_cold=True,
    )
    assert concept_lookup.get("name") == "SmokeConceptA"

    pattern = memory_service.memory_update_pattern(
        category="smoke",
        pattern_name="smoke_pattern",
        pattern_text="Pattern text",
        evidence_observation_ids=[obs["id"]],
    )
    assert pattern["status"] in {"created", "updated"}

    fetched_pattern = memory_service.memory_get_pattern(
        category="smoke",
        pattern_name="smoke_pattern",
        include_cold=True,
    )
    assert fetched_pattern["status"] == "found"

    pattern_list = memory_service.memory_patterns(
        category="smoke",
        include_cold=True,
    )
    assert pattern_list["status"] == "ok"

    guide = memory_service.memory_user_guide(format="markdown", verbosity="short")
    assert guide["status"] == "ok"

    bootstrap = memory_service.memory_bootstrap(ai_name="Kee", ai_platform="Claude")
    assert bootstrap["status"] == "ok"

    search = memory_service.memory_search(
        query="Service tool smoke observation",
        limit=5,
    )
    assert search["count"] >= 1

    recall = memory_service.memory_recall(domain="smoke", include_cold=True)
    assert recall["count"] >= 1

    archive = memory_service.archive_memory(
        memory_ids=[f"observation:{obs['id']}"],
        reason="smoke archive",
        dry_run=False,
    )
    assert archive["archived_count"] == 1

    candidates = memory_service.list_archive_candidates(limit=10)
    assert candidates["status"] == "ok"

    rehydrate = memory_service.rehydrate_memory(
        memory_ids=[f"observation:{obs['id']}"],
        reason="smoke rehydrate",
        dry_run=False,
    )
    assert rehydrate["rehydrated_count"] == 1

    archive_again = memory_service.archive_memory(
        memory_ids=[f"observation:{obs['id']}"],
        reason="smoke archive again",
        dry_run=False,
    )
    assert archive_again["archived_count"] == 1

    purge = memory_service.purge_memory_to_archive(
        memory_ids=[f"observation:{obs['id']}"],
        reason="smoke purge",
        dry_run=False,
    )
    assert purge["affected"] == 1

    archived_list = memory_service.list_archived_memories(limit=10)
    assert archived_list["status"] == "ok"

    restore = memory_service.restore_archived_memory(
        memory_ids=[f"observation:{obs['id']}"],
        target_tier="hot",
        reason="smoke restore",
        dry_run=False,
    )
    assert restore["affected"] == 1

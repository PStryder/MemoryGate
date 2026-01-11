import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")

from core.services.memory_service import (
    memory_add_concept_alias,
    memory_add_concept_relationship,
    memory_related_concepts,
    memory_store_concept,
)


def test_memory_related_concepts_resolves_alias(server_db):
    concept_a = memory_store_concept(
        name="AliasConceptA",
        concept_type="project",
        description="Concept A",
    )
    concept_b = memory_store_concept(
        name="AliasConceptB",
        concept_type="project",
        description="Concept B",
    )
    assert concept_a["status"] == "stored"
    assert concept_b["status"] == "stored"

    alias_result = memory_add_concept_alias(
        concept_name="AliasConceptA",
        alias="AliasConceptAAlt",
    )
    assert alias_result["status"] == "created"

    rel_result = memory_add_concept_relationship(
        from_concept="AliasConceptA",
        to_concept="AliasConceptB",
        rel_type="related_to",
        weight=0.7,
    )
    assert rel_result["status"] in {"created", "updated"}

    primary = memory_related_concepts("AliasConceptA")
    alias = memory_related_concepts("AliasConceptAAlt")

    assert primary["status"] == "found"
    assert alias["status"] == "found"
    assert primary["outgoing"] == alias["outgoing"]
    assert primary["incoming"] == alias["incoming"]

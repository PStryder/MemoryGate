import os

import pytest

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")

from core.models import Observation
from core.config import (
    MAX_METADATA_BYTES,
    MAX_RELATIONSHIP_ITEMS,
    MAX_SHORT_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
)
from core.services.memory_service import memory_store, memory_store_concept, memory_update_pattern


def _create_observations(db_session, count: int) -> list[int]:
    observations = []
    for index in range(count):
        obs = Observation(
            observation=f"Evidence {index}",
            confidence=0.9,
            domain="pattern_test",
        )
        db_session.add(obs)
        observations.append(obs)
    db_session.commit()
    for obs in observations:
        db_session.refresh(obs)
    return [obs.id for obs in observations]


def test_memory_update_pattern_rejects_negative_evidence_ids(server_db):
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="invalid_evidence_negative",
        pattern_text="Invalid evidence test",
        evidence_observation_ids=[-1],
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "evidence_observation_ids"


def test_memory_update_pattern_rejects_nonexistent_evidence_ids(server_db):
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="invalid_evidence_missing",
        pattern_text="Invalid evidence test",
        evidence_observation_ids=[999999],
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "evidence_observation_ids"


def test_memory_update_pattern_accepts_max_evidence_ids(server_db, db_session):
    evidence_ids = _create_observations(db_session, MAX_RELATIONSHIP_ITEMS)
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="max_evidence",
        pattern_text="Max evidence list test",
        evidence_observation_ids=evidence_ids,
    )
    assert result["status"] == "created"


def test_memory_update_pattern_rejects_overflow_evidence_ids(server_db):
    evidence_ids = list(range(1, MAX_RELATIONSHIP_ITEMS + 2))
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="overflow_evidence",
        pattern_text="Overflow evidence list test",
        evidence_observation_ids=evidence_ids,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "evidence_observation_ids"


@pytest.mark.parametrize("value", ["", "   "])
def test_memory_update_pattern_rejects_empty_strings(server_db, value):
    result = memory_update_pattern(
        category=value,
        pattern_name="pattern_name",
        pattern_text="pattern_text",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "category"


def test_memory_update_pattern_rejects_overlong_fields(server_db):
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="x" * (MAX_SHORT_TEXT_LENGTH + 1),
        pattern_text="pattern_text",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "pattern_name"


def test_memory_update_pattern_allows_unicode(server_db):
    result = memory_update_pattern(
        category="pattern_test",
        pattern_name="café_pattern",
        pattern_text="naïve pattern text",
    )
    assert result["status"] == "created"


@pytest.mark.parametrize("value", ["", "   "])
def test_memory_store_concept_rejects_empty_name(server_db, value):
    result = memory_store_concept(
        name=value,
        concept_type="project",
        description="Test description",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "name"


def test_memory_store_concept_rejects_overlong_description(server_db):
    result = memory_store_concept(
        name="ConceptLongDesc",
        concept_type="project",
        description="x" * (MAX_TEXT_LENGTH + 1),
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "description"


def test_memory_store_concept_rejects_overlong_name(server_db):
    result = memory_store_concept(
        name="x" * (MAX_SHORT_TEXT_LENGTH + 1),
        concept_type="project",
        description="Description",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "name"


def test_memory_store_concept_rejects_metadata_overflow(server_db):
    oversized = "x" * (MAX_METADATA_BYTES + 50)
    result = memory_store_concept(
        name="ConceptMetaOverflow",
        concept_type="project",
        description="Description",
        metadata={"blob": oversized},
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "metadata"


def test_memory_store_concept_allows_unicode(server_db):
    result = memory_store_concept(
        name="CaféConcept",
        concept_type="project",
        description="Unicode description",
    )
    assert result["status"] == "stored"


def test_memory_store_rejects_missing_evidence_ids(server_db):
    result = memory_store(
        observation="Evidence validation missing",
        evidence=["999999"],
        confidence=0.9,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert result["field"] == "evidence_observation_ids"


def test_memory_store_accepts_existing_evidence_ids(server_db):
    seed = memory_store(
        observation="Evidence validation seed",
        confidence=0.9,
    )
    result = memory_store(
        observation="Evidence validation valid",
        evidence=[str(seed["id"])],
        confidence=0.9,
    )
    assert result["status"] == "stored"

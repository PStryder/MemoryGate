"""
Quota enforcement and usage estimation helpers.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import func

import core.config as config
from core.context import require_tenant_id_value, resolve_tenant_id, RequestContext
from core.db import DB
from core.errors import ValidationIssue
from core.models import (
    Concept,
    Document,
    Embedding,
    MemorySummary,
    Observation,
    Pattern,
    TenantStorageUsage,
)


def _text_bytes(value: Optional[str]) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = str(value)
    return len(value.encode("utf-8"))


def _json_bytes(value) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _embedding_bytes_per_row() -> int:
    if config.DB_BACKEND_EFFECTIVE != "postgres":
        return 0
    if config.VECTOR_BACKEND_EFFECTIVE != "pgvector":
        return 0
    return int(config.EMBEDDING_DIM) * 4


def estimate_embedding_bytes() -> int:
    return _embedding_bytes_per_row()


def estimate_observation_bytes(
    observation: str,
    *,
    domain: Optional[str] = None,
    evidence: Optional[list[str]] = None,
) -> int:
    return (
        _text_bytes(observation)
        + _text_bytes(domain)
        + _json_bytes(evidence or [])
    )


def estimate_document_bytes(
    *,
    title: str,
    doc_type: str,
    url: str,
    content_summary: str,
    key_concepts: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> int:
    return (
        _text_bytes(title)
        + _text_bytes(doc_type)
        + _text_bytes(url)
        + _text_bytes(content_summary)
        + _json_bytes(key_concepts or [])
        + _json_bytes(metadata or {})
    )


def estimate_concept_bytes(
    *,
    name: str,
    concept_type: str,
    description: str,
    metadata: Optional[dict] = None,
) -> int:
    return (
        _text_bytes(name)
        + _text_bytes(concept_type)
        + _text_bytes(description)
        + _json_bytes(metadata or {})
    )


def estimate_pattern_bytes(
    *,
    category: str,
    pattern_name: str,
    pattern_text: str,
    evidence_observation_ids: Optional[list[int]] = None,
) -> int:
    return (
        _text_bytes(category)
        + _text_bytes(pattern_name)
        + _text_bytes(pattern_text)
        + _json_bytes(evidence_observation_ids or [])
    )


def estimate_chain_create_bytes(
    *,
    chain_type: str,
    title: Optional[str],
    metadata: Optional[dict],
    store_id: Optional[str] = None,
    scope: Optional[str] = None,
) -> int:
    return (
        _text_bytes(chain_type)
        + _text_bytes(title)
        + _json_bytes(metadata or {})
        + _text_bytes(store_id)
        + _text_bytes(scope)
    )


def estimate_chain_append_bytes(
    *,
    item_type: str,
    item_id: Optional[str],
    text: Optional[str],
    role: Optional[str],
) -> int:
    return (
        _text_bytes(item_type)
        + _text_bytes(item_id)
        + _text_bytes(text)
        + _text_bytes(role)
    )


def estimate_chain_update_bytes(
    *,
    title: Optional[str],
    metadata: Optional[dict],
    status: Optional[str],
) -> int:
    return _text_bytes(title) + _json_bytes(metadata or {}) + _text_bytes(status)


def _load_usage_row(db, tenant_id: str) -> TenantStorageUsage:
    row = (
        db.query(TenantStorageUsage)
        .filter(TenantStorageUsage.tenant_id == tenant_id)
        .first()
    )
    if row is None:
        row = TenantStorageUsage(tenant_id=tenant_id, storage_used_bytes=0)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def enforce_quota_or_raise(
    db,
    *,
    tenant_id: Optional[str],
    bytes_to_write: int,
) -> None:
    if bytes_to_write <= 0:
        return
    if config.STORAGE_QUOTA_BYTES <= 0:
        return
    effective_tenant_id = require_tenant_id_value(tenant_id)
    usage = _load_usage_row(db, effective_tenant_id)
    current = usage.storage_used_bytes or 0
    projected = current + bytes_to_write
    if projected > config.STORAGE_QUOTA_BYTES:
        raise ValidationIssue(
            "Storage quota exceeded",
            field="storage_used_bytes",
            error_type="quota_exceeded",
        )


def record_storage_usage(
    db,
    *,
    tenant_id: Optional[str],
    bytes_delta: int,
) -> None:
    if bytes_delta == 0:
        return
    if config.STORAGE_QUOTA_BYTES <= 0:
        return
    effective_tenant_id = require_tenant_id_value(tenant_id)
    usage = _load_usage_row(db, effective_tenant_id)
    current = usage.storage_used_bytes or 0
    usage.storage_used_bytes = max(0, current + bytes_delta)
    db.commit()


def calculate_storage_usage_bytes(db, *, tenant_id: str) -> int:
    total = 0
    embed_bytes = _embedding_bytes_per_row()

    observations = (
        db.query(Observation)
        .filter(Observation.tenant_id == tenant_id)
        .yield_per(500)
    )
    for obs in observations:
        total += estimate_observation_bytes(
            obs.observation,
            domain=obs.domain,
            evidence=obs.evidence,
        )

    patterns = (
        db.query(Pattern)
        .filter(Pattern.tenant_id == tenant_id)
        .yield_per(500)
    )
    for row in patterns:
        total += estimate_pattern_bytes(
            category=row.category,
            pattern_name=row.pattern_name,
            pattern_text=row.pattern_text,
            evidence_observation_ids=row.evidence_observation_ids,
        )

    concepts = (
        db.query(Concept)
        .filter(Concept.tenant_id == tenant_id)
        .yield_per(500)
    )
    for row in concepts:
        total += estimate_concept_bytes(
            name=row.name,
            concept_type=row.type,
            description=row.description or "",
            metadata=row.metadata_,
        )

    documents = (
        db.query(Document)
        .filter(Document.tenant_id == tenant_id)
        .yield_per(200)
    )
    for row in documents:
        total += estimate_document_bytes(
            title=row.title,
            doc_type=row.doc_type,
            url=row.url or "",
            content_summary=row.content_summary or "",
            key_concepts=row.key_concepts,
            metadata=row.metadata_,
        )

    summaries = (
        db.query(MemorySummary)
        .filter(MemorySummary.tenant_id == tenant_id)
        .yield_per(200)
    )
    for row in summaries:
        total += (
            _text_bytes(row.summary_text)
            + _text_bytes(row.source_type)
            + _json_bytes(row.source_ids)
            + _json_bytes(row.metadata_)
        )

    if embed_bytes > 0:
        embed_count = (
            db.query(func.count(Embedding.source_id))
            .filter(Embedding.tenant_id == tenant_id)
            .scalar()
        )
        if embed_count:
            total += int(embed_count) * embed_bytes

    return total


def recompute_usage_for_tenant(tenant_id: str) -> dict:
    db = DB.SessionLocal()
    try:
        usage_bytes = calculate_storage_usage_bytes(db, tenant_id=tenant_id)
        row = _load_usage_row(db, tenant_id)
        row.storage_used_bytes = usage_bytes
        db.commit()
        return {
            "status": "ok",
            "tenant_id": tenant_id,
            "storage_used_bytes": usage_bytes,
            "storage_used_gb": gb_from_bytes(usage_bytes),
        }
    finally:
        db.close()


def gb_from_bytes(value: int) -> float:
    return round(float(value) / (1024 ** 3), 4)


def quota_status(context: Optional[RequestContext] = None) -> dict:
    tenant_id = resolve_tenant_id(context)
    if not tenant_id:
        return {"status": "error", "message": "tenant_id is required"}
    db = DB.SessionLocal()
    try:
        usage = _load_usage_row(db, tenant_id)
        storage_used_bytes = usage.storage_used_bytes or 0
        return {
            "status": "ok",
            "tenant_id": tenant_id,
            "storage_used_bytes": storage_used_bytes,
            "storage_used_gb": gb_from_bytes(storage_used_bytes),
            "hard_limit_bytes": config.STORAGE_QUOTA_BYTES,
            "enforcement_enabled": config.STORAGE_QUOTA_BYTES > 0,
        }
    finally:
        db.close()

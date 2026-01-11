"""
Archive and rehydration services.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, List

from sqlalchemy import and_, func, or_

import core.config as config
from core.context import RequestContext
from core.audit import ALLOWED_ACTOR_TYPES, log_event
from core.audit_constants import (
    EVENT_MEMORY_ARCHIVED,
    EVENT_MEMORY_PURGED_TO_ARCHIVE,
    EVENT_MEMORY_REHYDRATED,
    EVENT_MEMORY_RESTORED_FROM_ARCHIVE,
    EVENT_RETENTION_ARCHIVE_EVICTED,
)
from core.db import DB
from core.errors import ValidationIssue
from core.models import (
    ArchivedMemory,
    Concept,
    ConceptAlias,
    ConceptRelationship,
    Document,
    Embedding,
    MemorySummary,
    MemoryTier,
    Observation,
    Pattern,
    TombstoneAction,
    MEMORY_MODELS,
)
from core.services.memory_search import search_cold_memory
from core.services.memory_shared import (
    _apply_rehydrate_bump,
    _collect_records_by_refs,
    _collect_summary_threshold_records,
    _collect_threshold_records,
    _find_summary_for_source,
    _parse_memory_ref,
    _serialize_memory_id,
    _store_embedding,
    _summary_text_for_record,
    _validate_limit,
    _vector_search_enabled,
    _write_tombstone,
    ARCHIVE_LIMIT_DEFAULT,
    ARCHIVE_LIMIT_MAX,
    ARCHIVE_MULTIPLIER,
    EMBEDDING_DIM,
    REHYDRATE_LIMIT_MAX,
    SUMMARY_TRIGGER_SCORE,
    STORAGE_QUOTA_BYTES,
    TOMBSTONES_ENABLED,
    logger,
    service_tool,
)

ARCHIVE_SOURCE_TYPES = set(MEMORY_MODELS.keys()) | {"summary"}


def _resolve_audit_actor(
    context: Optional[RequestContext],
    actor_label: Optional[str],
) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    actor_value = (actor_label or "").strip()
    normalized = actor_value.lower() if actor_value else ""
    if normalized in ALLOWED_ACTOR_TYPES:
        actor_type = normalized
        actor_id = None
    else:
        actor_type = "integration"
        actor_id = actor_value or None
    org_id = None
    user_id = None
    request_id = None
    if context:
        if context.auth:
            if context.auth.tenant_id:
                org_id = str(context.auth.tenant_id)
            if context.auth.user_id is not None:
                user_id = str(context.auth.user_id)
        request_id = context.request_id
    return actor_type, actor_id, org_id, user_id, request_id


def _log_audit_event(
    db,
    *,
    event_type: str,
    target_type: str,
    target_ids: list,
    count_affected: int,
    reason: Optional[str],
    actor_label: Optional[str],
    context: Optional[RequestContext],
    metadata: Optional[dict] = None,
) -> None:
    if not target_ids:
        return
    actor_type, actor_id, org_id, user_id, request_id = _resolve_audit_actor(context, actor_label)
    log_event(
        db,
        event_type=event_type,
        actor_type=actor_type,
        actor_id=actor_id,
        org_id=org_id,
        user_id=user_id,
        target_type=target_type,
        target_ids=target_ids,
        count_affected=count_affected,
        reason=reason,
        request_id=request_id,
        metadata=metadata,
    )


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _deserialize_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _estimate_text_size(*values: Optional[str]) -> int:
    total = 0
    for value in values:
        if value:
            total += len(value)
    return total


def _estimate_json_size(value) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(value))


def _estimate_archive_size(mem_type: str, payload: dict) -> int:
    text_size = 0
    json_size = 0
    if mem_type == "observation":
        text_size += _estimate_text_size(payload.get("observation"), payload.get("domain"))
        json_size += _estimate_json_size(payload.get("evidence"))
    elif mem_type == "pattern":
        text_size += _estimate_text_size(
            payload.get("category"),
            payload.get("pattern_name"),
            payload.get("pattern_text"),
        )
        json_size += _estimate_json_size(payload.get("evidence_observation_ids"))
    elif mem_type == "concept":
        text_size += _estimate_text_size(
            payload.get("name"),
            payload.get("name_key"),
            payload.get("type"),
            payload.get("status"),
            payload.get("domain"),
            payload.get("description"),
        )
        json_size += _estimate_json_size(payload.get("metadata"))
        json_size += _estimate_json_size(payload.get("aliases"))
        json_size += _estimate_json_size(payload.get("relationships"))
    elif mem_type == "document":
        text_size += _estimate_text_size(
            payload.get("title"),
            payload.get("doc_type"),
            payload.get("content_summary"),
            payload.get("url"),
        )
        json_size += _estimate_json_size(payload.get("key_concepts"))
        json_size += _estimate_json_size(payload.get("metadata"))
    elif mem_type == "summary":
        text_size += _estimate_text_size(payload.get("summary_text"), payload.get("source_type"))
        json_size += _estimate_json_size(payload.get("source_ids"))
        json_size += _estimate_json_size(payload.get("metadata"))
    else:
        text_size += _estimate_text_size(str(payload))

    embedding_size = 0
    if _vector_search_enabled() and mem_type in MEMORY_MODELS:
        embedding_size = EMBEDDING_DIM * 4

    return int(text_size + json_size + embedding_size)


def _build_archive_payload(db, mem_type: str, record) -> dict:
    if mem_type == "observation":
        return {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "timestamp": _serialize_datetime(record.timestamp),
            "observation": record.observation,
            "confidence": record.confidence,
            "domain": record.domain,
            "evidence": record.evidence,
            "session_id": record.session_id,
            "ai_instance_id": record.ai_instance_id,
            "access_count": record.access_count,
            "last_accessed_at": _serialize_datetime(record.last_accessed_at),
            "archived_at": _serialize_datetime(record.archived_at),
            "archived_reason": record.archived_reason,
            "archived_by": record.archived_by,
            "score": record.score,
            "floor_score": record.floor_score,
            "purge_eligible": record.purge_eligible,
        }

    if mem_type == "pattern":
        return {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "category": record.category,
            "pattern_name": record.pattern_name,
            "pattern_text": record.pattern_text,
            "confidence": record.confidence,
            "last_updated": _serialize_datetime(record.last_updated),
            "evidence_observation_ids": record.evidence_observation_ids,
            "session_id": record.session_id,
            "ai_instance_id": record.ai_instance_id,
            "access_count": record.access_count,
            "last_accessed_at": _serialize_datetime(record.last_accessed_at),
            "archived_at": _serialize_datetime(record.archived_at),
            "archived_reason": record.archived_reason,
            "archived_by": record.archived_by,
            "score": record.score,
            "floor_score": record.floor_score,
            "purge_eligible": record.purge_eligible,
        }

    if mem_type == "concept":
        aliases = [
            {
                "alias": alias.alias,
                "alias_key": alias.alias_key,
                "created_at": _serialize_datetime(alias.created_at),
            }
            for alias in (record.aliases or [])
        ]
        outgoing = db.query(ConceptRelationship).filter(
            ConceptRelationship.from_concept_id == record.id
        ).all()
        incoming = db.query(ConceptRelationship).filter(
            ConceptRelationship.to_concept_id == record.id
        ).all()
        relationships = {
            "outgoing": [
                {
                    "to_concept_id": rel.to_concept_id,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description,
                    "created_at": _serialize_datetime(rel.created_at),
                }
                for rel in outgoing
            ],
            "incoming": [
                {
                    "from_concept_id": rel.from_concept_id,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description,
                    "created_at": _serialize_datetime(rel.created_at),
                }
                for rel in incoming
            ],
        }
        return {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "name": record.name,
            "name_key": record.name_key,
            "type": record.type,
            "status": record.status,
            "domain": record.domain,
            "description": record.description,
            "metadata": record.metadata_,
            "ai_instance_id": record.ai_instance_id,
            "created_at": _serialize_datetime(record.created_at),
            "access_count": record.access_count,
            "last_accessed_at": _serialize_datetime(record.last_accessed_at),
            "archived_at": _serialize_datetime(record.archived_at),
            "archived_reason": record.archived_reason,
            "archived_by": record.archived_by,
            "score": record.score,
            "floor_score": record.floor_score,
            "purge_eligible": record.purge_eligible,
            "aliases": aliases,
            "relationships": relationships,
        }

    if mem_type == "document":
        return {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "title": record.title,
            "doc_type": record.doc_type,
            "content_summary": record.content_summary,
            "url": record.url,
            "publication_date": _serialize_datetime(record.publication_date),
            "key_concepts": record.key_concepts,
            "metadata": record.metadata_,
            "created_at": _serialize_datetime(record.created_at),
            "access_count": record.access_count,
            "last_accessed_at": _serialize_datetime(record.last_accessed_at),
            "archived_at": _serialize_datetime(record.archived_at),
            "archived_reason": record.archived_reason,
            "archived_by": record.archived_by,
            "score": record.score,
            "floor_score": record.floor_score,
            "purge_eligible": record.purge_eligible,
        }

    if mem_type == "summary":
        return {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "source_type": record.source_type,
            "source_id": record.source_id,
            "source_ids": record.source_ids,
            "summary_text": record.summary_text,
            "metadata": record.metadata_,
            "created_at": _serialize_datetime(record.created_at),
            "access_count": record.access_count,
            "last_accessed_at": _serialize_datetime(record.last_accessed_at),
            "archived_at": _serialize_datetime(record.archived_at),
            "archived_reason": record.archived_reason,
            "archived_by": record.archived_by,
            "score": record.score,
            "floor_score": record.floor_score,
            "purge_eligible": record.purge_eligible,
        }

    return {}


def _embedding_text_for_payload(mem_type: str, payload: dict) -> Optional[str]:
    if mem_type == "observation":
        return payload.get("observation")
    if mem_type == "pattern":
        return payload.get("pattern_text")
    if mem_type == "concept":
        return payload.get("description")
    if mem_type == "document":
        return payload.get("content_summary")
    return None


def _archive_cold_record_to_store(
    db,
    mem_type: str,
    record,
    reason: str,
    actor: str,
) -> ArchivedMemory:
    payload = _build_archive_payload(db, mem_type, record)
    size_estimate = _estimate_archive_size(mem_type, payload)
    tier_value = record.tier.value if isinstance(record.tier, MemoryTier) else record.tier
    archive_row = ArchivedMemory(
        tenant_id=record.tenant_id,
        source_type=mem_type,
        source_id=record.id,
        payload=payload,
        archived_at=datetime.utcnow(),
        purge_reason=reason,
        purge_actor=actor,
        size_bytes_estimate=size_estimate,
        original_embedding=None,
        metadata_={"source_tier": tier_value},
    )
    db.add(archive_row)

    if mem_type in MEMORY_MODELS:
        db.query(Embedding).filter(
            Embedding.source_type == mem_type,
            Embedding.source_id == record.id,
        ).delete(synchronize_session=False)

    if mem_type == "concept":
        db.query(ConceptRelationship).filter(
            ConceptRelationship.from_concept_id == record.id
        ).delete(synchronize_session=False)
        db.query(ConceptRelationship).filter(
            ConceptRelationship.to_concept_id == record.id
        ).delete(synchronize_session=False)
        db.query(ConceptAlias).filter(
            ConceptAlias.concept_id == record.id
        ).delete(synchronize_session=False)

    db.delete(record)
    return archive_row


def _restore_archived_record(
    db,
    archive_row: ArchivedMemory,
    target_tier: MemoryTier,
    reason: str,
    actor: str,
    bump_score: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    mem_type = archive_row.source_type
    payload = archive_row.payload or {}
    source_id = payload.get("id") or archive_row.source_id

    if mem_type == "summary":
        existing = db.query(MemorySummary).filter(MemorySummary.id == source_id).first()
        if existing:
            return False, [f"summary:{source_id} already exists"]

        summary = MemorySummary(
            id=source_id,
            tenant_id=payload.get("tenant_id") or config.DEFAULT_TENANT_ID,
            source_type=payload.get("source_type"),
            source_id=payload.get("source_id"),
            source_ids=payload.get("source_ids") or [],
            summary_text=payload.get("summary_text") or "",
            metadata_=payload.get("metadata") or {},
            created_at=_deserialize_datetime(payload.get("created_at")),
            access_count=payload.get("access_count") or 0,
            last_accessed_at=_deserialize_datetime(payload.get("last_accessed_at")),
            tier=target_tier,
            archived_at=None,
            archived_reason=None,
            archived_by=None,
            score=payload.get("score") or 0.0,
            floor_score=payload.get("floor_score") or -9999.0,
            purge_eligible=False,
        )
        if target_tier == MemoryTier.cold:
            summary.archived_at = _deserialize_datetime(payload.get("archived_at")) or datetime.utcnow()
            summary.archived_reason = payload.get("archived_reason")
            summary.archived_by = payload.get("archived_by")
        if target_tier == MemoryTier.hot and bump_score:
            summary.access_count = (summary.access_count or 0) + 1
            summary.last_accessed_at = datetime.utcnow()
            _apply_rehydrate_bump(summary)

        db.add(summary)
        _write_tombstone(
            db,
            f"summary:{source_id}",
            TombstoneAction.rehydrated,
            from_tier=None,
            to_tier=target_tier,
            reason=reason,
            actor=actor,
        )
        db.delete(archive_row)
        return True, errors

    model = MEMORY_MODELS.get(mem_type)
    if not model:
        return False, [f"unsupported type: {mem_type}"]

    existing = db.query(model).filter(model.id == source_id).first()
    if existing:
        return False, [f"{mem_type}:{source_id} already exists"]

    if mem_type == "observation":
        record = Observation(
            id=source_id,
            tenant_id=payload.get("tenant_id") or config.DEFAULT_TENANT_ID,
            timestamp=_deserialize_datetime(payload.get("timestamp")),
            observation=payload.get("observation"),
            confidence=payload.get("confidence", 0.8),
            domain=payload.get("domain"),
            evidence=payload.get("evidence") or [],
            session_id=payload.get("session_id"),
            ai_instance_id=payload.get("ai_instance_id"),
            access_count=payload.get("access_count") or 0,
            last_accessed_at=_deserialize_datetime(payload.get("last_accessed_at")),
            tier=target_tier,
            archived_at=None,
            archived_reason=None,
            archived_by=None,
            score=payload.get("score") or 0.0,
            floor_score=payload.get("floor_score") or -9999.0,
            purge_eligible=False,
        )
    elif mem_type == "pattern":
        record = Pattern(
            id=source_id,
            tenant_id=payload.get("tenant_id") or config.DEFAULT_TENANT_ID,
            category=payload.get("category") or "",
            pattern_name=payload.get("pattern_name") or "",
            pattern_text=payload.get("pattern_text") or "",
            confidence=payload.get("confidence", 0.8),
            last_updated=_deserialize_datetime(payload.get("last_updated")),
            evidence_observation_ids=payload.get("evidence_observation_ids") or [],
            session_id=payload.get("session_id"),
            ai_instance_id=payload.get("ai_instance_id"),
            access_count=payload.get("access_count") or 0,
            last_accessed_at=_deserialize_datetime(payload.get("last_accessed_at")),
            tier=target_tier,
            archived_at=None,
            archived_reason=None,
            archived_by=None,
            score=payload.get("score") or 0.0,
            floor_score=payload.get("floor_score") or -9999.0,
            purge_eligible=False,
        )
    elif mem_type == "concept":
        record = Concept(
            id=source_id,
            tenant_id=payload.get("tenant_id") or config.DEFAULT_TENANT_ID,
            name=payload.get("name") or "",
            name_key=payload.get("name_key") or (payload.get("name") or "").lower(),
            type=payload.get("type") or "",
            status=payload.get("status"),
            domain=payload.get("domain"),
            description=payload.get("description") or "",
            metadata_=payload.get("metadata") or {},
            ai_instance_id=payload.get("ai_instance_id"),
            created_at=_deserialize_datetime(payload.get("created_at")),
            access_count=payload.get("access_count") or 0,
            last_accessed_at=_deserialize_datetime(payload.get("last_accessed_at")),
            tier=target_tier,
            archived_at=None,
            archived_reason=None,
            archived_by=None,
            score=payload.get("score") or 0.0,
            floor_score=payload.get("floor_score") or -9999.0,
            purge_eligible=False,
        )
    else:
        record = Document(
            id=source_id,
            tenant_id=payload.get("tenant_id") or config.DEFAULT_TENANT_ID,
            title=payload.get("title") or "",
            doc_type=payload.get("doc_type") or "",
            content_summary=payload.get("content_summary"),
            url=payload.get("url"),
            publication_date=_deserialize_datetime(payload.get("publication_date")),
            key_concepts=payload.get("key_concepts") or [],
            metadata_=payload.get("metadata") or {},
            created_at=_deserialize_datetime(payload.get("created_at")),
            access_count=payload.get("access_count") or 0,
            last_accessed_at=_deserialize_datetime(payload.get("last_accessed_at")),
            tier=target_tier,
            archived_at=None,
            archived_reason=None,
            archived_by=None,
            score=payload.get("score") or 0.0,
            floor_score=payload.get("floor_score") or -9999.0,
            purge_eligible=False,
        )

    if target_tier == MemoryTier.cold:
        record.archived_at = _deserialize_datetime(payload.get("archived_at")) or datetime.utcnow()
        record.archived_reason = payload.get("archived_reason")
        record.archived_by = payload.get("archived_by")

    if target_tier == MemoryTier.hot and bump_score:
        record.access_count = (record.access_count or 0) + 1
        record.last_accessed_at = datetime.utcnow()
        _apply_rehydrate_bump(record)

    db.add(record)

    if mem_type == "concept":
        for alias in payload.get("aliases") or []:
            alias_key = alias.get("alias_key") or (alias.get("alias") or "").lower()
            if not alias_key:
                continue
            existing_alias = db.query(ConceptAlias).filter(
                ConceptAlias.alias_key == alias_key
            ).first()
            if existing_alias:
                errors.append(f"alias '{alias.get('alias')}' already exists")
                continue
            db.add(ConceptAlias(
                concept_id=record.id,
                tenant_id=record.tenant_id,
                alias=alias.get("alias"),
                alias_key=alias_key,
                created_at=_deserialize_datetime(alias.get("created_at")),
            ))

        relationships = payload.get("relationships") or {}
        for rel in relationships.get("outgoing", []) or []:
            to_id = rel.get("to_concept_id")
            rel_type = rel.get("rel_type")
            if not to_id or not rel_type:
                continue
            if not db.query(Concept).filter(Concept.id == to_id).first():
                errors.append(f"missing concept {to_id} for outgoing {rel_type}")
                continue
            existing_rel = db.query(ConceptRelationship).filter(
                ConceptRelationship.from_concept_id == record.id,
                ConceptRelationship.to_concept_id == to_id,
                ConceptRelationship.rel_type == rel_type,
            ).first()
            if existing_rel:
                continue
            db.add(ConceptRelationship(
                from_concept_id=record.id,
                to_concept_id=to_id,
                rel_type=rel_type,
                tenant_id=record.tenant_id,
                weight=rel.get("weight", 0.5),
                description=rel.get("description"),
                created_at=_deserialize_datetime(rel.get("created_at")),
            ))

        for rel in relationships.get("incoming", []) or []:
            from_id = rel.get("from_concept_id")
            rel_type = rel.get("rel_type")
            if not from_id or not rel_type:
                continue
            if not db.query(Concept).filter(Concept.id == from_id).first():
                errors.append(f"missing concept {from_id} for incoming {rel_type}")
                continue
            existing_rel = db.query(ConceptRelationship).filter(
                ConceptRelationship.from_concept_id == from_id,
                ConceptRelationship.to_concept_id == record.id,
                ConceptRelationship.rel_type == rel_type,
            ).first()
            if existing_rel:
                continue
            db.add(ConceptRelationship(
                from_concept_id=from_id,
                to_concept_id=record.id,
                rel_type=rel_type,
                tenant_id=record.tenant_id,
                weight=rel.get("weight", 0.5),
                description=rel.get("description"),
                created_at=_deserialize_datetime(rel.get("created_at")),
            ))

    _write_tombstone(
        db,
        _serialize_memory_id(mem_type, source_id),
        TombstoneAction.rehydrated,
        from_tier=None,
        to_tier=target_tier,
        reason=reason,
        actor=actor,
    )

    db.delete(archive_row)
    return True, errors


def _archive_quota_bytes() -> int:
    return int(STORAGE_QUOTA_BYTES * ARCHIVE_MULTIPLIER)


def _enforce_archive_quota(db) -> dict:
    quota = _archive_quota_bytes()
    if quota <= 0:
        return {"evicted": 0, "bytes_before": 0, "bytes_after": 0}

    total = db.query(func.coalesce(func.sum(ArchivedMemory.size_bytes_estimate), 0)).scalar() or 0
    if total <= quota:
        return {"evicted": 0, "bytes_before": int(total), "bytes_after": int(total)}

    running = 0
    evicted = 0
    evicted_ids: list[int] = []
    oldest_archived_at: Optional[datetime] = None
    newest_archived_at: Optional[datetime] = None
    for row in db.query(ArchivedMemory).order_by(ArchivedMemory.archived_at.asc()).all():
        running += row.size_bytes_estimate or 0
        if len(evicted_ids) < 50:
            evicted_ids.append(row.id)
        if row.archived_at:
            if oldest_archived_at is None or row.archived_at < oldest_archived_at:
                oldest_archived_at = row.archived_at
            if newest_archived_at is None or row.archived_at > newest_archived_at:
                newest_archived_at = row.archived_at
        db.delete(row)
        evicted += 1
        if total - running <= quota:
            break

    if evicted_ids:
        log_event(
            db,
            event_type=EVENT_RETENTION_ARCHIVE_EVICTED,
            actor_type="system",
            target_type="memory",
            target_ids=evicted_ids,
            count_affected=evicted,
            reason="archive_quota_evicted",
            metadata={
                "bytes_before": int(total),
                "bytes_after": int(max(total - running, 0)),
                "quota_bytes": quota,
                "oldest_archived_at": oldest_archived_at.isoformat() if oldest_archived_at else None,
                "newest_archived_at": newest_archived_at.isoformat() if newest_archived_at else None,
                "sample_size": len(evicted_ids),
            },
        )

    db.commit()
    logger.warning(
        "archive_quota_eviction",
        extra={
            "evicted_count": evicted,
            "bytes_before": int(total),
            "bytes_after": int(max(total - running, 0)),
            "quota_bytes": quota,
        },
    )
    return {"evicted": evicted, "bytes_before": int(total), "bytes_after": int(max(total - running, 0))}

@service_tool
def archive_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    cluster_ids: Optional[List[str]] = None,
    threshold: Optional[dict] = None,
    mode: str = "archive_and_tombstone",
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = True,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Archive hot records into the cold tier.
    """
    if cluster_ids:
        return {"status": "error", "message": "cluster_ids not supported"}
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)

    mode = mode.strip().lower()
    valid_modes = {"archive_only", "archive_and_tombstone", "archive_and_summarize_then_archive"}
    if mode not in valid_modes:
        return {"status": "error", "message": f"Invalid mode. Must be one of: {', '.join(sorted(valid_modes))}"}

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        candidates: list[tuple[str, object]] = []
        summary_records: list[MemorySummary] = []

        if memory_ids:
            refs = [_parse_memory_ref(raw) for raw in memory_ids]
            candidates.extend(_collect_records_by_refs(db, refs))

        if summary_ids:
            summary_records.extend(
                db.query(MemorySummary).filter(MemorySummary.id.in_(summary_ids)).all()
            )

        if threshold:
            below_score = threshold.get("below_score")
            threshold_type = threshold.get("type", "memory").lower()
            if below_score is None:
                return {"status": "error", "message": "threshold requires below_score"}
            if threshold_type not in {"memory", "summary", "any"}:
                return {"status": "error", "message": "threshold.type must be memory|summary|any"}

            if threshold_type in {"memory", "any"}:
                candidates.extend(
                    _collect_threshold_records(
                        db,
                        tier=MemoryTier.hot,
                        below_score=below_score,
                        above_score=None,
                        types=list(MEMORY_MODELS.keys()),
                        limit=limit,
                    )
                )
            if threshold_type in {"summary", "any"}:
                summary_records.extend(
                    _collect_summary_threshold_records(
                        db,
                        tier=MemoryTier.hot,
                        below_score=below_score,
                        above_score=None,
                        limit=limit,
                    )
                )

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for mem_type, record in candidates:
            key = (mem_type, record.id)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append((mem_type, record))
        candidates = unique_candidates[:limit]

        summary_records = list({summary.id: summary for summary in summary_records}.values())[:limit]

        if dry_run:
            return {
                "status": "dry_run",
                "candidate_count": len(candidates),
                "summary_candidate_count": len(summary_records),
                "candidates": [
                    {"type": mem_type, "id": record.id, "score": record.score}
                    for mem_type, record in candidates
                ],
                "summary_candidates": [
                    {"id": summary.id, "score": summary.score}
                    for summary in summary_records
                ],
            }

        archived_ids = []
        archived_memory_ids = []
        archived_summary_ids = []
        already_archived_ids = []
        tombstones_written = 0
        summaries_created = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.hot:
                already_archived_ids.append(_serialize_memory_id(mem_type, record.id))
                continue

            if mode == "archive_and_summarize_then_archive":
                summary = _find_summary_for_source(db, mem_type, record.id)
                summary_text = _summary_text_for_record(mem_type, record)
                if summary:
                    summary.summary_text = summary_text
                else:
                    summary = MemorySummary(
                        source_type=mem_type,
                        source_id=record.id,
                        source_ids=[record.id],
                        summary_text=summary_text,
                        metadata_={"reason": reason},
                    )
                    db.add(summary)
                    summaries_created += 1
                _write_tombstone(
                    db,
                    _serialize_memory_id(mem_type, record.id),
                    TombstoneAction.summarized,
                    from_tier=record.tier,
                    to_tier=record.tier,
                    reason=reason,
                    actor=actor_name,
                )
                tombstones_written += 1 if TOMBSTONES_ENABLED else 0

            record.tier = MemoryTier.cold
            record.archived_at = datetime.utcnow()
            record.archived_reason = reason
            record.archived_by = actor_name
            record.purge_eligible = False
            archived_id = _serialize_memory_id(mem_type, record.id)
            archived_ids.append(archived_id)
            archived_memory_ids.append(archived_id)
            _write_tombstone(
                db,
                archived_id,
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        for summary in summary_records:
            if summary.tier != MemoryTier.hot:
                already_archived_ids.append(f"summary:{summary.id}")
                continue
            summary.tier = MemoryTier.cold
            summary.archived_at = datetime.utcnow()
            summary.archived_reason = reason
            summary.archived_by = actor_name
            summary.purge_eligible = False
            archived_ids.append(f"summary:{summary.id}")
            archived_summary_ids.append(summary.id)
            _write_tombstone(
                db,
                f"summary:{summary.id}",
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        _log_audit_event(
            db,
            event_type=EVENT_MEMORY_ARCHIVED,
            target_type="memory",
            target_ids=archived_memory_ids,
            count_affected=len(archived_memory_ids),
            reason=reason,
            actor_label=actor_name,
            context=context,
            metadata={"mode": mode},
        )
        _log_audit_event(
            db,
            event_type=EVENT_MEMORY_ARCHIVED,
            target_type="summary",
            target_ids=archived_summary_ids,
            count_affected=len(archived_summary_ids),
            reason=reason,
            actor_label=actor_name,
            context=context,
            metadata={"mode": mode},
        )

        db.commit()

        return {
            "status": "archived",
            "archived_count": len(archived_ids),
            "archived_ids": archived_ids,
            "already_archived_count": len(already_archived_ids),
            "already_archived_ids": already_archived_ids,
            "tombstones_written": tombstones_written,
            "summaries_created": summaries_created,
        }
    finally:
        db.close()

@service_tool
def rehydrate_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    cluster_ids: Optional[List[str]] = None,
    threshold: Optional[dict] = None,
    query: Optional[str] = None,
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = False,
    limit: int = 50,
    bump_score: bool = True,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Rehydrate cold records back into hot tier.
    """
    if cluster_ids:
        return {"status": "error", "message": "cluster_ids not supported"}
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", REHYDRATE_LIMIT_MAX)

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        candidates: list[tuple[str, object]] = []
        summary_records: list[MemorySummary] = []

        if memory_ids:
            refs = [_parse_memory_ref(raw) for raw in memory_ids]
            candidates.extend(_collect_records_by_refs(db, refs))

        if summary_ids:
            summary_records.extend(
                db.query(MemorySummary).filter(MemorySummary.id.in_(summary_ids)).all()
            )

        if query:
            cold_results = search_cold_memory(query=query, top_k=limit)
            for row in cold_results.get("results", []):
                candidates.extend(_collect_records_by_refs(
                    db,
                    [(_parse_memory_ref(_serialize_memory_id(row["source_type"], row["id"])))],
                ))

        if threshold:
            below_score = threshold.get("below_score")
            above_score = threshold.get("above_score")
            threshold_type = threshold.get("type", "memory").lower()
            if threshold_type not in {"memory", "summary", "any"}:
                return {"status": "error", "message": "threshold.type must be memory|summary|any"}

            if threshold_type in {"memory", "any"}:
                candidates.extend(
                    _collect_threshold_records(
                        db,
                        tier=MemoryTier.cold,
                        below_score=below_score,
                        above_score=above_score,
                        types=list(MEMORY_MODELS.keys()),
                        limit=limit,
                    )
                )
            if threshold_type in {"summary", "any"}:
                summary_records.extend(
                    _collect_summary_threshold_records(
                        db,
                        tier=MemoryTier.cold,
                        below_score=below_score,
                        above_score=above_score,
                        limit=limit,
                    )
                )

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for mem_type, record in candidates:
            key = (mem_type, record.id)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append((mem_type, record))
        candidates = unique_candidates[:limit]

        summary_records = list({summary.id: summary for summary in summary_records}.values())[:limit]

        if dry_run:
            return {
                "status": "dry_run",
                "candidate_count": len(candidates),
                "summary_candidate_count": len(summary_records),
                "candidates": [
                    {"type": mem_type, "id": record.id, "score": record.score}
                    for mem_type, record in candidates
                ],
                "summary_candidates": [
                    {"id": summary.id, "score": summary.score}
                    for summary in summary_records
                ],
            }

        rehydrated_ids = []
        rehydrated_memory_ids = []
        rehydrated_summary_ids = []
        already_hot_ids = []
        tombstones_written = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.cold:
                already_hot_ids.append(_serialize_memory_id(mem_type, record.id))
                continue
            record.tier = MemoryTier.hot
            record.archived_at = None
            record.archived_reason = None
            record.archived_by = None
            record.purge_eligible = False
            if bump_score:
                record.access_count = (record.access_count or 0) + 1
                record.last_accessed_at = datetime.utcnow()
                _apply_rehydrate_bump(record)
            rehydrated_id = _serialize_memory_id(mem_type, record.id)
            rehydrated_ids.append(rehydrated_id)
            rehydrated_memory_ids.append(rehydrated_id)
            _write_tombstone(
                db,
                rehydrated_id,
                TombstoneAction.rehydrated,
                from_tier=MemoryTier.cold,
                to_tier=MemoryTier.hot,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        for summary in summary_records:
            if summary.tier != MemoryTier.cold:
                already_hot_ids.append(f"summary:{summary.id}")
                continue
            summary.tier = MemoryTier.hot
            summary.archived_at = None
            summary.archived_reason = None
            summary.archived_by = None
            summary.purge_eligible = False
            if bump_score:
                summary.access_count = (summary.access_count or 0) + 1
                summary.last_accessed_at = datetime.utcnow()
                _apply_rehydrate_bump(summary)
            rehydrated_ids.append(f"summary:{summary.id}")
            rehydrated_summary_ids.append(summary.id)
            _write_tombstone(
                db,
                f"summary:{summary.id}",
                TombstoneAction.rehydrated,
                from_tier=MemoryTier.cold,
                to_tier=MemoryTier.hot,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        _log_audit_event(
            db,
            event_type=EVENT_MEMORY_REHYDRATED,
            target_type="memory",
            target_ids=rehydrated_memory_ids,
            count_affected=len(rehydrated_memory_ids),
            reason=reason,
            actor_label=actor_name,
            context=context,
            metadata={"bump_score": bump_score},
        )
        _log_audit_event(
            db,
            event_type=EVENT_MEMORY_REHYDRATED,
            target_type="summary",
            target_ids=rehydrated_summary_ids,
            count_affected=len(rehydrated_summary_ids),
            reason=reason,
            actor_label=actor_name,
            context=context,
            metadata={"bump_score": bump_score},
        )

        db.commit()

        return {
            "status": "rehydrated",
            "rehydrated_count": len(rehydrated_ids),
            "rehydrated_ids": rehydrated_ids,
            "already_hot_count": len(already_hot_ids),
            "already_hot_ids": already_hot_ids,
            "tombstones_written": tombstones_written,
        }
    finally:
        db.close()

@service_tool
def list_archive_candidates(
    below_score: float = SUMMARY_TRIGGER_SCORE,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    List archive candidates without mutation.
    """
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)
    db = DB.SessionLocal()
    try:
        candidates = _collect_threshold_records(
            db,
            tier=MemoryTier.hot,
            below_score=below_score,
            above_score=None,
            types=list(MEMORY_MODELS.keys()),
            limit=limit,
        )
        return {
            "status": "ok",
            "candidate_count": len(candidates),
            "candidates": [
                {"type": mem_type, "id": record.id, "score": record.score}
                for mem_type, record in candidates
            ],
        }
    finally:
        db.close()


@service_tool
def list_archived_memories(
    memory_ids: Optional[List[str]] = None,
    source_type: Optional[str] = None,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    offset: int = 0,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    List archived records without returning payload contents.
    """
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)
    if offset < 0:
        return {"status": "error", "message": "offset must be >= 0"}

    errors: list[dict] = []
    refs: list[tuple[str, int]] = []

    if memory_ids:
        for raw in memory_ids:
            try:
                mem_type, mem_id = _parse_memory_ref(raw)
            except ValidationIssue as exc:
                errors.append({"id": str(raw), "message": str(exc)})
                continue
            if mem_type not in ARCHIVE_SOURCE_TYPES:
                errors.append({"id": str(raw), "message": f"unsupported type: {mem_type}"})
                continue
            refs.append((mem_type, mem_id))

    if source_type:
        source_type = source_type.strip().lower()
        if source_type not in ARCHIVE_SOURCE_TYPES:
            return {"status": "error", "message": f"unsupported source_type: {source_type}"}

    seen = set()
    unique_refs = []
    for mem_type, mem_id in refs:
        key = (mem_type, mem_id)
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append((mem_type, mem_id))

    db = DB.SessionLocal()
    try:
        query = db.query(ArchivedMemory)
        if unique_refs:
            filters = [
                and_(
                    ArchivedMemory.source_type == mem_type,
                    ArchivedMemory.source_id == mem_id,
                )
                for mem_type, mem_id in unique_refs
            ]
            query = query.filter(or_(*filters))
        elif source_type:
            query = query.filter(ArchivedMemory.source_type == source_type)

        rows = (
            query.order_by(ArchivedMemory.archived_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return {
            "status": "ok",
            "requested_count": len(memory_ids or []),
            "count": len(rows),
            "items": [
                {
                    "id": row.id,
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "archived_at": row.archived_at.isoformat() if row.archived_at else None,
                    "purge_reason": row.purge_reason,
                    "purge_actor": row.purge_actor,
                    "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                    "size_bytes_estimate": row.size_bytes_estimate,
                    "metadata": row.metadata_,
                }
                for row in rows
            ],
            "errors": errors,
        }
    finally:
        db.close()


@service_tool
def purge_memory_to_archive(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = True,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Purge cold records into the archive store (reversible).
    """
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)

    errors: list[dict] = []
    refs: list[tuple[str, int]] = []

    if memory_ids:
        for raw in memory_ids:
            try:
                mem_type, mem_id = _parse_memory_ref(raw)
            except ValidationIssue as exc:
                errors.append({"id": str(raw), "message": str(exc)})
                continue
            if mem_type not in ARCHIVE_SOURCE_TYPES:
                errors.append({"id": str(raw), "message": f"unsupported type: {mem_type}"})
                continue
            refs.append((mem_type, mem_id))

    if summary_ids:
        for summary_id in summary_ids:
            if not isinstance(summary_id, int):
                errors.append({"id": str(summary_id), "message": "summary_ids must be integers"})
                continue
            refs.append(("summary", summary_id))

    seen = set()
    unique_refs = []
    for mem_type, mem_id in refs:
        key = (mem_type, mem_id)
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append((mem_type, mem_id))
    unique_refs = unique_refs[:limit]

    if not unique_refs:
        return {
            "status": "error",
            "message": "no valid memory_ids provided",
            "errors": errors,
        }

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        archived_ids: list[str] = []
        archived_memory_ids: list[str] = []
        archived_summary_ids: list[int] = []
        skipped: list[dict] = []
        tombstones_written = 0
        candidates: list[dict] = []

        for mem_type, mem_id in unique_refs:
            if mem_type == "summary":
                record = db.query(MemorySummary).filter(MemorySummary.id == mem_id).first()
            else:
                model = MEMORY_MODELS.get(mem_type)
                if not model:
                    errors.append({"id": f"{mem_type}:{mem_id}", "message": "unsupported type"})
                    continue
                record = db.query(model).filter(model.id == mem_id).first()

            if not record:
                skipped.append({"id": f"{mem_type}:{mem_id}", "reason": "not_found"})
                continue

            if record.tier != MemoryTier.cold:
                skipped.append({"id": f"{mem_type}:{mem_id}", "reason": "not_cold"})
                continue

            existing_archive = db.query(ArchivedMemory).filter(
                ArchivedMemory.source_type == mem_type,
                ArchivedMemory.source_id == mem_id,
            ).first()
            if existing_archive:
                skipped.append({"id": f"{mem_type}:{mem_id}", "reason": "already_archived"})
                continue

            candidates.append(
                {
                    "type": mem_type,
                    "id": mem_id,
                    "score": record.score,
                    "tier": record.tier.value if isinstance(record.tier, MemoryTier) else record.tier,
                }
            )

            if dry_run:
                continue

            _archive_cold_record_to_store(db, mem_type, record, reason, actor_name)
            _write_tombstone(
                db,
                f"{mem_type}:{mem_id}",
                TombstoneAction.purged,
                from_tier=MemoryTier.cold,
                to_tier=None,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0
            archived_ids.append(f"{mem_type}:{mem_id}")
            if mem_type == "summary":
                archived_summary_ids.append(mem_id)
            else:
                archived_memory_ids.append(f"{mem_type}:{mem_id}")

        if dry_run:
            return {
                "status": "dry_run",
                "requested_count": len(unique_refs),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "skipped": skipped,
                "errors": errors,
            }

        if archived_ids:
            _log_audit_event(
                db,
                event_type=EVENT_MEMORY_PURGED_TO_ARCHIVE,
                target_type="memory",
                target_ids=archived_memory_ids,
                count_affected=len(archived_memory_ids),
                reason=reason,
                actor_label=actor_name,
                context=context,
                metadata={"mode": "manual"},
            )
            _log_audit_event(
                db,
                event_type=EVENT_MEMORY_PURGED_TO_ARCHIVE,
                target_type="summary",
                target_ids=archived_summary_ids,
                count_affected=len(archived_summary_ids),
                reason=reason,
                actor_label=actor_name,
                context=context,
                metadata={"mode": "manual"},
            )
            db.commit()
        quota_stats = _enforce_archive_quota(db) if archived_ids else {"evicted": 0}

        return {
            "status": "archived",
            "requested_count": len(unique_refs),
            "affected": len(archived_ids),
            "archived_ids": archived_ids,
            "skipped": len(skipped),
            "skipped_ids": skipped,
            "errors": errors,
            "tombstones_written": tombstones_written,
            "archive_evicted": quota_stats.get("evicted", 0),
        }
    finally:
        db.close()


@service_tool
def restore_archived_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    target_tier: str = "cold",
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = False,
    limit: int = REHYDRATE_LIMIT_MAX,
    bump_score: bool = True,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Restore archived records back into hot or cold tiers.
    """
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", REHYDRATE_LIMIT_MAX)

    tier_value = (target_tier or "").strip().lower()
    if tier_value not in {"hot", "cold"}:
        return {"status": "error", "message": "target_tier must be 'hot' or 'cold'"}
    target = MemoryTier.hot if tier_value == "hot" else MemoryTier.cold

    errors: list[dict] = []
    refs: list[tuple[str, int]] = []

    if memory_ids:
        for raw in memory_ids:
            try:
                mem_type, mem_id = _parse_memory_ref(raw)
            except ValidationIssue as exc:
                errors.append({"id": str(raw), "message": str(exc)})
                continue
            if mem_type not in ARCHIVE_SOURCE_TYPES:
                errors.append({"id": str(raw), "message": f"unsupported type: {mem_type}"})
                continue
            refs.append((mem_type, mem_id))

    if summary_ids:
        for summary_id in summary_ids:
            if not isinstance(summary_id, int):
                errors.append({"id": str(summary_id), "message": "summary_ids must be integers"})
                continue
            refs.append(("summary", summary_id))

    seen = set()
    unique_refs = []
    for mem_type, mem_id in refs:
        key = (mem_type, mem_id)
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append((mem_type, mem_id))
    unique_refs = unique_refs[:limit]

    if not unique_refs:
        return {
            "status": "error",
            "message": "no valid memory_ids provided",
            "errors": errors,
        }

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        restored_ids: list[str] = []
        skipped: list[dict] = []
        tombstones_written = 0
        candidates: list[dict] = []

        for mem_type, mem_id in unique_refs:
            archive_row = db.query(ArchivedMemory).filter(
                ArchivedMemory.source_type == mem_type,
                ArchivedMemory.source_id == mem_id,
            ).first()
            if not archive_row:
                skipped.append({"id": f"{mem_type}:{mem_id}", "reason": "not_archived"})
                continue

            payload = archive_row.payload or {}
            candidates.append(
                {
                    "type": mem_type,
                    "id": mem_id,
                    "archived_at": archive_row.archived_at.isoformat() if archive_row.archived_at else None,
                }
            )

            if dry_run:
                continue

            restored, restore_errors = _restore_archived_record(
                db,
                archive_row,
                target,
                reason,
                actor_name,
                bump_score and target == MemoryTier.hot,
            )
            if not restored:
                skipped.append({"id": f"{mem_type}:{mem_id}", "reason": "restore_failed"})
                for message in restore_errors:
                    errors.append({"id": f"{mem_type}:{mem_id}", "message": message})
                db.rollback()
                continue

            target_ids = [mem_id] if mem_type == "summary" else [f"{mem_type}:{mem_id}"]
            _log_audit_event(
                db,
                event_type=EVENT_MEMORY_RESTORED_FROM_ARCHIVE,
                target_type="summary" if mem_type == "summary" else "memory",
                target_ids=target_ids,
                count_affected=1,
                reason=reason,
                actor_label=actor_name,
                context=context,
                metadata={"target_tier": target.value},
            )
            db.commit()
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0
            restored_ids.append(f"{mem_type}:{mem_id}")
            for message in restore_errors:
                errors.append({"id": f"{mem_type}:{mem_id}", "message": message})

            if mem_type in MEMORY_MODELS:
                text_value = _embedding_text_for_payload(mem_type, payload)
                if text_value:
                    record_id = payload.get("id") or mem_id
                    _store_embedding(db, mem_type, record_id, text_value, replace=True)

        if dry_run:
            return {
                "status": "dry_run",
                "requested_count": len(unique_refs),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "skipped": skipped,
                "errors": errors,
            }

        return {
            "status": "restored",
            "requested_count": len(unique_refs),
            "affected": len(restored_ids),
            "restored_ids": restored_ids,
            "skipped": len(skipped),
            "skipped_ids": skipped,
            "errors": errors,
            "tombstones_written": tombstones_written,
        }
    finally:
        db.close()

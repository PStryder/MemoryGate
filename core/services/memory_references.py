"""
Ref-based memory retrieval services.

Provides direct access to memories by their "type:id" reference.
"""

from __future__ import annotations

from typing import Optional, Callable

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import Observation, Pattern, Concept, Document, MemoryTier
from core.services.ai_memory_policy import apply_ai_memory_filter, get_ai_memory_mode
from core.services.memory_shared import (
    _apply_fetch_bump,
    _parse_memory_ref,
    _validate_required_text,
    _validate_list,
    MAX_SHORT_TEXT_LENGTH,
    MAX_LIST_ITEMS,
    SCORE_BUMP_ALPHA,
    service_tool,
)


SUPPORTED_REF_TYPES = {"observation", "pattern", "concept", "document"}


def _normalize_ref(mem_type: str, mem_id: int) -> str:
    return f"{mem_type}:{mem_id}"


def _validate_ref(raw) -> tuple[str, int, str]:
    if isinstance(raw, str):
        _validate_required_text(raw, "ref", MAX_SHORT_TEXT_LENGTH)
    mem_type, mem_id = _parse_memory_ref(raw)
    if mem_type not in SUPPORTED_REF_TYPES:
        raise ValidationIssue(
            f"Unsupported memory type: {mem_type}",
            field="ref",
            error_type="invalid_type",
        )
    if mem_id <= 0:
        raise ValidationIssue(
            "ref id must be a positive integer",
            field="ref",
            error_type="invalid_id",
        )
    return mem_type, mem_id, _normalize_ref(mem_type, mem_id)


def _serialize_observation(record: Observation) -> dict:
    return {
        "ref": _normalize_ref("observation", record.id),
        "id": record.id,
        "type": "observation",
        "observation": record.observation,
        "confidence": record.confidence,
        "domain": record.domain,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        "evidence": record.evidence,
        "ai_name": record.ai_instance.name if record.ai_instance else None,
        "session_title": record.session.title if record.session else None,
        "tier": record.tier.value if record.tier else "hot",
        "score": record.score,
        "access_count": record.access_count,
    }


def _serialize_pattern(record: Pattern) -> dict:
    return {
        "ref": _normalize_ref("pattern", record.id),
        "id": record.id,
        "type": "pattern",
        "category": record.category,
        "pattern_name": record.pattern_name,
        "pattern_text": record.pattern_text,
        "confidence": record.confidence,
        "evidence_observation_ids": record.evidence_observation_ids,
        "last_updated": record.last_updated.isoformat() if record.last_updated else None,
        "tier": record.tier.value if record.tier else "hot",
        "score": record.score,
        "access_count": record.access_count,
    }


def _serialize_concept(record: Concept) -> dict:
    return {
        "ref": _normalize_ref("concept", record.id),
        "id": record.id,
        "type": "concept",
        "name": record.name,
        "concept_type": record.type,
        "description": record.description,
        "domain": record.domain,
        "status": record.status,
        "metadata": record.metadata_,
        "tier": record.tier.value if record.tier else "hot",
        "score": record.score,
        "access_count": record.access_count,
    }


def _serialize_document(record: Document) -> dict:
    return {
        "ref": _normalize_ref("document", record.id),
        "id": record.id,
        "type": "document",
        "title": record.title,
        "doc_type": record.doc_type,
        "content_summary": record.content_summary,
        "url": record.url,
        "key_concepts": record.key_concepts,
        "publication_date": record.publication_date.isoformat() if record.publication_date else None,
        "tier": record.tier.value if record.tier else "hot",
        "score": record.score,
        "access_count": record.access_count,
    }


_SERIALIZERS: dict[str, Callable[[object], dict]] = {
    "observation": _serialize_observation,
    "pattern": _serialize_pattern,
    "concept": _serialize_concept,
    "document": _serialize_document,
}

_MODELS: dict[str, type] = {
    "observation": Observation,
    "pattern": Pattern,
    "concept": Concept,
    "document": Document,
}


def _fetch_record(
    db,
    *,
    mem_type: str,
    mem_id: int,
    tenant_id: Optional[str],
    include_cold: bool,
    ai_instance_id: Optional[int],
    ai_mode: str,
):
    model = _MODELS.get(mem_type)
    if not model:
        return None
    query = db.query(model).filter(model.id == mem_id)
    if tenant_id:
        query = query.filter(model.tenant_id == tenant_id)
    query = apply_ai_memory_filter(
        query,
        model=model,
        entity_type=mem_type,
        tenant_id=tenant_id,
        ai_instance_id=ai_instance_id,
        mode=ai_mode,
    )
    if not include_cold:
        query = query.filter(model.tier == MemoryTier.hot)
    return query.first()


@service_tool
def memory_get_by_ref(
    ref: str,
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Get a single memory by its type:id reference."""
    try:
        mem_type, mem_id, normalized_ref = _validate_ref(ref)
    except ValidationIssue as exc:
        return {"status": "error", "message": str(exc)}

    try:
        tenant_id = resolve_tenant_id(context)
    except ValidationIssue as exc:
        return {"status": "error", "message": str(exc)}

    db = DB.SessionLocal()
    try:
        ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
        record = _fetch_record(
            db,
            mem_type=mem_type,
            mem_id=mem_id,
            tenant_id=tenant_id,
            include_cold=include_cold,
            ai_instance_id=ai_instance_id,
            ai_mode=ai_mode,
        )
        if not record:
            return {"status": "not_found", "ref": normalized_ref}

        if record.tier == MemoryTier.hot:
            _apply_fetch_bump(record, SCORE_BUMP_ALPHA)
            db.commit()

        serializer = _SERIALIZERS[mem_type]
        return {"status": "found", "result": serializer(record)}
    finally:
        db.close()


@service_tool
def memory_get_many_by_refs(
    refs: list[str],
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Get multiple memories by their type:id references."""
    try:
        _validate_list(refs, "refs", MAX_LIST_ITEMS)
    except ValidationIssue as exc:
        return {"status": "error", "message": str(exc)}

    normalized: list[tuple[str, int, str]] = []
    try:
        for ref in refs:
            normalized.append(_validate_ref(ref))
    except ValidationIssue as exc:
        return {"status": "error", "message": str(exc)}

    try:
        tenant_id = resolve_tenant_id(context)
    except ValidationIssue as exc:
        return {"status": "error", "message": str(exc)}

    if not normalized:
        return {
            "status": "ok",
            "requested": 0,
            "found": 0,
            "missing": [],
            "results": [],
        }

    db = DB.SessionLocal()
    try:
        results: list[dict] = []
        missing: list[str] = []
        bumped = False
        ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)

        for mem_type, mem_id, normalized_ref in normalized:
            record = _fetch_record(
                db,
                mem_type=mem_type,
                mem_id=mem_id,
                tenant_id=tenant_id,
                include_cold=include_cold,
                ai_instance_id=ai_instance_id,
                ai_mode=ai_mode,
            )
            if not record:
                missing.append(normalized_ref)
                continue
            if record.tier == MemoryTier.hot:
                _apply_fetch_bump(record, SCORE_BUMP_ALPHA)
                bumped = True
            serializer = _SERIALIZERS[mem_type]
            results.append(serializer(record))

        if bumped:
            db.commit()

        return {
            "status": "ok",
            "requested": len(normalized),
            "found": len(results),
            "missing": missing,
            "results": results,
        }
    finally:
        db.close()

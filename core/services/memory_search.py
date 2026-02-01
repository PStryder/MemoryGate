
"""
Search and recall services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Iterable

from sqlalchemy import func, and_, or_, desc, text

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import (
    AIInstance,
    Session,
    Observation,
    Pattern,
    Concept,
    Document,
    Embedding,
    MemorySummary,
    MemoryTier,
    MemoryRelationship,
    RelationshipResidue,
    MemoryChain,
    MemoryChainItem,
)
from core.services.ai_memory_policy import apply_ai_memory_filter, get_ai_memory_mode
from core.services.memory_shared import (
    _apply_fetch_bump,
    _embed_or_raise,
    _validate_required_text,
    _validate_limit,
    _validate_confidence,
    _validate_optional_text,
    _validate_string_list,
    _vector_search_enabled,
    COLD_SEARCH_ENABLED,
    MAX_QUERY_LENGTH,
    MAX_RESULT_LIMIT,
    MAX_DOMAIN_LENGTH,
    MAX_SHORT_TEXT_LENGTH,
    MAX_TAG_ITEMS,
    MAX_LIST_ITEM_LENGTH,
    SCORE_BUMP_ALPHA,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    service_tool,
)


def _context_tenant_id(context: Optional[RequestContext]) -> Optional[str]:
    return resolve_tenant_id(context)


def validate_memory_search_inputs(
    *,
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    include_chains: bool,
    include_edges: bool,
    max_chains_per_item: int,
    max_edges_per_item: int,
    edge_min_weight: Optional[float],
    edge_rel_type: Optional[str],
    edge_direction: str,
) -> None:
    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    if not isinstance(include_chains, bool):
        raise ValidationIssue(
            "include_chains must be a boolean",
            field="include_chains",
            error_type="invalid_type",
        )
    if not isinstance(include_edges, bool):
        raise ValidationIssue(
            "include_edges must be a boolean",
            field="include_edges",
            error_type="invalid_type",
        )
    _validate_limit(max_chains_per_item, "max_chains_per_item", MAX_RESULT_LIMIT)
    _validate_limit(max_edges_per_item, "max_edges_per_item", MAX_RESULT_LIMIT)
    if edge_min_weight is not None:
        _validate_confidence(edge_min_weight, "edge_min_weight")
    _validate_optional_text(edge_rel_type, "edge_rel_type", MAX_SHORT_TEXT_LENGTH)
    direction_value = (edge_direction or "both").strip().lower()
    if direction_value not in {"both", "out", "in"}:
        raise ValidationIssue(
            "edge_direction must be one of: both, out, in",
            field="edge_direction",
            error_type="invalid_value",
        )


def validate_search_cold_inputs(
    *,
    query: str,
    top_k: int,
    type_filter: Optional[str],
    source: Optional[str],
    tags: Optional[list[str]],
) -> None:
    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(top_k, "top_k", 50)
    _validate_optional_text(type_filter, "type_filter", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(source, "source", MAX_SHORT_TEXT_LENGTH)
    _validate_string_list(tags, "tags", MAX_TAG_ITEMS, MAX_LIST_ITEM_LENGTH)


def validate_search_cold_dates(
    date_from: Optional[str],
    date_to: Optional[str],
) -> tuple[Optional[datetime], Optional[datetime]]:
    dt_from = None
    dt_to = None
    if date_from:
        _validate_optional_text(date_from, "date_from", MAX_SHORT_TEXT_LENGTH)
        dt_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
    if date_to:
        _validate_optional_text(date_to, "date_to", MAX_SHORT_TEXT_LENGTH)
        dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
    return dt_from, dt_to


def validate_memory_recall_inputs(
    *,
    domain: Optional[str],
    min_confidence: float,
    limit: int,
    ai_name: Optional[str],
) -> None:
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)


def candidate_timestamp(item: tuple[str, object]) -> float:
    source_type, row = item
    if source_type == "observation":
        ts = row.timestamp
    elif source_type == "pattern":
        ts = row.last_updated
    elif source_type == "concept":
        ts = row.created_at
    else:
        ts = row.created_at
    return ts.timestamp() if ts else 0.0


def metadata_matches_tags(metadata: Optional[object], tags: Iterable[str]) -> bool:
    tag_set = set(tags)
    if not tag_set:
        return True
    if isinstance(metadata, list):
        return bool(tag_set.intersection({str(item) for item in metadata}))
    if isinstance(metadata, dict):
        meta_tags = metadata.get("tags", [])
        if isinstance(meta_tags, list):
            return bool(tag_set.intersection({str(item) for item in meta_tags}))
    return False

def vector_search_rows(
    db,
    *,
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float],
    max_score: Optional[float],
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
):
    ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
    query_embedding = _embed_or_raise(query)

    ai_filter_sql = ""
    if ai_mode != "shared_all" and ai_instance_id and tenant_id:
        ai_filter_sql = """
        AND (
            (e.source_type = 'observation' AND (
                o.ai_instance_id = :ai_instance_id
                OR (
                    :ai_mode = 'selective'
                    AND EXISTS (
                        SELECT 1 FROM ai_entity_shares s
                        WHERE s.tenant_id = :tenant_id
                        AND s.entity_type = 'observation'
                        AND s.entity_id = o.id
                        AND s.shared_with_ai_instance_id = :ai_instance_id
                    )
                )
            ))
            OR (e.source_type = 'pattern' AND (
                p.ai_instance_id = :ai_instance_id
                OR (
                    :ai_mode = 'selective'
                    AND EXISTS (
                        SELECT 1 FROM ai_entity_shares s
                        WHERE s.tenant_id = :tenant_id
                        AND s.entity_type = 'pattern'
                        AND s.entity_id = p.id
                        AND s.shared_with_ai_instance_id = :ai_instance_id
                    )
                )
            ))
            OR (e.source_type = 'concept' AND (
                c.ai_instance_id = :ai_instance_id
                OR (
                    :ai_mode = 'selective'
                    AND EXISTS (
                        SELECT 1 FROM ai_entity_shares s
                        WHERE s.tenant_id = :tenant_id
                        AND s.entity_type = 'concept'
                        AND s.entity_id = c.id
                        AND s.shared_with_ai_instance_id = :ai_instance_id
                    )
                )
            ))
            OR (e.source_type = 'document' AND (
                EXISTS (
                    SELECT 1 FROM ai_entity_shares s
                    WHERE s.tenant_id = :tenant_id
                    AND s.entity_type = 'document'
                    AND s.entity_id = d.id
                    AND s.shared_with_ai_instance_id = :ai_instance_id
                )
            ))
        )
        """

    sql = text(
        """
        SELECT 
            e.source_type,
            e.source_id,
            CASE 
                WHEN e.source_type = 'observation' THEN o.observation
                WHEN e.source_type = 'pattern' THEN p.pattern_text
                WHEN e.source_type = 'concept' THEN c.description
                WHEN e.source_type = 'document' THEN d.content_summary
            END as content,
            CASE 
                WHEN e.source_type = 'observation' THEN o.confidence
                WHEN e.source_type = 'pattern' THEN p.confidence
                ELSE 1.0
            END as confidence,
            CASE 
                WHEN e.source_type = 'observation' THEN o.domain
                WHEN e.source_type = 'pattern' THEN p.category
                WHEN e.source_type = 'concept' THEN c.domain
                WHEN e.source_type = 'document' THEN d.doc_type
            END as domain_or_category,
            CASE 
                WHEN e.source_type = 'observation' THEN o.timestamp
                WHEN e.source_type = 'pattern' THEN p.last_updated
                WHEN e.source_type = 'concept' THEN c.created_at
                WHEN e.source_type = 'document' THEN d.created_at
            END as timestamp,
            CASE 
                WHEN e.source_type = 'observation' THEN o.evidence
                WHEN e.source_type = 'pattern' THEN p.evidence_observation_ids
                WHEN e.source_type = 'concept' THEN c.metadata
                WHEN e.source_type = 'document' THEN d.key_concepts
            END as metadata,
            CASE 
                WHEN e.source_type = 'observation' THEN o.score
                WHEN e.source_type = 'pattern' THEN p.score
                WHEN e.source_type = 'concept' THEN c.score
                WHEN e.source_type = 'document' THEN d.score
            END as score,
            CASE 
                WHEN e.source_type = 'observation' THEN o.tier
                WHEN e.source_type = 'pattern' THEN p.tier
                WHEN e.source_type = 'concept' THEN c.tier
                WHEN e.source_type = 'document' THEN d.tier
            END as tier,
            CASE 
                WHEN e.source_type = 'observation' THEN obs_ai.name
                WHEN e.source_type = 'pattern' THEN pat_ai.name
                WHEN e.source_type = 'concept' THEN con_ai.name
                ELSE NULL
            END as ai_name,
            CASE 
                WHEN e.source_type = 'observation' THEN obs_s.title
                WHEN e.source_type = 'pattern' THEN pat_s.title
                ELSE NULL
            END as session_title,
            CASE 
                WHEN e.source_type = 'concept' THEN c.name
                WHEN e.source_type = 'pattern' THEN p.pattern_name
                WHEN e.source_type = 'document' THEN d.title
                ELSE NULL
            END as item_name,
            1 - (e.embedding <=> cast(:embedding as vector)) as similarity
        FROM embeddings e
        LEFT JOIN observations o ON e.source_type = 'observation' AND e.source_id = o.id
        LEFT JOIN patterns p ON e.source_type = 'pattern' AND e.source_id = p.id
        LEFT JOIN concepts c ON e.source_type = 'concept' AND e.source_id = c.id
        LEFT JOIN documents d ON e.source_type = 'document' AND e.source_id = d.id
        LEFT JOIN ai_instances obs_ai
            ON o.ai_instance_id = obs_ai.id
            AND (:tenant_id IS NULL OR obs_ai.tenant_id = :tenant_id)
        LEFT JOIN ai_instances pat_ai
            ON p.ai_instance_id = pat_ai.id
            AND (:tenant_id IS NULL OR pat_ai.tenant_id = :tenant_id)
        LEFT JOIN ai_instances con_ai
            ON c.ai_instance_id = con_ai.id
            AND (:tenant_id IS NULL OR con_ai.tenant_id = :tenant_id)
        LEFT JOIN sessions obs_s
            ON o.session_id = obs_s.id
            AND (:tenant_id IS NULL OR obs_s.tenant_id = :tenant_id)
        LEFT JOIN sessions pat_s
            ON p.session_id = pat_s.id
            AND (:tenant_id IS NULL OR pat_s.tenant_id = :tenant_id)
        WHERE (
            CASE 
                WHEN e.source_type = 'observation' THEN o.confidence
                WHEN e.source_type = 'pattern' THEN p.confidence
                ELSE 1.0
            END >= :min_confidence
        )
        AND (
            :domain IS NULL 
            OR (e.source_type = 'observation' AND o.domain = :domain)
        )
        AND (
            :tier IS NULL
            OR (e.source_type = 'observation' AND o.tier = :tier)
            OR (e.source_type = 'pattern' AND p.tier = :tier)
            OR (e.source_type = 'concept' AND c.tier = :tier)
            OR (e.source_type = 'document' AND d.tier = :tier)
        )
        AND (
            :min_score IS NULL
            OR (
                CASE 
                    WHEN e.source_type = 'observation' THEN o.score
                    WHEN e.source_type = 'pattern' THEN p.score
                    WHEN e.source_type = 'concept' THEN c.score
                    WHEN e.source_type = 'document' THEN d.score
                END >= :min_score
            )
        )
        AND (
            :max_score IS NULL
            OR (
                CASE 
                    WHEN e.source_type = 'observation' THEN o.score
                    WHEN e.source_type = 'pattern' THEN p.score
                    WHEN e.source_type = 'concept' THEN c.score
                    WHEN e.source_type = 'document' THEN d.score
                END <= :max_score
            )
        )
        AND (
            :tenant_id IS NULL
            OR (
                (e.source_type = 'observation' AND o.tenant_id = :tenant_id)
                OR (e.source_type = 'pattern' AND p.tenant_id = :tenant_id)
                OR (e.source_type = 'concept' AND c.tenant_id = :tenant_id)
                OR (e.source_type = 'document' AND d.tenant_id = :tenant_id)
            )
        )
        """
        + ai_filter_sql
        + """
        ORDER BY e.embedding <=> cast(:embedding as vector)
        LIMIT :limit
    """
    )

    return db.execute(
        sql,
        {
            "embedding": str(query_embedding),
            "min_confidence": min_confidence,
            "domain": domain,
            "limit": limit,
            "tier": tier_filter.value if tier_filter else None,
            "min_score": min_score,
            "max_score": max_score,
            "tenant_id": tenant_id,
            "ai_instance_id": ai_instance_id,
            "ai_mode": ai_mode,
        },
    ).fetchall()

def keyword_candidates(
    db,
    *,
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float],
    max_score: Optional[float],
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
) -> List[tuple[str, object]]:
    ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
    pattern = f"%{query}%"
    candidates: list[tuple[str, object]] = []

    def add_candidates(
        model,
        source_type: str,
        content_attr: str,
        timestamp_attr: str,
    ) -> None:
        query_builder = db.query(model)
        if tenant_id:
            query_builder = query_builder.filter(model.tenant_id == tenant_id)
        query_builder = apply_ai_memory_filter(
            query_builder,
            model=model,
            entity_type=source_type,
            tenant_id=tenant_id,
            ai_instance_id=ai_instance_id,
            mode=ai_mode,
        )
        if tier_filter is not None:
            query_builder = query_builder.filter(model.tier == tier_filter)
        if min_score is not None:
            query_builder = query_builder.filter(model.score >= min_score)
        if max_score is not None:
            query_builder = query_builder.filter(model.score <= max_score)
        if source_type in {"observation", "pattern"} and min_confidence > 0:
            query_builder = query_builder.filter(model.confidence >= min_confidence)
        if source_type == "observation" and domain:
            query_builder = query_builder.filter(model.domain == domain)

        content_column = getattr(model, content_attr)
        query_builder = query_builder.filter(content_column.ilike(pattern))
        query_builder = query_builder.order_by(desc(getattr(model, timestamp_attr)))
        rows = query_builder.limit(limit).all()
        for row in rows:
            candidates.append((source_type, row))

    add_candidates(Observation, "observation", "observation", "timestamp")
    add_candidates(Pattern, "pattern", "pattern_text", "last_updated")
    add_candidates(Concept, "concept", "description", "created_at")
    add_candidates(Document, "document", "content_summary", "created_at")

    return candidates


def recall_results(
    db,
    *,
    domain: Optional[str],
    min_confidence: float,
    limit: int,
    ai_name: Optional[str],
    include_cold: bool,
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
):
    ai_join = Observation.ai_instance_id == AIInstance.id
    session_join = Observation.session_id == Session.id
    if tenant_id:
        ai_join = and_(ai_join, AIInstance.tenant_id == tenant_id)
        session_join = and_(session_join, Session.tenant_id == tenant_id)
    query = db.query(Observation).join(AIInstance, ai_join, isouter=True).join(
        Session, session_join, isouter=True
    )
    if tenant_id:
        query = query.filter(Observation.tenant_id == tenant_id)
    ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
    query = apply_ai_memory_filter(
        query,
        model=Observation,
        entity_type="observation",
        tenant_id=tenant_id,
        ai_instance_id=ai_instance_id,
        mode=ai_mode,
    )

    if domain:
        query = query.filter(Observation.domain == domain)
    if min_confidence > 0:
        query = query.filter(Observation.confidence >= min_confidence)
    if ai_name:
        query = query.filter(AIInstance.name == ai_name)
    if not include_cold:
        query = query.filter(Observation.tier == MemoryTier.hot)

    return query.order_by(desc(Observation.timestamp)).limit(limit).all()


def bump_vector_results(db, results, tenant_id: Optional[str], bump_score: bool) -> None:
    if not bump_score:
        return
    for row in results:
        if row.tier != MemoryTier.hot.value:
            continue
        if row.source_type == "observation":
            record_query = db.query(Observation).filter(Observation.id == row.source_id)
            if tenant_id:
                record_query = record_query.filter(Observation.tenant_id == tenant_id)
            record = record_query.first()
        elif row.source_type == "pattern":
            record_query = db.query(Pattern).filter(Pattern.id == row.source_id)
            if tenant_id:
                record_query = record_query.filter(Pattern.tenant_id == tenant_id)
            record = record_query.first()
        elif row.source_type == "concept":
            record_query = db.query(Concept).filter(Concept.id == row.source_id)
            if tenant_id:
                record_query = record_query.filter(Concept.tenant_id == tenant_id)
            record = record_query.first()
        elif row.source_type == "document":
            record_query = db.query(Document).filter(Document.id == row.source_id)
            if tenant_id:
                record_query = record_query.filter(Document.tenant_id == tenant_id)
            record = record_query.first()
        else:
            record = None
        if record:
            _apply_fetch_bump(record, SCORE_BUMP_ALPHA)
    db.commit()


def bump_keyword_results(db, selected, bump_score: bool) -> None:
    if not bump_score:
        return
    for _, row in selected:
        if row.tier == MemoryTier.hot:
            _apply_fetch_bump(row, SCORE_BUMP_ALPHA)
    db.commit()


def bump_recall_results(db, results, include_cold: bool) -> None:
    if include_cold:
        return
    for obs in results:
        _apply_fetch_bump(obs, SCORE_BUMP_ALPHA)
    db.commit()


def serialize_vector_results(results, include_evidence: bool) -> list[dict]:
    return [
        {
            "source_type": row.source_type,
            "id": row.source_id,
            "ref": f"{row.source_type}:{row.source_id}",
            "content": row.content,
            "snippet": (row.content or "")[:200],
            "name": row.item_name,
            "confidence": row.confidence,
            "domain": row.domain_or_category,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "metadata": row.metadata if include_evidence else None,
            "ai_name": row.ai_name,
            "session_title": row.session_title,
            "similarity": float(row.similarity),
            "score": float(row.score) if row.score is not None else None,
            "tier": row.tier,
        }
        for row in results
    ]


def serialize_keyword_results(selected, include_evidence: bool) -> list[dict]:
    results = []
    for source_type, row in selected:
        if source_type == "observation":
            content = row.observation
            confidence = row.confidence
            domain_or_category = row.domain
            timestamp = row.timestamp
            metadata = row.evidence
            ai_name = row.ai_instance.name if row.ai_instance else None
            session_title = row.session.title if row.session else None
            item_name = None
        elif source_type == "pattern":
            content = row.pattern_text
            confidence = row.confidence
            domain_or_category = row.category
            timestamp = row.last_updated
            metadata = row.evidence_observation_ids
            ai_name = row.ai_instance.name if row.ai_instance else None
            session_title = row.session.title if row.session else None
            item_name = row.pattern_name
        elif source_type == "concept":
            content = row.description
            confidence = 1.0
            domain_or_category = row.domain
            timestamp = row.created_at
            metadata = row.metadata_
            ai_name = row.ai_instance.name if row.ai_instance else None
            session_title = None
            item_name = row.name
        else:
            content = row.content_summary
            confidence = 1.0
            domain_or_category = row.doc_type
            timestamp = row.created_at
            metadata = row.key_concepts
            ai_name = None
            session_title = None
            item_name = row.title

        results.append(
            {
                "source_type": source_type,
                "id": row.id,
                "ref": f"{source_type}:{row.id}",
                "content": content,
                "snippet": (content or "")[:200],
                "name": item_name,
                "confidence": confidence,
                "domain": domain_or_category,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "metadata": metadata if include_evidence else None,
                "ai_name": ai_name,
                "session_title": session_title,
                "similarity": 0.0,
                "score": float(row.score) if row.score is not None else None,
                "tier": row.tier.value if isinstance(row.tier, MemoryTier) else row.tier,
            }
        )

    return results


def serialize_recall_results(results) -> list[dict]:
    return [
        {
            "id": obs.id,
            "observation": obs.observation,
            "confidence": obs.confidence,
            "domain": obs.domain,
            "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
            "evidence": obs.evidence,
            "ai_name": obs.ai_instance.name if obs.ai_instance else None,
            "session_title": obs.session.title if obs.session else None,
        }
        for obs in results
    ]

def _search_memory_impl(
    *,
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    include_evidence: bool = True,
    bump_score: bool = True,
    tenant_id: Optional[str] = None,
    ai_instance_id: Optional[int] = None,
) -> dict:
    if not _vector_search_enabled():
        return _search_memory_keyword_impl(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            tier_filter=tier_filter,
            min_score=min_score,
            max_score=max_score,
            include_evidence=include_evidence,
            bump_score=bump_score,
            tenant_id=tenant_id,
            ai_instance_id=ai_instance_id,
        )

    db = DB.SessionLocal()
    try:
        results = vector_search_rows(
            db,
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            tier_filter=tier_filter,
            min_score=min_score,
            max_score=max_score,
            tenant_id=tenant_id,
            ai_instance_id=ai_instance_id,
        )
        bump_vector_results(db, results, tenant_id, bump_score)
        payload = serialize_vector_results(results, include_evidence)
        return {
            "query": query,
            "count": len(results),
            "results": payload,
        }
    finally:
        db.close()


def _search_memory_keyword_impl(
    *,
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float],
    max_score: Optional[float],
    include_evidence: bool,
    bump_score: bool,
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
) -> dict:
    db = DB.SessionLocal()
    try:
        candidates = keyword_candidates(
            db,
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            tier_filter=tier_filter,
            min_score=min_score,
            max_score=max_score,
            tenant_id=tenant_id,
            ai_instance_id=ai_instance_id,
        )
        candidates.sort(key=candidate_timestamp, reverse=True)
        selected = candidates[:limit]

        bump_keyword_results(db, selected, bump_score)

        results = serialize_keyword_results(selected, include_evidence)
        return {
            "query": query,
            "count": len(results),
            "results": results,
        }
    finally:
        db.close()


def _count_cold_matches(
    tenant_id: Optional[str],
    domain: Optional[str],
    min_confidence: float,
) -> int:
    if not tenant_id:
        return 0
    db = DB.SessionLocal()
    try:
        count = 0
        obs_query = db.query(func.count(Observation.id)).filter(
            Observation.tenant_id == tenant_id,
            Observation.tier == MemoryTier.cold,
            Observation.confidence >= min_confidence,
        )
        if domain:
            obs_query = obs_query.filter(Observation.domain == domain)
        count += obs_query.scalar() or 0

        pat_query = db.query(func.count(Pattern.id)).filter(
            Pattern.tenant_id == tenant_id,
            Pattern.tier == MemoryTier.cold,
            Pattern.confidence >= min_confidence,
        )
        count += pat_query.scalar() or 0

        con_query = db.query(func.count(Concept.id)).filter(
            Concept.tenant_id == tenant_id,
            Concept.tier == MemoryTier.cold,
        )
        count += con_query.scalar() or 0

        return count
    finally:
        db.close()


def _split_ref(ref: str) -> Optional[tuple[str, str]]:
    if not isinstance(ref, str) or ":" not in ref:
        return None
    mem_type, mem_id = ref.split(":", 1)
    mem_type = mem_type.strip().lower()
    mem_id = mem_id.strip()
    if not mem_type or not mem_id:
        return None
    return mem_type, mem_id


def _label_for_result(row: dict) -> Optional[str]:
    if row.get("name"):
        return row["name"]
    if row.get("snippet"):
        return row["snippet"]
    return row.get("content")


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _compute_friction_stats(
    db,
    *,
    tenant_id: Optional[str],
    edge_ids: set[str],
) -> dict[str, dict]:
    if not edge_ids:
        return {}
    import uuid

    edge_values: set[object] = set()
    try:
        python_type = RelationshipResidue.__table__.c.edge_id.type.python_type
    except (NotImplementedError, AttributeError):
        python_type = None
    if python_type is uuid.UUID:
        for value in edge_ids:
            try:
                edge_values.add(uuid.UUID(value))
            except ValueError:
                continue
    else:
        edge_values = set(edge_ids)
    if not edge_values:
        return {}

    query = db.query(RelationshipResidue).filter(RelationshipResidue.edge_id.in_(edge_values))
    if tenant_id:
        query = query.filter(RelationshipResidue.tenant_id == tenant_id)
    rows = query.all()
    stats: dict[str, dict] = {}
    for row in rows:
        edge_key = str(row.edge_id)
        entry = stats.setdefault(
            edge_key,
            {
                "count": 0,
                "last_residue_at": None,
                "alt_total": 0,
                "confidence_values": [],
            },
        )
        entry["count"] += 1
        if row.created_at and (
            entry["last_residue_at"] is None or row.created_at > entry["last_residue_at"]
        ):
            entry["last_residue_at"] = row.created_at
        if isinstance(row.alternatives_considered, list):
            entry["alt_total"] += len(row.alternatives_considered)
        metrics = row.friction_metrics or {}
        if isinstance(metrics, dict):
            trajectory = metrics.get("confidence_trajectory")
            if isinstance(trajectory, list):
                for value in trajectory:
                    if isinstance(value, (int, float)):
                        entry["confidence_values"].append(float(value))

    computed: dict[str, dict] = {}
    for edge_key, entry in stats.items():
        count = entry["count"]
        avg_alt = (entry["alt_total"] / count) if count else 0.0
        variance = _variance(entry["confidence_values"])
        friction_score = count + 0.25 * avg_alt + 0.5 * variance
        computed[edge_key] = {
            "friction_residue_count": count,
            "last_residue_at": entry["last_residue_at"].isoformat()
            if entry["last_residue_at"]
            else None,
            "friction_score": friction_score,
        }
    return computed


def _attach_chain_metadata(
    db,
    *,
    tenant_id: Optional[str],
    results: list[dict],
    max_chains_per_item: int,
) -> None:
    if not results:
        return

    refs = [row["ref"] for row in results if row.get("ref")]
    if not refs:
        return

    query = (
        db.query(MemoryChainItem, MemoryChain)
        .join(MemoryChain, MemoryChainItem.chain_id == MemoryChain.id)
        .filter(MemoryChainItem.memory_id.in_(refs))
    )
    if tenant_id:
        query = query.filter(MemoryChainItem.tenant_id == tenant_id).filter(
            MemoryChain.tenant_id == tenant_id
        )
    query = query.order_by(MemoryChainItem.seq.asc(), MemoryChain.created_at.desc())
    rows = query.all()

    chains_by_ref: dict[str, list[dict]] = {ref: [] for ref in refs}
    for item, chain in rows:
        items = chains_by_ref.setdefault(item.memory_id, [])
        if len(items) >= max_chains_per_item:
            continue
        items.append(
            {
                "chain_id": str(chain.id),
                "title": chain.title,
                "kind": chain.kind,
                "role": item.role,
                "seq": item.seq,
            }
        )

    for row in results:
        row["chains"] = chains_by_ref.get(row.get("ref"), [])


def _attach_edge_metadata(
    db,
    *,
    tenant_id: Optional[str],
    results: list[dict],
    max_edges_per_item: int,
    edge_min_weight: Optional[float],
    edge_rel_type: Optional[str],
    edge_direction: str,
) -> None:
    if not results:
        return

    refs = [row.get("ref") for row in results if row.get("ref")]
    if not refs:
        return
    refs_set = set(refs)
    refs_by_type: dict[str, set[str]] = {}
    for ref in refs_set:
        parts = _split_ref(ref)
        if not parts:
            continue
        mem_type, mem_id = parts
        refs_by_type.setdefault(mem_type, set()).add(mem_id)

    if not refs_by_type:
        for row in results:
            row["edges"] = []
        return

    direction_value = (edge_direction or "both").strip().lower()
    conditions = []
    if direction_value in {"out", "both"}:
        for mem_type, ids in refs_by_type.items():
            conditions.append(
                and_(MemoryRelationship.from_type == mem_type, MemoryRelationship.from_id.in_(ids))
            )
    if direction_value in {"in", "both"}:
        for mem_type, ids in refs_by_type.items():
            conditions.append(
                and_(MemoryRelationship.to_type == mem_type, MemoryRelationship.to_id.in_(ids))
            )

    if not conditions:
        for row in results:
            row["edges"] = []
        return

    query = db.query(MemoryRelationship).filter(or_(*conditions))
    if tenant_id:
        query = query.filter(MemoryRelationship.tenant_id == tenant_id)
    if edge_rel_type:
        query = query.filter(MemoryRelationship.rel_type == edge_rel_type)
    if edge_min_weight is not None:
        query = query.filter(MemoryRelationship.weight >= edge_min_weight)

    rows = query.order_by(MemoryRelationship.created_at.desc()).all()

    label_map = {row["ref"]: _label_for_result(row) for row in results if row.get("ref")}
    edges_by_ref: dict[str, list[dict]] = {ref: [] for ref in refs_set}
    for rel in rows:
        from_ref = f"{rel.from_type}:{rel.from_id}"
        to_ref = f"{rel.to_type}:{rel.to_id}"
        edge_id = str(rel.id)

        if direction_value in {"out", "both"} and from_ref in refs_set:
            edges_by_ref[from_ref].append(
                {
                    "edge_id": edge_id,
                    "rel_type": rel.rel_type,
                    "direction": "out",
                    "weight": rel.weight,
                    "description": rel.description,
                    "to_ref": to_ref,
                    "to_label": label_map.get(to_ref),
                    "metadata": rel.metadata_,
                    "_created_at": rel.created_at,
                }
            )
        if direction_value in {"in", "both"} and to_ref in refs_set:
            edges_by_ref[to_ref].append(
                {
                    "edge_id": edge_id,
                    "rel_type": rel.rel_type,
                    "direction": "in",
                    "weight": rel.weight,
                    "description": rel.description,
                    "from_ref": from_ref,
                    "from_label": label_map.get(from_ref),
                    "metadata": rel.metadata_,
                    "_created_at": rel.created_at,
                }
            )

    friction_stats = _compute_friction_stats(
        db,
        tenant_id=tenant_id,
        edge_ids={str(rel.id) for rel in rows},
    )

    for ref, entries in edges_by_ref.items():
        for entry in entries:
            created_at = entry.get("_created_at")
            entry["_created_at_epoch"] = created_at.timestamp() if created_at else 0.0
        entries.sort(
            key=lambda entry: (
                entry["weight"] if entry["weight"] is not None else -1.0,
                entry["_created_at_epoch"],
            ),
            reverse=True,
        )
        trimmed = entries[:max_edges_per_item]
        for entry in trimmed:
            entry.pop("_created_at", None)
            entry.pop("_created_at_epoch", None)
            stats = friction_stats.get(entry.get("edge_id"))
            if stats:
                entry.update(stats)
            else:
                entry["friction_residue_count"] = 0
                entry["last_residue_at"] = None
                entry["friction_score"] = 0.0
        edges_by_ref[ref] = trimmed

    for row in results:
        row["edges"] = edges_by_ref.get(row.get("ref"), [])

@service_tool
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None,
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
    include_chains: bool = False,
    include_edges: bool = False,
    max_chains_per_item: int = 3,
    max_edges_per_item: int = 5,
    edge_min_weight: Optional[float] = None,
    edge_rel_type: Optional[str] = None,
    edge_direction: str = "both",
    context: Optional[RequestContext] = None,
) -> dict:
    validate_memory_search_inputs(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        domain=domain,
        include_chains=include_chains,
        include_edges=include_edges,
        max_chains_per_item=max_chains_per_item,
        max_edges_per_item=max_edges_per_item,
        edge_min_weight=edge_min_weight,
        edge_rel_type=edge_rel_type,
        edge_direction=edge_direction,
    )

    tier_filter = None if include_cold else MemoryTier.hot
    tenant_id = _context_tenant_id(context)
    results = _search_memory_impl(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        domain=domain,
        tier_filter=tier_filter,
        include_evidence=True,
        bump_score=True,
        tenant_id=tenant_id,
        ai_instance_id=ai_instance_id,
    )
    for row in results.get("results", []):
        if not row.get("ref"):
            row["ref"] = f"{row.get('source_type', '')}:{row.get('id')}"

    if not include_cold:
        cold_count = _count_cold_matches(tenant_id, domain, min_confidence)
        if cold_count > 0:
            results["cold_matches_available"] = cold_count
            results["cold_hint"] = "Use include_cold=true or search_cold_memory() to search archived memories"

    if include_chains or include_edges:
        db = DB.SessionLocal()
        try:
            if include_chains:
                _attach_chain_metadata(
                    db,
                    tenant_id=tenant_id,
                    results=results.get("results", []),
                    max_chains_per_item=max_chains_per_item,
                )
            if include_edges:
                _attach_edge_metadata(
                    db,
                    tenant_id=tenant_id,
                    results=results.get("results", []),
                    max_edges_per_item=max_edges_per_item,
                    edge_min_weight=edge_min_weight,
                    edge_rel_type=edge_rel_type,
                    edge_direction=edge_direction,
                )
        finally:
            db.close()

    return results


@service_tool
def search_cold_memory(
    query: str,
    top_k: int = 10,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    type_filter: Optional[str] = None,
    source: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags: Optional[List[str]] = None,
    include_evidence: bool = True,
    bump_score: bool = False,
    ai_instance_id: Optional[int] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    if not COLD_SEARCH_ENABLED:
        return {"status": "error", "message": "Cold search is disabled"}

    validate_search_cold_inputs(
        query=query,
        top_k=top_k,
        type_filter=type_filter,
        source=source,
        tags=tags,
    )
    dt_from, dt_to = validate_search_cold_dates(date_from, date_to)

    fetch_limit = min(max(top_k * 5, top_k), MAX_RESULT_LIMIT)
    tenant_id = _context_tenant_id(context)
    results = _search_memory_impl(
        query=query,
        limit=fetch_limit,
        min_confidence=0.0,
        domain=None,
        tier_filter=MemoryTier.cold,
        min_score=min_score,
        max_score=max_score,
        include_evidence=include_evidence or bool(tags),
        bump_score=bump_score,
        tenant_id=tenant_id,
        ai_instance_id=ai_instance_id,
    )

    filtered = []
    for row in results["results"]:
        if type_filter and row["source_type"] != type_filter:
            continue
        if source and row.get("ai_name") != source:
            continue
        if dt_from or dt_to:
            if not row.get("timestamp"):
                continue
            timestamp = datetime.fromisoformat(row["timestamp"])
            if dt_from and timestamp < dt_from:
                continue
            if dt_to and timestamp > dt_to:
                continue
        if tags and not metadata_matches_tags(row.get("metadata"), tags):
            continue
        filtered.append(row)
        if len(filtered) >= top_k:
            break

    return {
        "query": query,
        "count": len(filtered),
        "results": filtered,
    }


@service_tool
def memory_recall(
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    ai_name: Optional[str] = None,
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    validate_memory_recall_inputs(
        domain=domain,
        min_confidence=min_confidence,
        limit=limit,
        ai_name=ai_name,
    )

    db = DB.SessionLocal()
    try:
        tenant_id = _context_tenant_id(context)
        results = recall_results(
            db,
            domain=domain,
            min_confidence=min_confidence,
            limit=limit,
            ai_name=ai_name,
            include_cold=include_cold,
            tenant_id=tenant_id,
            ai_instance_id=ai_instance_id,
        )

        bump_recall_results(db, results, include_cold)

        return {
            "count": len(results),
            "filters": {
                "domain": domain,
                "min_confidence": min_confidence,
                "ai_name": ai_name,
            },
            "results": serialize_recall_results(results),
        }
    finally:
        db.close()


@service_tool
def memory_stats(context: Optional[RequestContext] = None) -> dict:
    db = DB.SessionLocal()
    try:
        tenant_id = _context_tenant_id(context)
        obs_query = db.query(func.count(Observation.id))
        pattern_query = db.query(func.count(Pattern.id))
        concept_query = db.query(func.count(Concept.id))
        document_query = db.query(func.count(Document.id))
        summary_query = db.query(func.count(MemorySummary.id))
        session_query = db.query(func.count(Session.id))
        ai_query = db.query(func.count(AIInstance.id))
        active_agents_query = db.query(func.count(AIInstance.id)).filter(
            AIInstance.merged_into_ai_instance_id.is_(None),
            AIInstance.is_archived.is_(False),
        )
        embedding_query = db.query(func.count(Embedding.source_id))

        if tenant_id:
            obs_query = obs_query.filter(Observation.tenant_id == tenant_id)
            pattern_query = pattern_query.filter(Pattern.tenant_id == tenant_id)
            concept_query = concept_query.filter(Concept.tenant_id == tenant_id)
            document_query = document_query.filter(Document.tenant_id == tenant_id)
            summary_query = summary_query.filter(MemorySummary.tenant_id == tenant_id)
            session_query = session_query.filter(Session.tenant_id == tenant_id)
            ai_query = ai_query.filter(AIInstance.tenant_id == tenant_id)
            active_agents_query = active_agents_query.filter(AIInstance.tenant_id == tenant_id)
            embedding_query = embedding_query.filter(Embedding.tenant_id == tenant_id)

        obs_count = obs_query.scalar()
        pattern_count = pattern_query.scalar()
        concept_count = concept_query.scalar()
        document_count = document_query.scalar()
        summary_count = summary_query.scalar()
        session_count = session_query.scalar()
        ai_count = ai_query.scalar()
        active_agents_count = active_agents_query.scalar()
        embedding_count = embedding_query.scalar()

        ai_instances_query = db.query(AIInstance)
        if tenant_id:
            ai_instances_query = ai_instances_query.filter(AIInstance.tenant_id == tenant_id)
        ai_instances = ai_instances_query.all()

        domains_query = db.query(Observation.domain, func.count(Observation.id))
        if tenant_id:
            domains_query = domains_query.filter(Observation.tenant_id == tenant_id)
        domains = domains_query.group_by(Observation.domain).all()

        def _count(model, tier: Optional[MemoryTier] = None) -> int:
            query = db.query(func.count(model.id))
            if tenant_id:
                query = query.filter(model.tenant_id == tenant_id)
            if tier is not None:
                query = query.filter(model.tier == tier)
            return query.scalar()

        hot_counts = {
            "observations": _count(Observation, MemoryTier.hot),
            "patterns": _count(Pattern, MemoryTier.hot),
            "concepts": _count(Concept, MemoryTier.hot),
            "documents": _count(Document, MemoryTier.hot),
            "summaries": _count(MemorySummary, MemoryTier.hot),
        }
        cold_counts = {
            "observations": _count(Observation, MemoryTier.cold),
            "patterns": _count(Pattern, MemoryTier.cold),
            "concepts": _count(Concept, MemoryTier.cold),
            "documents": _count(Document, MemoryTier.cold),
            "summaries": _count(MemorySummary, MemoryTier.cold),
        }

        return {
            "status": "healthy",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "counts": {
                "observations": obs_count,
                "patterns": pattern_count,
                "concepts": concept_count,
                "documents": document_count,
                "summaries": summary_count,
                "sessions": session_count,
                "ai_instances": ai_count,
                "active_agents": active_agents_count,
                "embeddings": embedding_count,
            },
            "tiers": {
                "hot": hot_counts,
                "cold": cold_counts,
            },
            "ai_instances": [{"name": ai.name, "platform": ai.platform} for ai in ai_instances],
            "domains": {domain or "untagged": count for domain, count in domains},
        }
    finally:
        db.close()

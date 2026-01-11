"""
Search and recall services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import text, func, desc

from core.context import RequestContext
from core.db import DB
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
)
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

def _search_memory_impl(
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    include_evidence: bool = True,
    bump_score: bool = True,
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
        )

    db = DB.SessionLocal()
    try:
        query_embedding = _embed_or_raise(query)

        sql = text("""
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
            LEFT JOIN ai_instances obs_ai ON o.ai_instance_id = obs_ai.id
            LEFT JOIN ai_instances pat_ai ON p.ai_instance_id = pat_ai.id
            LEFT JOIN ai_instances con_ai ON c.ai_instance_id = con_ai.id
            LEFT JOIN sessions obs_s ON o.session_id = obs_s.id
            LEFT JOIN sessions pat_s ON p.session_id = pat_s.id
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
            ORDER BY e.embedding <=> cast(:embedding as vector)
            LIMIT :limit
        """)

        results = db.execute(sql, {
            "embedding": str(query_embedding),
            "min_confidence": min_confidence,
            "domain": domain,
            "limit": limit,
            "tier": tier_filter.value if tier_filter else None,
            "min_score": min_score,
            "max_score": max_score,
        }).fetchall()

        if bump_score:
            for row in results:
                if row.tier != MemoryTier.hot.value:
                    continue
                if row.source_type == 'observation':
                    record = db.query(Observation).filter(Observation.id == row.source_id).first()
                elif row.source_type == 'pattern':
                    record = db.query(Pattern).filter(Pattern.id == row.source_id).first()
                elif row.source_type == 'concept':
                    record = db.query(Concept).filter(Concept.id == row.source_id).first()
                elif row.source_type == 'document':
                    record = db.query(Document).filter(Document.id == row.source_id).first()
                else:
                    record = None
                if record:
                    _apply_fetch_bump(record, SCORE_BUMP_ALPHA)
            db.commit()

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "source_type": row.source_type,
                    "id": row.source_id,
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
        }
    finally:
        db.close()

def _search_memory_keyword_impl(
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float],
    max_score: Optional[float],
    include_evidence: bool,
    bump_score: bool,
) -> dict:
    db = DB.SessionLocal()
    try:
        pattern = f"%{query}%"
        candidates: list[tuple[str, object]] = []

        def add_candidates(
            model,
            source_type: str,
            content_attr: str,
            timestamp_attr: str,
        ) -> None:
            query_builder = db.query(model)
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

        candidates.sort(key=candidate_timestamp, reverse=True)
        selected = candidates[:limit]

        if bump_score:
            for source_type, row in selected:
                if row.tier == MemoryTier.hot:
                    _apply_fetch_bump(row, SCORE_BUMP_ALPHA)
            db.commit()

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

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }
    finally:
        db.close()

@service_tool
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Unified semantic search across all memory types (observations, patterns, concepts, documents).
    
    Args:
        query: Search query text
        limit: Maximum results to return (default 5)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        domain: Optional domain filter (applies to observations only)
        include_cold: Include cold tier records
    
    Returns:
        List of matching items from all sources with similarity scores and source_type
    """
    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)

    tier_filter = None if include_cold else MemoryTier.hot
    return _search_memory_impl(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        domain=domain,
        tier_filter=tier_filter,
        include_evidence=True,
        bump_score=True,
    )

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
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Explicit search over cold-tier memory records.
    """
    if not COLD_SEARCH_ENABLED:
        return {"status": "error", "message": "Cold search is disabled"}

    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(top_k, "top_k", 50)
    _validate_optional_text(type_filter, "type_filter", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(source, "source", MAX_SHORT_TEXT_LENGTH)
    _validate_string_list(tags, "tags", MAX_TAG_ITEMS, MAX_LIST_ITEM_LENGTH)

    dt_from = None
    dt_to = None
    if date_from:
        _validate_optional_text(date_from, "date_from", MAX_SHORT_TEXT_LENGTH)
        dt_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
    if date_to:
        _validate_optional_text(date_to, "date_to", MAX_SHORT_TEXT_LENGTH)
        dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

    fetch_limit = min(max(top_k * 5, top_k), MAX_RESULT_LIMIT)
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
        if tags:
            metadata = row.get("metadata") or []
            tag_set = set(tags)
            match = False
            if isinstance(metadata, list):
                match = bool(tag_set.intersection({str(item) for item in metadata}))
            elif isinstance(metadata, dict):
                meta_tags = metadata.get("tags", [])
                if isinstance(meta_tags, list):
                    match = bool(tag_set.intersection({str(item) for item in meta_tags}))
            if not match:
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
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Recall observations by domain and/or confidence filter.
    
    Args:
        domain: Filter by domain/category
        min_confidence: Minimum confidence threshold
        limit: Maximum results (default 10)
        ai_name: Filter by AI instance name
        include_cold: Include cold tier records
    
    Returns:
        List of matching observations
    """
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        query = db.query(Observation).join(
            AIInstance, Observation.ai_instance_id == AIInstance.id, isouter=True
        ).join(
            Session, Observation.session_id == Session.id, isouter=True
        )
        
        if domain:
            query = query.filter(Observation.domain == domain)
        if min_confidence > 0:
            query = query.filter(Observation.confidence >= min_confidence)
        if ai_name:
            query = query.filter(AIInstance.name == ai_name)
        if not include_cold:
            query = query.filter(Observation.tier == MemoryTier.hot)
        
        results = query.order_by(desc(Observation.timestamp)).limit(limit).all()
        
        if not include_cold:
            for obs in results:
                _apply_fetch_bump(obs, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "count": len(results),
            "filters": {
                "domain": domain,
                "min_confidence": min_confidence,
                "ai_name": ai_name
            },
            "results": [
                {
                    "id": obs.id,
                    "observation": obs.observation,
                    "confidence": obs.confidence,
                    "domain": obs.domain,
                    "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
                    "evidence": obs.evidence,
                    "ai_name": obs.ai_instance.name if obs.ai_instance else None,
                    "session_title": obs.session.title if obs.session else None
                }
                for obs in results
            ]
        }
    finally:
        db.close()

@service_tool
def memory_stats(context: Optional[RequestContext] = None) -> dict:
    """
    Get memory system statistics.
    
    Returns:
        Counts and statistics about stored data
    """
    db = DB.SessionLocal()
    try:
        obs_count = db.query(func.count(Observation.id)).scalar()
        pattern_count = db.query(func.count(Pattern.id)).scalar()
        concept_count = db.query(func.count(Concept.id)).scalar()
        document_count = db.query(func.count(Document.id)).scalar()
        summary_count = db.query(func.count(MemorySummary.id)).scalar()
        session_count = db.query(func.count(Session.id)).scalar()
        ai_count = db.query(func.count(AIInstance.id)).scalar()
        embedding_count = db.query(func.count(Embedding.source_id)).scalar()
        
        # Get AI instances
        ai_instances = db.query(AIInstance).all()
        
        # Get domain distribution
        domains = db.query(
            Observation.domain, func.count(Observation.id)
        ).group_by(Observation.domain).all()
        
        hot_counts = {
            "observations": db.query(func.count(Observation.id)).filter(Observation.tier == MemoryTier.hot).scalar(),
            "patterns": db.query(func.count(Pattern.id)).filter(Pattern.tier == MemoryTier.hot).scalar(),
            "concepts": db.query(func.count(Concept.id)).filter(Concept.tier == MemoryTier.hot).scalar(),
            "documents": db.query(func.count(Document.id)).filter(Document.tier == MemoryTier.hot).scalar(),
            "summaries": db.query(func.count(MemorySummary.id)).filter(MemorySummary.tier == MemoryTier.hot).scalar(),
        }
        cold_counts = {
            "observations": db.query(func.count(Observation.id)).filter(Observation.tier == MemoryTier.cold).scalar(),
            "patterns": db.query(func.count(Pattern.id)).filter(Pattern.tier == MemoryTier.cold).scalar(),
            "concepts": db.query(func.count(Concept.id)).filter(Concept.tier == MemoryTier.cold).scalar(),
            "documents": db.query(func.count(Document.id)).filter(Document.tier == MemoryTier.cold).scalar(),
            "summaries": db.query(func.count(MemorySummary.id)).filter(MemorySummary.tier == MemoryTier.cold).scalar(),
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
                "embeddings": embedding_count
            },
            "tiers": {
                "hot": hot_counts,
                "cold": cold_counts,
            },
            "ai_instances": [
                {"name": ai.name, "platform": ai.platform}
                for ai in ai_instances
            ],
            "domains": {
                domain or "untagged": count 
                for domain, count in domains
            }
        }
    finally:
        db.close()

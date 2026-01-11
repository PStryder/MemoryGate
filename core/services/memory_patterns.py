"""
Pattern services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import desc

from core.context import RequestContext
from core.db import DB
from core.models import Pattern, MemoryTier
from core.services.memory_shared import (
    _apply_fetch_bump,
    _store_embedding,
    _validate_required_text,
    _validate_optional_text,
    _validate_confidence,
    _validate_list,
    _validate_limit,
    _validate_evidence_observation_ids,
    MAX_SHORT_TEXT_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_TEXT_LENGTH,
    MAX_RESULT_LIMIT,
    MAX_RELATIONSHIP_ITEMS,
    SCORE_BUMP_ALPHA,
    service_tool,
)
from core.services.memory_storage import get_or_create_ai_instance, get_or_create_session

@service_tool
def memory_update_pattern(
    category: str,
    pattern_name: str,
    pattern_text: str,
    confidence: float = 0.8,
    evidence_observation_ids: Optional[List[int]] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    conversation_id: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Create or update a pattern (synthesized understanding across observations).
    
    Patterns evolve as understanding grows. This tool performs an upsert:
    - If pattern exists (by category + pattern_name), updates it
    - If pattern doesn't exist, creates it
    
    Args:
        category: Pattern category/domain
        pattern_name: Unique name within category
        pattern_text: The synthesized pattern description (gets embedded)
        confidence: Confidence level 0.0-1.0 (default 0.8)
        evidence_observation_ids: List of observation IDs supporting this pattern
        ai_name: Optional AI instance name
        ai_platform: Optional AI platform
        conversation_id: Optional conversation UUID
    
    Returns:
        Pattern with status (created/updated)
    """
    _validate_required_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_required_text(pattern_name, "pattern_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(pattern_text, "pattern_text", MAX_TEXT_LENGTH)
    _validate_confidence(confidence, "confidence")
    _validate_list(evidence_observation_ids, "evidence_observation_ids", MAX_RELATIONSHIP_ITEMS)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        _validate_evidence_observation_ids(db, evidence_observation_ids)
        # Get AI instance and session if provided
        ai_instance_id = None
        session_id = None
        
        if ai_name and ai_platform:
            ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
            ai_instance_id = ai_instance.id
            
            if conversation_id:
                session = get_or_create_session(db, conversation_id, ai_instance_id=ai_instance_id)
                session_id = session.id
        
        # Check if pattern exists
        existing = db.query(Pattern).filter(
            Pattern.category == category,
            Pattern.pattern_name == pattern_name
        ).first()
        
        if existing:
            # Update existing pattern
            existing.pattern_text = pattern_text
            existing.confidence = confidence
            existing.evidence_observation_ids = evidence_observation_ids or []
            existing.last_updated = datetime.utcnow()
            if ai_instance_id:
                existing.ai_instance_id = ai_instance_id
            if session_id:
                existing.session_id = session_id
            
            db.commit()
            db.refresh(existing)
            
            # Update embedding (if enabled)
            _store_embedding(db, "pattern", existing.id, pattern_text, replace=True)
            
            return {
                "status": "updated",
                "id": existing.id,
                "category": category,
                "pattern_name": pattern_name,
                "confidence": confidence
            }
        else:
            # Create new pattern
            pattern = Pattern(
                category=category,
                pattern_name=pattern_name,
                pattern_text=pattern_text,
                confidence=confidence,
                evidence_observation_ids=evidence_observation_ids or [],
                ai_instance_id=ai_instance_id,
                session_id=session_id
            )
            db.add(pattern)
            db.commit()
            db.refresh(pattern)
            
            # Generate and store embedding (if enabled)
            _store_embedding(db, "pattern", pattern.id, pattern_text)
            
            return {
                "status": "created",
                "id": pattern.id,
                "category": category,
                "pattern_name": pattern_name,
                "confidence": confidence
            }
    finally:
        db.close()

@service_tool
def memory_get_pattern(
    category: str,
    pattern_name: str,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Get a specific pattern by category and name.
    
    Args:
        category: Pattern category
        pattern_name: Pattern name within category
        include_cold: Include cold tier records
    
    Returns:
        Pattern details or not_found status
    """
    _validate_required_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_required_text(pattern_name, "pattern_name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        pattern_query = db.query(Pattern).filter(
            Pattern.category == category,
            Pattern.pattern_name == pattern_name
        )
        if not include_cold:
            pattern_query = pattern_query.filter(Pattern.tier == MemoryTier.hot)
        pattern = pattern_query.first()
        
        if not pattern:
            return {
                "status": "not_found",
                "category": category,
                "pattern_name": pattern_name
            }
        
        if pattern.tier == MemoryTier.hot:
            _apply_fetch_bump(pattern, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "status": "found",
            "id": pattern.id,
            "category": category,
            "pattern_name": pattern_name,
            "pattern_text": pattern.pattern_text,
            "confidence": pattern.confidence,
            "evidence_observation_ids": pattern.evidence_observation_ids,
            "last_updated": pattern.last_updated.isoformat() if pattern.last_updated else None,
            "access_count": pattern.access_count
        }
    finally:
        db.close()

@service_tool
def memory_patterns(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 20,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    List patterns with optional filtering by category and confidence.
    
    Args:
        category: Optional category filter
        min_confidence: Minimum confidence threshold (default 0.0)
        limit: Maximum results (default 20)
        include_cold: Include cold tier records
    
    Returns:
        List of matching patterns
    """
    _validate_optional_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    db = DB.SessionLocal()
    try:
        query = db.query(Pattern)
        
        if category:
            query = query.filter(Pattern.category == category)
        if min_confidence > 0:
            query = query.filter(Pattern.confidence >= min_confidence)
        if not include_cold:
            query = query.filter(Pattern.tier == MemoryTier.hot)
        
        results = query.order_by(desc(Pattern.last_updated)).limit(limit).all()

        if not include_cold:
            for pattern in results:
                _apply_fetch_bump(pattern, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "count": len(results),
            "filters": {
                "category": category,
                "min_confidence": min_confidence
            },
            "results": [
                {
                    "id": p.id,
                    "category": p.category,
                    "pattern_name": p.pattern_name,
                    "pattern_text": p.pattern_text,
                    "confidence": p.confidence,
                    "evidence_count": len(p.evidence_observation_ids) if p.evidence_observation_ids else 0,
                    "last_updated": p.last_updated.isoformat() if p.last_updated else None
                }
                for p in results
            ]
        }
    finally:
        db.close()

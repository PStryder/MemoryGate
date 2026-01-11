"""
Concept and relationship services.
"""

from __future__ import annotations

from typing import Optional

from core.context import RequestContext
from core.db import DB
from core.models import Concept, ConceptAlias, ConceptRelationship, MemoryTier
from core.services.memory_shared import (
    _apply_fetch_bump,
    _store_embedding,
    _validate_required_text,
    _validate_optional_text,
    _validate_confidence,
    _validate_metadata,
    MAX_SHORT_TEXT_LENGTH,
    MAX_CONCEPT_TYPE_LENGTH,
    MAX_TEXT_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_STATUS_LENGTH,
    SCORE_BUMP_ALPHA,
    service_tool,
)
from core.services.memory_storage import get_or_create_ai_instance

@service_tool
def memory_store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[dict] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Store a new concept in the knowledge graph with embedding.
    
    Args:
        name: Concept name (case will be preserved)
        concept_type: Type of concept (project/framework/component/construct/theory)
        description: Description text (this gets embedded for semantic search)
        domain: Optional domain/category
        status: Optional status (active/archived/deprecated/etc)
        metadata: Optional metadata dict
        ai_name: Optional AI instance name
        ai_platform: Optional AI platform
    
    Returns:
        The stored concept with its ID
    """
    _validate_required_text(name, "name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(concept_type, "concept_type", MAX_CONCEPT_TYPE_LENGTH)
    _validate_required_text(description, "description", MAX_TEXT_LENGTH)
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_optional_text(status, "status", MAX_STATUS_LENGTH)
    _validate_metadata(metadata, "metadata")
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        # Get AI instance if provided
        ai_instance_id = None
        if ai_name and ai_platform:
            ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
            ai_instance_id = ai_instance.id
        
        # Check if concept already exists (case-insensitive)
        name_key = name.lower()
        existing = db.query(Concept).filter(Concept.name_key == name_key).first()
        if existing:
            return {
                "status": "error",
                "message": f"Concept '{name}' already exists with ID {existing.id}",
                "existing_id": existing.id
            }
        
        # Create concept
        concept = Concept(
            name=name,
            name_key=name_key,
            type=concept_type,
            description=description,
            domain=domain,
            status=status,
            metadata_=metadata or {},
            ai_instance_id=ai_instance_id
        )
        db.add(concept)
        db.commit()
        db.refresh(concept)
        
        # Generate and store embedding from description (if enabled)
        _store_embedding(db, "concept", concept.id, description)
        
        return {
            "status": "stored",
            "id": concept.id,
            "name": name,
            "type": concept_type,
            "description": description
        }
    finally:
        db.close()

def _resolve_concept_by_name(db, name: str, include_cold: bool) -> Optional[Concept]:
    name_key = name.lower()
    concept_query = db.query(Concept).filter(Concept.name_key == name_key)
    if not include_cold:
        concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
    concept = concept_query.first()

    if not concept:
        alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == name_key).first()
        if alias:
            concept_query = db.query(Concept).filter(Concept.id == alias.concept_id)
            if not include_cold:
                concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
            concept = concept_query.first()

    return concept

@service_tool
def memory_get_concept(
    name: str,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Get a concept by name (case-insensitive, alias-aware).
    
    Args:
        name: Concept name or alias to look up
        include_cold: Include cold tier records
    
    Returns:
        Concept details or None if not found
    """
    _validate_required_text(name, "name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        concept = _resolve_concept_by_name(db, name, include_cold)
        if not concept:
            return {"status": "not_found", "name": name}
        
        if concept.tier == MemoryTier.hot:
            _apply_fetch_bump(concept, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "status": "found",
            "id": concept.id,
            "name": concept.name,
            "type": concept.type,
            "description": concept.description,
            "domain": concept.domain,
            "status": concept.status,
            "metadata": concept.metadata_,
            "access_count": concept.access_count
        }
    finally:
        db.close()

@service_tool
def memory_add_concept_alias(
    concept_name: str,
    alias: str,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Add an alternative name (alias) for a concept.
    
    Args:
        concept_name: Primary concept name
        alias: Alternative name to add
    
    Returns:
        Status of alias creation
    """
    _validate_required_text(concept_name, "concept_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(alias, "alias", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        from models import ConceptAlias
        
        # Find the concept
        concept_key = concept_name.lower()
        concept = db.query(Concept).filter(Concept.name_key == concept_key).first()
        if not concept:
            return {"status": "error", "message": f"Concept '{concept_name}' not found"}
        
        # Check if alias already exists
        alias_key = alias.lower()
        existing_alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == alias_key).first()
        if existing_alias:
            return {"status": "error", "message": f"Alias '{alias}' already exists"}
        
        # Check if alias conflicts with existing concept name
        existing_concept = db.query(Concept).filter(Concept.name_key == alias_key).first()
        if existing_concept:
            return {"status": "error", "message": f"Alias '{alias}' conflicts with existing concept"}
        
        # Create alias
        new_alias = ConceptAlias(
            concept_id=concept.id,
            alias=alias,
            alias_key=alias_key
        )
        db.add(new_alias)
        db.commit()
        
        return {
            "status": "created",
            "concept_id": concept.id,
            "concept_name": concept.name,
            "alias": alias
        }
    finally:
        db.close()

@service_tool
def memory_add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Create a relationship between two concepts.
    
    Args:
        from_concept: Source concept name
        to_concept: Target concept name
        rel_type: Relationship type (enables/version_of/part_of/related_to/implements/demonstrates)
        weight: Relationship strength 0.0-1.0 (default 0.5)
        description: Optional description of relationship
    
    Returns:
        Status of relationship creation
    """
    _validate_required_text(from_concept, "from_concept", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(to_concept, "to_concept", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(rel_type, "rel_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(description, "description", MAX_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        from models import ConceptRelationship
        
        # Valid relationship types
        valid_types = ['enables', 'version_of', 'part_of', 'related_to', 'implements', 'demonstrates']
        if rel_type not in valid_types:
            return {"status": "error", "message": f"Invalid rel_type. Must be one of: {', '.join(valid_types)}"}
        
        # Validate weight
        if not 0.0 <= weight <= 1.0:
            return {"status": "error", "message": "Weight must be between 0.0 and 1.0"}
        
        # Find both concepts (case-insensitive, alias-aware)
        from_key = from_concept.lower()
        to_key = to_concept.lower()
        
        from_c = db.query(Concept).filter(Concept.name_key == from_key).first()
        to_c = db.query(Concept).filter(Concept.name_key == to_key).first()
        
        if not from_c:
            return {"status": "error", "message": f"Source concept '{from_concept}' not found"}
        if not to_c:
            return {"status": "error", "message": f"Target concept '{to_concept}' not found"}
        
        # Check if relationship already exists
        existing = db.query(ConceptRelationship).filter(
            ConceptRelationship.from_concept_id == from_c.id,
            ConceptRelationship.to_concept_id == to_c.id,
            ConceptRelationship.rel_type == rel_type
        ).first()
        
        if existing:
            # Update existing relationship
            existing.weight = weight
            if description:
                existing.description = description
            db.commit()
            return {
                "status": "updated",
                "from": from_c.name,
                "to": to_c.name,
                "rel_type": rel_type,
                "weight": weight
            }
        
        # Create new relationship
        rel = ConceptRelationship(
            from_concept_id=from_c.id,
            to_concept_id=to_c.id,
            rel_type=rel_type,
            weight=weight,
            description=description
        )
        db.add(rel)
        db.commit()
        
        return {
            "status": "created",
            "from": from_c.name,
            "to": to_c.name,
            "rel_type": rel_type,
            "weight": weight
        }
    finally:
        db.close()

@service_tool
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """
    Get concepts related to a given concept.
    
    Args:
        concept_name: Concept to find relationships for
        rel_type: Optional filter by relationship type
        min_weight: Minimum relationship weight (default 0.0)
        include_cold: Include cold tier concepts
    
    Returns:
        List of related concepts with relationship details
    """
    _validate_required_text(concept_name, "concept_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(rel_type, "rel_type", MAX_SHORT_TEXT_LENGTH)
    _validate_confidence(min_weight, "min_weight")

    db = DB.SessionLocal()
    try:
        from models import ConceptRelationship
        
        # Find the concept (alias-aware)
        concept = _resolve_concept_by_name(db, concept_name, include_cold)
        if not concept:
            return {"status": "not_found", "concept": concept_name}
        
        # Get outgoing relationships
        query = db.query(
            ConceptRelationship, Concept
        ).join(
            Concept, ConceptRelationship.to_concept_id == Concept.id
        ).filter(
            ConceptRelationship.from_concept_id == concept.id,
            ConceptRelationship.weight >= min_weight
        )

        if not include_cold:
            query = query.filter(Concept.tier == MemoryTier.hot)
        
        if rel_type:
            query = query.filter(ConceptRelationship.rel_type == rel_type)
        
        outgoing = query.all()
        
        # Get incoming relationships
        query = db.query(
            ConceptRelationship, Concept
        ).join(
            Concept, ConceptRelationship.from_concept_id == Concept.id
        ).filter(
            ConceptRelationship.to_concept_id == concept.id,
            ConceptRelationship.weight >= min_weight
        )

        if not include_cold:
            query = query.filter(Concept.tier == MemoryTier.hot)
        
        if rel_type:
            query = query.filter(ConceptRelationship.rel_type == rel_type)
        
        incoming = query.all()
        
        return {
            "status": "found",
            "concept": concept.name,
            "outgoing": [
                {
                    "to": c.name,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description
                }
                for rel, c in outgoing
            ],
            "incoming": [
                {
                    "from": c.name,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description
                }
                for rel, c in incoming
            ]
        }
    finally:
        db.close()

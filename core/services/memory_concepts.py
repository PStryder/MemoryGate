"""
Concept and relationship services.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import func, or_

from core.context import RequestContext, resolve_tenant_id
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
from core.services.memory_quota import (
    enforce_quota_or_raise,
    estimate_concept_bytes,
    estimate_embedding_bytes,
    record_storage_usage,
)
from core.services.ai_identity import ensure_ai_context
from core.services.ai_memory_policy import apply_ai_memory_filter, get_ai_memory_mode


def _context_values(context: Optional[RequestContext]) -> tuple[Optional[str], Optional[str]]:
    tenant_id = resolve_tenant_id(context)
    user_pk = None
    if context and context.auth and context.auth.user_id is not None:
        user_pk = str(context.auth.user_id)
    return tenant_id, user_pk


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _key_matches(column, key: str):
    return or_(column == key, func.trim(column) == key)


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
    agent_uuid: Optional[str] = None,
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
    effective_agent_uuid = agent_uuid or (context.agent_uuid if context else None)
    _validate_optional_text(effective_agent_uuid, "agent_uuid", MAX_SHORT_TEXT_LENGTH)

    name_clean = name.strip()
    db = DB.SessionLocal()
    try:
        tenant_id, user_pk = _context_values(context)
        bytes_to_write = estimate_concept_bytes(
            name=name_clean,
            concept_type=concept_type,
            description=description,
            metadata=metadata,
        )
        bytes_to_write += estimate_embedding_bytes()
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        # Get AI instance if provided
        ai_instance_id = None
        ai_context = None
        if effective_agent_uuid or (ai_name and ai_platform):
            ai_context = ensure_ai_context(
                db,
                tenant_id,
                agent_uuid=effective_agent_uuid,
                ai_name=ai_name,
                ai_platform=ai_platform,
                user_id=user_pk,
            )
            ai_instance_id = ai_context["ai_instance_id"]

        # Check if concept already exists (case-insensitive)
        name_key = _normalize_key(name_clean)
        existing_query = db.query(Concept).filter(_key_matches(Concept.name_key, name_key))
        if tenant_id:
            existing_query = existing_query.filter(Concept.tenant_id == tenant_id)
        existing = existing_query.first()
        if existing:
            return {
                "status": "error",
                "message": f"Concept '{name_clean}' already exists with ID {existing.id}",
                "existing_id": existing.id,
            }

        # Create concept
        payload = {
            "name": name_clean,
            "name_key": name_key,
            "type": concept_type,
            "description": description,
            "domain": domain,
            "status": status,
            "metadata_": metadata or {},
            "ai_instance_id": ai_instance_id,
            "created_by_user_pk": user_pk,
        }
        if tenant_id:
            payload["tenant_id"] = tenant_id
        concept = Concept(**payload)
        db.add(concept)
        db.commit()
        db.refresh(concept)

        # Generate and store embedding from description (if enabled)
        _store_embedding(db, "concept", concept.id, description, tenant_id=tenant_id)
        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)

        result = {
            "ack": "SUCCESS",
            "status": "stored",
            "message": f"Concept '{name_clean}' stored successfully (id={concept.id})",
            "id": concept.id,
            "ref": f"concept:{concept.id}",
            "name": name_clean,
            "type": concept_type,
            "description": description,
        }
        if ai_context:
            result.update(
                {
                    key: ai_context[key]
                    for key in (
                        "agent_uuid",
                        "canonical_name",
                        "canonical_platform",
                        "agent_id_instructions",
                        "agent_id_nag",
                        "needs_user_confirmation",
                        "agent_identity_status",
                    )
                    if ai_context.get(key) is not None
                }
            )
        return result
    finally:
        db.close()


def _resolve_concept_by_name(
    db,
    name: str,
    include_cold: bool,
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
    ai_mode: str,
) -> Optional[Concept]:
    """Resolve a concept by name or alias, with optional cold-tier lookup."""
    name_key = _normalize_key(name)
    concept_query = db.query(Concept).filter(_key_matches(Concept.name_key, name_key))
    if tenant_id:
        concept_query = concept_query.filter(Concept.tenant_id == tenant_id)
    concept_query = apply_ai_memory_filter(
        concept_query,
        model=Concept,
        entity_type="concept",
        tenant_id=tenant_id,
        ai_instance_id=ai_instance_id,
        mode=ai_mode,
    )
    if not include_cold:
        concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
    concept = concept_query.first()

    if not concept:
        alias_query = db.query(ConceptAlias).filter(_key_matches(ConceptAlias.alias_key, name_key))
        if tenant_id:
            alias_query = alias_query.filter(ConceptAlias.tenant_id == tenant_id)
        alias = alias_query.first()
        if alias:
            concept_query = db.query(Concept).filter(Concept.id == alias.concept_id)
            if tenant_id:
                concept_query = concept_query.filter(Concept.tenant_id == tenant_id)
            concept_query = apply_ai_memory_filter(
                concept_query,
                model=Concept,
                entity_type="concept",
                tenant_id=tenant_id,
                ai_instance_id=ai_instance_id,
                mode=ai_mode,
            )
            if not include_cold:
                concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
            concept = concept_query.first()

    return concept


@service_tool
def memory_get_concept(
    name: str,
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
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
        tenant_id, _ = _context_values(context)
        ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
        concept = _resolve_concept_by_name(db, name, include_cold, tenant_id, ai_instance_id, ai_mode)
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
            "access_count": concept.access_count,
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
        # Find the concept
        concept_key = _normalize_key(concept_name)
        tenant_id, _ = _context_values(context)
        concept_query = db.query(Concept).filter(_key_matches(Concept.name_key, concept_key))
        if tenant_id:
            concept_query = concept_query.filter(Concept.tenant_id == tenant_id)
        concept = concept_query.first()
        if not concept:
            return {"status": "error", "message": f"Concept '{concept_name}' not found"}

        # Check if alias already exists
        alias_clean = alias.strip()
        alias_key = _normalize_key(alias_clean)
        alias_query = db.query(ConceptAlias).filter(_key_matches(ConceptAlias.alias_key, alias_key))
        if tenant_id:
            alias_query = alias_query.filter(ConceptAlias.tenant_id == tenant_id)
        existing_alias = alias_query.first()
        if existing_alias:
            return {"status": "error", "message": f"Alias '{alias}' already exists"}

        # Check if alias conflicts with existing concept name
        existing_query = db.query(Concept).filter(_key_matches(Concept.name_key, alias_key))
        if tenant_id:
            existing_query = existing_query.filter(Concept.tenant_id == tenant_id)
        existing_concept = existing_query.first()
        if existing_concept:
            return {"status": "error", "message": f"Alias '{alias}' conflicts with existing concept"}

        # Create alias
        alias_tenant_id = getattr(concept, "tenant_id", None) or tenant_id
        payload = {
            "concept_id": concept.id,
            "alias": alias_clean,
            "alias_key": alias_key,
        }
        if alias_tenant_id:
            payload["tenant_id"] = alias_tenant_id
        new_alias = ConceptAlias(**payload)
        db.add(new_alias)
        db.commit()

        return {
            "ack": "SUCCESS",
            "status": "created",
            "message": f"Alias '{alias}' added to concept '{concept.name}'",
            "concept_id": concept.id,
            "concept_name": concept.name,
            "alias": alias,
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
        # Valid relationship types
        valid_types = ["enables", "version_of", "part_of", "related_to", "implements", "demonstrates"]
        if rel_type not in valid_types:
            return {"status": "error", "message": f"Invalid rel_type. Must be one of: {', '.join(valid_types)}"}

        # Validate weight
        if not 0.0 <= weight <= 1.0:
            return {"status": "error", "message": "Weight must be between 0.0 and 1.0"}

        # Find both concepts (case-insensitive, alias-aware)
        from_key = _normalize_key(from_concept)
        to_key = _normalize_key(to_concept)

        tenant_id, _ = _context_values(context)
        from_query = db.query(Concept).filter(_key_matches(Concept.name_key, from_key))
        to_query = db.query(Concept).filter(_key_matches(Concept.name_key, to_key))
        if tenant_id:
            from_query = from_query.filter(Concept.tenant_id == tenant_id)
            to_query = to_query.filter(Concept.tenant_id == tenant_id)
        from_c = from_query.first()
        to_c = to_query.first()

        if not from_c:
            return {"status": "error", "message": f"Source concept '{from_concept}' not found"}
        if not to_c:
            return {"status": "error", "message": f"Target concept '{to_concept}' not found"}

        # Check if relationship already exists
        rel_tenant_id = getattr(from_c, "tenant_id", None) or tenant_id
        existing_query = db.query(ConceptRelationship).filter(
            ConceptRelationship.from_concept_id == from_c.id,
            ConceptRelationship.to_concept_id == to_c.id,
            ConceptRelationship.rel_type == rel_type,
        )
        if rel_tenant_id:
            existing_query = existing_query.filter(ConceptRelationship.tenant_id == rel_tenant_id)
        existing = existing_query.first()

        if existing:
            # Update existing relationship
            existing.weight = weight
            if description:
                existing.description = description
            db.commit()
            return {
                "ack": "SUCCESS",
                "status": "updated",
                "message": f"Relationship '{from_c.name}' --{rel_type}--> '{to_c.name}' updated",
                "from": from_c.name,
                "to": to_c.name,
                "rel_type": rel_type,
                "weight": weight,
            }

        # Create new relationship
        payload = {
            "from_concept_id": from_c.id,
            "to_concept_id": to_c.id,
            "rel_type": rel_type,
            "weight": weight,
            "description": description,
        }
        if rel_tenant_id:
            payload["tenant_id"] = rel_tenant_id
        rel = ConceptRelationship(**payload)
        db.add(rel)
        db.commit()

        return {
            "ack": "SUCCESS",
            "status": "created",
            "message": f"Relationship '{from_c.name}' --{rel_type}--> '{to_c.name}' created",
            "from": from_c.name,
            "to": to_c.name,
            "rel_type": rel_type,
            "weight": weight,
        }
    finally:
        db.close()


@service_tool
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0,
    include_cold: bool = False,
    ai_instance_id: Optional[int] = None,
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
        # Find the concept (alias-aware)
        tenant_id, _ = _context_values(context)
        ai_mode = get_ai_memory_mode(db, tenant_id, ai_instance_id)
        concept = _resolve_concept_by_name(db, concept_name, include_cold, tenant_id, ai_instance_id, ai_mode)
        if not concept:
            return {"status": "not_found", "concept": concept_name}

        # Get outgoing relationships
        query = db.query(
            ConceptRelationship, Concept
        ).join(
            Concept, ConceptRelationship.to_concept_id == Concept.id
        ).filter(
            ConceptRelationship.from_concept_id == concept.id,
            ConceptRelationship.weight >= min_weight,
        )
        relation_tenant_id = getattr(concept, "tenant_id", None) or tenant_id
        if relation_tenant_id:
            query = query.filter(ConceptRelationship.tenant_id == relation_tenant_id)
            query = query.filter(Concept.tenant_id == relation_tenant_id)
        query = apply_ai_memory_filter(
            query,
            model=Concept,
            entity_type="concept",
            tenant_id=relation_tenant_id,
            ai_instance_id=ai_instance_id,
            mode=ai_mode,
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
            ConceptRelationship.weight >= min_weight,
        )
        if relation_tenant_id:
            query = query.filter(ConceptRelationship.tenant_id == relation_tenant_id)
            query = query.filter(Concept.tenant_id == relation_tenant_id)
        query = apply_ai_memory_filter(
            query,
            model=Concept,
            entity_type="concept",
            tenant_id=relation_tenant_id,
            ai_instance_id=ai_instance_id,
            mode=ai_mode,
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
                    "description": rel.description,
                }
                for rel, c in outgoing
            ],
            "incoming": [
                {
                    "from": c.name,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description,
                }
                for rel, c in incoming
            ],
        }
    finally:
        db.close()

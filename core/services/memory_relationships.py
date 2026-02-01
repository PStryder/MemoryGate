"""
Memory relationship services for generic edges between memories.

Supports:
- Creating relationships between any memory types
- Supersession pattern (immutable correction tracking)
- Relationship residue (compression friction tracking)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import (
    MemoryRelationship,
    RelationshipResidue,
    MemorySupersede,
    MemoryTier,
    TombstoneAction,
    MEMORY_MODELS,
    MEMORY_RELATIONSHIP_MODELS,
)
from core.services.memory_shared import (
    _validate_limit,
    _validate_metadata,
    _validate_optional_text,
    _validate_required_text,
    _write_tombstone,
    MAX_RESULT_LIMIT,
    MAX_SHORT_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    service_tool,
    logger,
)


class MemoryRelationshipNotFoundError(Exception):
    pass


SUPPORTED_TYPES: tuple[str, ...] = tuple(MEMORY_RELATIONSHIP_MODELS.keys())
ARCHIVABLE_SUPERSEDES_TYPES: set[str] = {"observation", "pattern", "concept"}

RESIDUE_EVENT_TYPES = {
    "created",
    "revised",
    "divergent",
    "confidence_update",
    "merged",
}


def parse_ref(ref: str, *, field: str = "ref") -> tuple[str, str]:
    """Parse a memory reference in 'type:id' format."""
    if not isinstance(ref, str):
        raise ValidationIssue(
            f"{field} must be a string",
            field=field,
            error_type="invalid_type",
        )
    value = ref.strip()
    if ":" not in value:
        raise ValidationIssue(
            f"{field} must be in the form 'type:id'",
            field=field,
            error_type="invalid_type",
        )
    mem_type, mem_id = value.split(":", 1)
    mem_type = mem_type.strip().lower()
    mem_id = mem_id.strip()
    if not mem_type or not mem_id:
        raise ValidationIssue(
            f"{field} must include both type and id",
            field=field,
            error_type="invalid_type",
        )
    return mem_type, mem_id


def format_ref(mem_type: str, mem_id: str) -> str:
    return f"{mem_type.strip().lower()}:{mem_id}"


def validate_type(mem_type: str, supported_types: Optional[tuple] = None) -> bool:
    if not isinstance(mem_type, str):
        return False
    allowed = set(supported_types or SUPPORTED_TYPES)
    return mem_type.strip().lower() in allowed


def _normalize_rel_type(rel_type: str) -> str:
    if not isinstance(rel_type, str):
        raise ValidationIssue(
            "rel_type must be a string",
            field="rel_type",
            error_type="invalid_type",
        )
    value = rel_type.strip()
    if not value:
        raise ValidationIssue(
            "rel_type is required",
            field="rel_type",
            error_type="required",
        )
    return value


def _normalize_weight(weight: Optional[float]) -> Optional[float]:
    if weight is None:
        return None
    if not isinstance(weight, (int, float)):
        raise ValidationIssue(
            "weight must be a number",
            field="weight",
            error_type="invalid_type",
        )
    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValidationIssue(
            "weight must be between 0 and 1",
            field="weight",
            error_type="invalid_value",
        )
    return value


def _normalize_metadata(metadata: Optional[dict]) -> Optional[dict]:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValidationIssue(
            "metadata must be an object",
            field="metadata",
            error_type="invalid_type",
        )
    return dict(metadata)


def _primary_key_column(model, *, field: str):
    pk_columns = list(model.__table__.primary_key.columns)
    if len(pk_columns) != 1:
        raise ValidationIssue(
            f"{field} must reference a single-column primary key",
            field=field,
            error_type="invalid_type",
        )
    return pk_columns[0]


def _coerce_model_id(pk_column, raw_id: str, *, field: str) -> tuple[str, object]:
    if not raw_id:
        raise ValidationIssue(
            f"{field} id is required",
            field=field,
            error_type="required",
        )
    python_type = None
    try:
        python_type = pk_column.type.python_type
    except (NotImplementedError, AttributeError):
        python_type = None
    if python_type is int:
        try:
            value = int(raw_id)
        except ValueError as exc:
            raise ValidationIssue(
                f"{field} id must be numeric",
                field=field,
                error_type="invalid_id",
            ) from exc
        return str(value), value
    if python_type is uuid.UUID:
        try:
            value = uuid.UUID(raw_id)
        except ValueError as exc:
            raise ValidationIssue(
                f"{field} id must be a valid UUID",
                field=field,
                error_type="invalid_id",
            ) from exc
        return str(value), value
    return raw_id, raw_id


def _resolve_ref(
    db,
    tenant_id: str,
    raw: str,
    *,
    field: str,
) -> tuple[str, str, object]:
    """Resolve a memory reference and return (type, canonical_id, record)."""
    mem_type, mem_id = parse_ref(raw, field=field)
    if not validate_type(mem_type):
        raise ValidationIssue(
            f"Unknown memory type: {mem_type}",
            field=field,
            error_type="invalid_type",
        )
    model = MEMORY_RELATIONSHIP_MODELS.get(mem_type)
    if not model:
        raise ValidationIssue(
            f"Unsupported memory type: {mem_type}",
            field=field,
            error_type="invalid_type",
        )
    pk_column = _primary_key_column(model, field=field)
    canonical_id, mem_id_value = _coerce_model_id(pk_column, mem_id, field=field)
    record = (
        db.query(model)
        .filter(pk_column == mem_id_value)
        .filter(model.tenant_id == tenant_id)
        .first()
    )
    if not record:
        raise MemoryRelationshipNotFoundError("Referenced memory not found")
    return mem_type, canonical_id, record


def serialize_memory_relationship(row: MemoryRelationship) -> dict:
    return {
        "id": str(row.id),
        "from_ref": format_ref(row.from_type, row.from_id),
        "to_ref": format_ref(row.to_type, row.to_id),
        "rel_type": row.rel_type,
        "weight": row.weight,
        "description": row.description,
        "metadata": row.metadata_,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def serialize_relationship_residue(row: RelationshipResidue) -> dict:
    return {
        "id": str(row.id),
        "edge_id": str(row.edge_id),
        "event_type": row.event_type,
        "actor": row.actor,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "encoded_rel_type": row.encoded_rel_type,
        "encoded_weight": row.encoded_weight,
        "alternatives_considered": row.alternatives_considered,
        "alternatives_ruled_out": row.alternatives_ruled_out,
        "friction_metrics": row.friction_metrics,
        "compression_texture": row.compression_texture,
    }


def _is_cold_tier(record) -> bool:
    tier_value = getattr(record, "tier", None)
    if tier_value is None:
        return False
    if isinstance(tier_value, MemoryTier):
        return tier_value == MemoryTier.cold
    return str(tier_value).lower() == "cold"


def _archive_node(
    db,
    *,
    tenant_id: str,
    node_type: str,
    node_id: str,
    record,
    reason: Optional[str] = None,
    actor_label: Optional[str] = None,
) -> bool:
    """Archive a node when it's superseded."""
    if node_type not in ARCHIVABLE_SUPERSEDES_TYPES:
        raise ValidationIssue(
            f"supersedes is not supported for type: {node_type}",
            field="rel_type",
            error_type="invalid_type",
        )

    if _is_cold_tier(record):
        return True

    archive_reason = (reason or "").strip() or "superseded"
    actor_name = actor_label or "system"
    record.tier = MemoryTier.cold
    record.archived_at = datetime.utcnow()
    record.archived_reason = archive_reason
    record.archived_by = actor_name
    record.purge_eligible = False

    memory_id = format_ref(node_type, node_id)
    _write_tombstone(
        db,
        memory_id,
        TombstoneAction.archived,
        from_tier=MemoryTier.hot,
        to_tier=MemoryTier.cold,
        reason=archive_reason,
        actor=actor_name,
        tenant_id=tenant_id,
    )
    return True


def _create_residue(
    db,
    *,
    tenant_id: str,
    edge: MemoryRelationship,
    event_type: str,
    actor: Optional[str],
    alternatives_considered: Optional[list],
    alternatives_ruled_out: Optional[list],
    friction_metrics: Optional[dict],
    compression_texture: Optional[str],
    encoded_rel_type: Optional[str],
    encoded_weight: Optional[float],
) -> RelationshipResidue:
    """Create a relationship residue record."""
    encoded_rel_type_value = encoded_rel_type or edge.rel_type
    encoded_weight_value = encoded_weight if encoded_weight is not None else edge.weight

    residue = RelationshipResidue(
        tenant_id=tenant_id,
        edge_id=edge.id,
        event_type=event_type,
        actor=actor,
        created_at=datetime.utcnow(),
        encoded_rel_type=encoded_rel_type_value,
        encoded_weight=encoded_weight_value,
        alternatives_considered=alternatives_considered,
        alternatives_ruled_out=alternatives_ruled_out,
        friction_metrics=friction_metrics,
        compression_texture=compression_texture,
    )
    db.add(residue)
    db.flush()
    return residue


@service_tool
def memory_add_relationship(
    from_ref: str,
    to_ref: str,
    rel_type: str,
    weight: Optional[float] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Add a relationship between two memories."""
    _validate_required_text(from_ref, "from_ref", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(to_ref, "to_ref", MAX_SHORT_TEXT_LENGTH)

    rel_type_value = _normalize_rel_type(rel_type)
    weight_value = _normalize_weight(weight)
    metadata_value = _normalize_metadata(metadata)

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        from_type, from_id, from_record = _resolve_ref(db, tenant_id, from_ref, field="from_ref")
        to_type, to_id, _ = _resolve_ref(db, tenant_id, to_ref, field="to_ref")

        if from_type == to_type and from_id == to_id:
            raise ValidationIssue(
                "from_ref and to_ref must be different",
                field="to_ref",
                error_type="invalid_value",
            )

        # Check for existing relationship
        existing = (
            db.query(MemoryRelationship)
            .filter(MemoryRelationship.tenant_id == tenant_id)
            .filter(MemoryRelationship.from_type == from_type)
            .filter(MemoryRelationship.from_id == from_id)
            .filter(MemoryRelationship.to_type == to_type)
            .filter(MemoryRelationship.to_id == to_id)
            .filter(MemoryRelationship.rel_type == rel_type_value)
            .first()
        )

        created = False
        if existing:
            if weight_value is not None:
                existing.weight = weight_value
            if description is not None:
                existing.description = description
            if metadata_value is not None:
                existing.metadata_ = metadata_value
            relationship = existing
        else:
            relationship = MemoryRelationship(
                tenant_id=tenant_id,
                from_type=from_type,
                from_id=from_id,
                to_type=to_type,
                to_id=to_id,
                rel_type=rel_type_value,
                weight=weight_value,
                description=description,
                metadata_=metadata_value,
                created_at=datetime.utcnow(),
                created_by_type=context.auth.actor if context and context.auth else None,
                created_by_id=str(context.auth.user_id)
                if context and context.auth and context.auth.user_id is not None
                else None,
            )
            db.add(relationship)
            db.flush()
            created = True

        archived_old = False
        if rel_type_value == "supersedes":
            reason = metadata_value.get("reason") if metadata_value else None
            actor_label = context.auth.actor if context and context.auth else None
            archived_old = _archive_node(
                db,
                tenant_id=tenant_id,
                node_type=from_type,
                node_id=from_id,
                record=from_record,
                reason=reason,
                actor_label=actor_label,
            )

        # Handle residue if provided in metadata
        if metadata_value and "residue" in metadata_value:
            residue_payload = metadata_value.get("residue")
            if residue_payload and isinstance(residue_payload, dict):
                if rel_type_value != "friction_residue":
                    event_type = residue_payload.get("event_type")
                    if not event_type:
                        event_type = "created" if created else "revised"
                    _create_residue(
                        db,
                        tenant_id=tenant_id,
                        edge=relationship,
                        event_type=event_type,
                        actor=residue_payload.get("actor") or (context.auth.actor if context and context.auth else None),
                        alternatives_considered=residue_payload.get("alternatives_considered"),
                        alternatives_ruled_out=residue_payload.get("alternatives_ruled_out"),
                        friction_metrics=residue_payload.get("friction_metrics"),
                        compression_texture=residue_payload.get("compression_texture"),
                        encoded_rel_type=residue_payload.get("encoded_rel_type"),
                        encoded_weight=residue_payload.get("encoded_weight"),
                    )

        db.commit()
        db.refresh(relationship)

        return {
            "status": "ok",
            "created": created,
            "archived_superseded": archived_old,
            "relationship": serialize_memory_relationship(relationship),
        }
    except MemoryRelationshipNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    except IntegrityError as exc:
        db.rollback()
        return {"status": "error", "message": "Relationship already exists"}
    finally:
        db.close()


@service_tool
def memory_list_relationships(
    ref: str,
    direction: str = "both",
    rel_type: Optional[str] = None,
    min_weight: Optional[float] = None,
    limit: int = 100,
    context: Optional[RequestContext] = None,
) -> dict:
    """List relationships for a memory."""
    _validate_required_text(ref, "ref", MAX_SHORT_TEXT_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    direction_value = (direction or "both").strip().lower()
    if direction_value not in {"both", "out", "in"}:
        raise ValidationIssue(
            "direction must be one of: both, out, in",
            field="direction",
            error_type="invalid_value",
        )
    if min_weight is not None:
        min_weight = _normalize_weight(min_weight)

    rel_type_value = rel_type.strip() if isinstance(rel_type, str) else None
    if rel_type_value == "":
        rel_type_value = None

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        mem_type, mem_id, _ = _resolve_ref(db, tenant_id, ref, field="ref")

        query = db.query(MemoryRelationship).filter(MemoryRelationship.tenant_id == tenant_id)
        if direction_value == "out":
            query = query.filter(MemoryRelationship.from_type == mem_type)
            query = query.filter(MemoryRelationship.from_id == mem_id)
        elif direction_value == "in":
            query = query.filter(MemoryRelationship.to_type == mem_type)
            query = query.filter(MemoryRelationship.to_id == mem_id)
        else:
            query = query.filter(
                or_(
                    (MemoryRelationship.from_type == mem_type)
                    & (MemoryRelationship.from_id == mem_id),
                    (MemoryRelationship.to_type == mem_type)
                    & (MemoryRelationship.to_id == mem_id),
                )
            )

        if rel_type_value:
            query = query.filter(MemoryRelationship.rel_type == rel_type_value)
        if min_weight is not None:
            query = query.filter(MemoryRelationship.weight >= min_weight)

        rows = (
            query.order_by(MemoryRelationship.created_at.asc())
            .limit(limit)
            .all()
        )

        return {
            "status": "ok",
            "count": len(rows),
            "relationships": [serialize_memory_relationship(row) for row in rows],
        }
    except MemoryRelationshipNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@service_tool
def memory_related(
    ref: str,
    rel_type: Optional[str] = None,
    min_weight: Optional[float] = None,
    limit: int = 50,
    context: Optional[RequestContext] = None,
) -> dict:
    """Find related memories via relationship traversal."""
    _validate_required_text(ref, "ref", MAX_SHORT_TEXT_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        mem_type, mem_id, _ = _resolve_ref(db, tenant_id, ref, field="ref")

        query = db.query(MemoryRelationship).filter(MemoryRelationship.tenant_id == tenant_id)
        query = query.filter(
            or_(
                (MemoryRelationship.from_type == mem_type)
                & (MemoryRelationship.from_id == mem_id),
                (MemoryRelationship.to_type == mem_type)
                & (MemoryRelationship.to_id == mem_id),
            )
        )

        if rel_type:
            query = query.filter(MemoryRelationship.rel_type == rel_type)
        if min_weight is not None:
            min_weight = _normalize_weight(min_weight)
            query = query.filter(MemoryRelationship.weight >= min_weight)

        rows = query.order_by(MemoryRelationship.created_at.asc()).limit(limit).all()

        results = []
        for row in rows:
            if row.from_type == mem_type and row.from_id == mem_id:
                neighbor_type = row.to_type
                neighbor_id = row.to_id
                direction = "out"
            else:
                neighbor_type = row.from_type
                neighbor_id = row.from_id
                direction = "in"
            results.append({
                "neighbor_ref": format_ref(neighbor_type, neighbor_id),
                "edge": serialize_memory_relationship(row),
                "direction": direction,
            })

        return {
            "status": "ok",
            "count": len(results),
            "related": results,
        }
    except MemoryRelationshipNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@service_tool
def memory_get_supersession(
    ref: str,
    context: Optional[RequestContext] = None,
) -> dict:
    """Check if a memory has been superseded."""
    _validate_required_text(ref, "ref", MAX_SHORT_TEXT_LENGTH)
    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        mem_type, mem_id, _ = _resolve_ref(db, tenant_id, ref, field="ref")

        row = (
            db.query(MemoryRelationship)
            .filter(MemoryRelationship.tenant_id == tenant_id)
            .filter(MemoryRelationship.from_type == mem_type)
            .filter(MemoryRelationship.from_id == mem_id)
            .filter(MemoryRelationship.rel_type == "supersedes")
            .order_by(MemoryRelationship.created_at.desc())
            .first()
        )

        if not row:
            return {"status": "ok", "superseded": False, "ref": ref}

        metadata = row.metadata_ or {}
        return {
            "status": "ok",
            "superseded": True,
            "ref": ref,
            "to_ref": format_ref(row.to_type, row.to_id),
            "reason": metadata.get("reason"),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    except MemoryRelationshipNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@service_tool
def relationship_add_residue(
    edge_id: str,
    event_type: str,
    actor: Optional[str] = None,
    alternatives_considered: Optional[list] = None,
    alternatives_ruled_out: Optional[list] = None,
    friction_metrics: Optional[dict] = None,
    compression_texture: Optional[str] = None,
    encoded_rel_type: Optional[str] = None,
    encoded_weight: Optional[float] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Add residue to an existing relationship edge."""
    _validate_required_text(edge_id, "edge_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(event_type, "event_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(actor, "actor", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(compression_texture, "compression_texture", MAX_TEXT_LENGTH)
    _validate_optional_text(encoded_rel_type, "encoded_rel_type", MAX_SHORT_TEXT_LENGTH)

    if event_type not in RESIDUE_EVENT_TYPES:
        raise ValidationIssue(
            f"event_type must be one of: {', '.join(RESIDUE_EVENT_TYPES)}",
            field="event_type",
            error_type="invalid_value",
        )

    if encoded_weight is not None:
        encoded_weight = _normalize_weight(encoded_weight)

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        edge = (
            db.query(MemoryRelationship)
            .filter(MemoryRelationship.id == edge_id)
            .filter(MemoryRelationship.tenant_id == tenant_id)
            .first()
        )
        if not edge:
            raise ValidationIssue(
                "Relationship edge not found",
                field="edge_id",
                error_type="not_found",
            )

        residue = _create_residue(
            db,
            tenant_id=tenant_id,
            edge=edge,
            event_type=event_type,
            actor=actor or (context.auth.actor if context and context.auth else None),
            alternatives_considered=alternatives_considered,
            alternatives_ruled_out=alternatives_ruled_out,
            friction_metrics=friction_metrics,
            compression_texture=compression_texture,
            encoded_rel_type=encoded_rel_type,
            encoded_weight=encoded_weight,
        )
        db.commit()
        db.refresh(residue)

        return {
            "status": "ok",
            "residue": serialize_relationship_residue(residue),
        }
    finally:
        db.close()


@service_tool
def relationship_list_residue(
    edge_id: str,
    limit: int = 20,
    actor: Optional[str] = None,
    event_type: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """List residue history for a relationship edge."""
    _validate_required_text(edge_id, "edge_id", MAX_SHORT_TEXT_LENGTH)
    _validate_limit(limit, "limit", 200)
    _validate_optional_text(actor, "actor", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(event_type, "event_type", MAX_SHORT_TEXT_LENGTH)

    if event_type and event_type not in RESIDUE_EVENT_TYPES:
        raise ValidationIssue(
            f"event_type must be one of: {', '.join(RESIDUE_EVENT_TYPES)}",
            field="event_type",
            error_type="invalid_value",
        )

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        edge = (
            db.query(MemoryRelationship)
            .filter(MemoryRelationship.id == edge_id)
            .filter(MemoryRelationship.tenant_id == tenant_id)
            .first()
        )
        if not edge:
            raise ValidationIssue(
                "Relationship edge not found",
                field="edge_id",
                error_type="not_found",
            )

        query = (
            db.query(RelationshipResidue)
            .filter(RelationshipResidue.edge_id == edge_id)
            .filter(RelationshipResidue.tenant_id == tenant_id)
        )
        if actor:
            query = query.filter(RelationshipResidue.actor == actor)
        if event_type:
            query = query.filter(RelationshipResidue.event_type == event_type)

        rows = (
            query.order_by(RelationshipResidue.created_at.desc())
            .limit(limit)
            .all()
        )

        return {
            "status": "ok",
            "edge_id": str(edge.id),
            "count": len(rows),
            "residue": [serialize_relationship_residue(row) for row in rows],
        }
    finally:
        db.close()

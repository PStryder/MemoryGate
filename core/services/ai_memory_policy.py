"""
AI memory topology helpers.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import and_, exists, or_

from core.models import AIEntityShare, AIMemoryPolicy

AI_MEMORY_MODES = {"shared_all", "separate", "selective"}
DEFAULT_AI_MEMORY_MODE = "shared_all"


def get_ai_memory_mode(db, tenant_id: Optional[str], ai_instance_id: Optional[int]) -> str:
    """Resolve the memory topology mode for an AI instance."""
    if not tenant_id or not ai_instance_id:
        return DEFAULT_AI_MEMORY_MODE
    row = (
        db.query(AIMemoryPolicy)
        .filter(AIMemoryPolicy.tenant_id == tenant_id)
        .filter(AIMemoryPolicy.ai_instance_id == ai_instance_id)
        .first()
    )
    if not row:
        return DEFAULT_AI_MEMORY_MODE
    return row.mode if row.mode in AI_MEMORY_MODES else DEFAULT_AI_MEMORY_MODE


def _share_exists_clause(
    *,
    tenant_id: str,
    entity_type: str,
    entity_id_column,
    ai_instance_id: int,
):
    return exists().where(
        and_(
            AIEntityShare.tenant_id == tenant_id,
            AIEntityShare.entity_type == entity_type,
            AIEntityShare.entity_id == entity_id_column,
            AIEntityShare.shared_with_ai_instance_id == ai_instance_id,
        )
    )


def apply_ai_memory_filter(
    query,
    *,
    model,
    entity_type: str,
    tenant_id: Optional[str],
    ai_instance_id: Optional[int],
    mode: str,
):
    """Apply AI memory visibility rules to a SQLAlchemy query."""
    if mode == DEFAULT_AI_MEMORY_MODE or not tenant_id or not ai_instance_id:
        return query

    share_exists = _share_exists_clause(
        tenant_id=tenant_id,
        entity_type=entity_type,
        entity_id_column=model.id,
        ai_instance_id=ai_instance_id,
    )

    if hasattr(model, "ai_instance_id"):
        if mode == "separate":
            return query.filter(model.ai_instance_id == ai_instance_id)
        return query.filter(or_(model.ai_instance_id == ai_instance_id, share_exists))

    # Models without ai_instance_id (e.g., documents) fall back to explicit shares.
    return query.filter(share_exists)


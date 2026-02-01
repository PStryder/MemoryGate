"""
Agent anchor services.

Provides functionality for managing agent anchor chains:
- Set anchor chain for an agent
- Get anchor chain for an agent
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import AnchorPointer, MemoryChain
from core.services.memory_shared import (
    _validate_required_text,
    MAX_SHORT_TEXT_LENGTH,
    service_tool,
    logger,
)


ANCHOR_SCOPE_TYPE_AGENT = "agent"
ANCHOR_KIND_AGENT_PROFILE = "agent_profile"


def _anchor_scope_key(ai_platform: str, ai_name: str) -> str:
    """Generate scope key for an agent."""
    return f"{ai_platform.strip()}::{ai_name.strip()}"


def _require_chain(db, tenant_id: str, chain_id: str) -> MemoryChain:
    """Get chain or raise ValidationIssue."""
    chain = (
        db.query(MemoryChain)
        .filter(MemoryChain.tenant_id == tenant_id)
        .filter(MemoryChain.id == chain_id)
        .first()
    )
    if not chain:
        raise ValidationIssue(
            "Anchor chain not found",
            field="anchor_chain_id",
            error_type="not_found",
        )
    return chain


@service_tool
def agent_anchor_set(
    ai_name: str,
    ai_platform: str,
    anchor_chain_id: str,
    anchor_kind: str = ANCHOR_KIND_AGENT_PROFILE,
    context: Optional[RequestContext] = None,
) -> dict:
    """Set the anchor chain for an AI agent."""
    _validate_required_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(anchor_chain_id, "anchor_chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(anchor_kind, "anchor_kind", MAX_SHORT_TEXT_LENGTH)

    if anchor_kind != ANCHOR_KIND_AGENT_PROFILE:
        raise ValidationIssue(
            "Unsupported anchor_kind",
            field="anchor_kind",
            error_type="invalid_value",
        )

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        # Verify chain exists
        _require_chain(db, tenant_id, anchor_chain_id)

        scope_key = _anchor_scope_key(ai_platform, ai_name)
        anchor = (
            db.query(AnchorPointer)
            .filter(AnchorPointer.tenant_id == tenant_id)
            .filter(AnchorPointer.scope_type == ANCHOR_SCOPE_TYPE_AGENT)
            .filter(AnchorPointer.scope_key == scope_key)
            .filter(AnchorPointer.anchor_kind == anchor_kind)
            .first()
        )

        now = datetime.utcnow()
        if anchor:
            anchor.chain_id = anchor_chain_id
            anchor.updated_at = now
        else:
            anchor = AnchorPointer(
                tenant_id=tenant_id,
                scope_type=ANCHOR_SCOPE_TYPE_AGENT,
                scope_key=scope_key,
                anchor_kind=anchor_kind,
                chain_id=anchor_chain_id,
                created_at=now,
                updated_at=now,
            )
            db.add(anchor)

        db.commit()

        return {
            "status": "ok",
            "anchor_kind": anchor_kind,
            "anchor_chain_id": str(anchor.chain_id),
            "scope_key": scope_key,
        }
    finally:
        db.close()


@service_tool
def agent_anchor_get(
    ai_name: str,
    ai_platform: str,
    anchor_kind: str = ANCHOR_KIND_AGENT_PROFILE,
    context: Optional[RequestContext] = None,
) -> dict:
    """Get the anchor chain for an AI agent."""
    _validate_required_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(anchor_kind, "anchor_kind", MAX_SHORT_TEXT_LENGTH)

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        scope_key = _anchor_scope_key(ai_platform, ai_name)
        anchor = (
            db.query(AnchorPointer)
            .filter(AnchorPointer.tenant_id == tenant_id)
            .filter(AnchorPointer.scope_type == ANCHOR_SCOPE_TYPE_AGENT)
            .filter(AnchorPointer.scope_key == scope_key)
            .filter(AnchorPointer.anchor_kind == anchor_kind)
            .first()
        )

        if not anchor:
            return {
                "status": "not_found",
                "scope_key": scope_key,
                "anchor_kind": anchor_kind,
            }

        return {
            "status": "ok",
            "anchor_kind": anchor.anchor_kind,
            "anchor_chain_id": str(anchor.chain_id),
            "scope_key": scope_key,
            "chain_head_seq": anchor.chain_head_seq,
            "created_at": anchor.created_at.isoformat() if anchor.created_at else None,
            "updated_at": anchor.updated_at.isoformat() if anchor.updated_at else None,
        }
    finally:
        db.close()

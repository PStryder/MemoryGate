"""
Audit logging helpers (DB-only, metadata-only).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import and_, or_

import core.config as config
from core.models import AuditEvent

ALLOWED_ACTOR_TYPES = {"user", "org_admin", "system", "integration", "mcp"}
ALLOWED_TARGET_TYPES = {"memory", "summary", "account", "org", "key", "export"}

FORBIDDEN_METADATA_KEYS = {
    "content",
    "observation",
    "description",
    "pattern_text",
    "embedding",
    "summary_text",
    "document_body",
    "raw_text",
}
MAX_METADATA_STRING_LENGTH = 500
MAX_TARGET_ID_LENGTH = 200


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _metadata_key_forbidden(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in FORBIDDEN_METADATA_KEYS:
        return True
    for token in FORBIDDEN_METADATA_KEYS:
        if token in normalized:
            return True
    return False


def _validate_metadata_value(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if _metadata_key_forbidden(key):
                raise ValueError(f"metadata key '{key}' is not allowed")
            next_path = f"{path}.{key}" if path else key
            _validate_metadata_value(item, next_path)
        return
    if isinstance(value, list):
        for item in value:
            _validate_metadata_value(item, path)
        return
    if isinstance(value, str) and len(value) > MAX_METADATA_STRING_LENGTH:
        raise ValueError(f"metadata value too long at '{path or 'value'}'")


def _coerce_target_ids(target_ids: Any) -> list[Any]:
    if not isinstance(target_ids, (list, tuple)):
        raise ValueError("target_ids must be a list")
    coerced: list[Any] = []
    for item in target_ids:
        if isinstance(item, str):
            if len(item) > MAX_TARGET_ID_LENGTH:
                raise ValueError("target_id value too long")
            coerced.append(item)
        elif isinstance(item, int):
            coerced.append(item)
        else:
            raise ValueError("target_ids must contain strings or integers")
    return coerced


def log_event(
    db,
    *,
    event_type: str,
    actor_type: str,
    actor_id: Optional[str] = None,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    target_type: str,
    target_ids: list[Any],
    count_affected: Optional[int] = None,
    reason: Optional[str] = None,
    request_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> AuditEvent:
    """
    Append an audit event (DB-only, metadata-only).
    """
    if not event_type or not isinstance(event_type, str):
        raise ValueError("event_type must be a non-empty string")
    if actor_type not in ALLOWED_ACTOR_TYPES:
        raise ValueError("actor_type must be one of: user|org_admin|system|integration|mcp")
    if target_type not in ALLOWED_TARGET_TYPES:
        raise ValueError("target_type must be one of: memory|summary|account|org|key|export")

    safe_target_ids = _coerce_target_ids(target_ids)

    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        _validate_metadata_value(metadata)

    event = AuditEvent(
        created_at=datetime.utcnow(),
        event_type=event_type,
        tenant_id=org_id or config.DEFAULT_TENANT_ID,
        actor_type=actor_type,
        actor_id=actor_id,
        org_id=org_id,
        user_id=user_id,
        target_type=target_type,
        target_ids=safe_target_ids,
        count_affected=count_affected,
        reason=reason,
        request_id=request_id,
        metadata_=metadata,
    )
    db.add(event)
    return event


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def list_audit_events(
    db,
    *,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None,
) -> dict:
    """
    Query audit events with optional filtering and cursor pagination.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    query = db.query(AuditEvent)
    if org_id:
        query = query.filter(AuditEvent.org_id == org_id)
    if user_id:
        query = query.filter(AuditEvent.user_id == user_id)
    if event_type:
        query = query.filter(AuditEvent.event_type == event_type)

    dt_from = _parse_dt(date_from)
    dt_to = _parse_dt(date_to)
    if dt_from:
        query = query.filter(AuditEvent.created_at >= dt_from)
    if dt_to:
        query = query.filter(AuditEvent.created_at <= dt_to)

    if cursor:
        cursor_event = db.query(AuditEvent).filter(AuditEvent.event_id == cursor).first()
        if cursor_event:
            query = query.filter(
                or_(
                    AuditEvent.created_at < cursor_event.created_at,
                    and_(
                        AuditEvent.created_at == cursor_event.created_at,
                        AuditEvent.event_id < cursor_event.event_id,
                    ),
                )
            )

    rows = (
        query.order_by(AuditEvent.created_at.desc(), AuditEvent.event_id.desc())
        .limit(limit)
        .all()
    )
    next_cursor = str(rows[-1].event_id) if rows else None
    return {
        "status": "ok",
        "count": len(rows),
        "events": [
            {
                "event_id": str(row.event_id),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "event_type": row.event_type,
                "event_version": row.event_version,
                "tenant_id": row.tenant_id,
                "actor_type": row.actor_type,
                "actor_id": row.actor_id,
                "org_id": row.org_id,
                "user_id": row.user_id,
                "target_type": row.target_type,
                "target_ids": row.target_ids,
                "count_affected": row.count_affected,
                "reason": row.reason,
                "request_id": row.request_id,
                "metadata": row.metadata_,
            }
            for row in rows
        ],
        "next_cursor": next_cursor,
    }


__all__ = [
    "AuditEvent",
    "log_event",
    "list_audit_events",
    "ALLOWED_ACTOR_TYPES",
    "ALLOWED_TARGET_TYPES",
]

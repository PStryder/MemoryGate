"""
Store management services.

Provides functionality for managing user store selection:
- List accessible stores
- Get active store
- Set active store
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import UserActiveStore
from core.services.memory_shared import (
    _validate_required_text,
    MAX_SHORT_TEXT_LENGTH,
    service_tool,
    logger,
)


PERSONAL_STORE_ID = "personal"


def _get_user_id(context: Optional[RequestContext]) -> Optional[str]:
    """Extract user_id from context."""
    if context and context.auth and context.auth.user_id:
        return str(context.auth.user_id)
    return None


def _list_accessible_stores(db, tenant_id: str, user_id: Optional[str]) -> list[dict]:
    """
    List stores accessible to the user.

    In single-tenant mode, returns just the personal store.
    In multi-tenant mode, would return personal + shared stores.
    """
    stores = []

    # Personal store is always accessible
    stores.append({
        "store_id": PERSONAL_STORE_ID,
        "name": "Personal",
        "type": "personal",
        "is_default": True,
    })

    # In a full multi-tenant implementation, would query SharedStore table here
    # For now, just return the personal store

    return stores


def _get_active_store(db, tenant_id: str, user_id: Optional[str]) -> Optional[str]:
    """Get the user's currently active store."""
    if not user_id:
        return PERSONAL_STORE_ID

    active = (
        db.query(UserActiveStore)
        .filter(UserActiveStore.tenant_id == tenant_id)
        .filter(UserActiveStore.user_id == user_id)
        .first()
    )

    if active and active.active_store_id:
        return active.active_store_id
    return PERSONAL_STORE_ID


def _set_active_store(db, tenant_id: str, user_id: str, store_id: str) -> None:
    """Set the user's active store."""
    active = (
        db.query(UserActiveStore)
        .filter(UserActiveStore.tenant_id == tenant_id)
        .filter(UserActiveStore.user_id == user_id)
        .first()
    )

    if active:
        active.active_store_id = store_id
        active.updated_at = datetime.utcnow()
    else:
        active = UserActiveStore(
            user_id=user_id,
            tenant_id=tenant_id,
            active_store_id=store_id,
            updated_at=datetime.utcnow(),
        )
        db.add(active)

    db.commit()


def _active_store_payload(stores: list[dict], active_store_id: str, tenant_id: str) -> dict:
    """Build response payload for active store."""
    active_store = next(
        (store for store in stores if store["store_id"] == active_store_id),
        None
    )
    return {
        "active_store_id": active_store_id,
        "store": active_store,
        "context": {
            "tenant_id": tenant_id,
        },
    }


@service_tool
def stores_list_accessible(
    context: Optional[RequestContext] = None,
) -> dict:
    """List all stores accessible to the current user."""
    tenant_id = resolve_tenant_id(context, required=True)
    user_id = _get_user_id(context)

    db = DB.SessionLocal()
    try:
        stores = _list_accessible_stores(db, tenant_id, user_id)
        return {
            "status": "ok",
            "stores": stores,
        }
    finally:
        db.close()


@service_tool
def stores_get_active(
    context: Optional[RequestContext] = None,
) -> dict:
    """Get the currently active store."""
    tenant_id = resolve_tenant_id(context, required=True)
    user_id = _get_user_id(context)

    db = DB.SessionLocal()
    try:
        stores = _list_accessible_stores(db, tenant_id, user_id)
        active_store_id = _get_active_store(db, tenant_id, user_id)
        return _active_store_payload(stores, active_store_id, tenant_id)
    finally:
        db.close()


@service_tool
def stores_set_active(
    store_id: str,
    persist_default: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
    """Set the active store for the current session/user."""
    _validate_required_text(store_id, "store_id", MAX_SHORT_TEXT_LENGTH)

    tenant_id = resolve_tenant_id(context, required=True)
    user_id = _get_user_id(context)

    if not user_id:
        raise ValidationIssue(
            "user_id is required to set active store",
            field="user_id",
            error_type="required",
        )

    db = DB.SessionLocal()
    try:
        # Verify store exists and is accessible
        stores = _list_accessible_stores(db, tenant_id, user_id)
        store_ids = {s["store_id"] for s in stores}

        if store_id not in store_ids:
            raise ValidationIssue(
                f"Store not found or not accessible: {store_id}",
                field="store_id",
                error_type="not_found",
            )

        # Set the active store
        _set_active_store(db, tenant_id, user_id, store_id)

        return _active_store_payload(stores, store_id, tenant_id)
    finally:
        db.close()

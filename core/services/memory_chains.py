"""
Memory chain services for ordered sequences of memories.

Chains provide a way to group memories in an ordered sequence, such as:
- anchor: Agent profile/context chains
- decision: Decision trails
- investigation: Investigation sequences
- workflow: Workflow steps
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from core.context import RequestContext, resolve_tenant_id
from core.db import DB
from core.errors import ValidationIssue
from core.models import (
    MemoryChain,
    MemoryChainItem,
    Observation,
    MEMORY_MODELS,
)
from core.services.memory_shared import (
    _store_embedding,
    _validate_limit,
    _validate_metadata,
    _validate_optional_text,
    _validate_required_text,
    MAX_RESULT_LIMIT,
    MAX_SHORT_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    MAX_TITLE_LENGTH,
    service_tool,
)
from core.services.memory_quota import (
    enforce_quota_or_raise,
    estimate_chain_append_bytes,
    estimate_chain_create_bytes,
    estimate_chain_update_bytes,
    estimate_embedding_bytes,
    estimate_observation_bytes,
    record_storage_usage,
)


class ChainNotFoundError(Exception):
    """Raised when a chain is not found."""
    pass


class ChainConflictError(Exception):
    """Raised when a chain operation conflicts."""
    pass


def _serialize_chain_item(item: MemoryChainItem) -> dict:
    return {
        "id": str(item.id),
        "memory_id": item.memory_id,
        "observation_id": item.observation_id,
        "seq": item.seq,
        "role": item.role,
        "linked_at": item.linked_at.isoformat() if item.linked_at else None,
    }


def _serialize_chain(chain: MemoryChain, items: Optional[list] = None) -> dict:
    payload = {
        "id": str(chain.id),
        "title": chain.title,
        "kind": chain.kind,
        "meta": chain.meta,
        "created_at": chain.created_at.isoformat() if chain.created_at else None,
        "updated_at": chain.updated_at.isoformat() if chain.updated_at else None,
    }
    if items is not None:
        payload["items"] = [_serialize_chain_item(item) for item in items]
    return payload


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValidationIssue(
            "timestamp must be an ISO 8601 string",
            field="timestamp",
            error_type="invalid_type",
        )
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationIssue(
            "timestamp must be ISO 8601 format",
            field="timestamp",
            error_type="invalid_value",
        ) from exc


def _merge_chain_meta(
    base_meta: Optional[dict],
    patch_meta: Optional[dict],
    *,
    store_id: Optional[str] = None,
    scope: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[dict]:
    meta = dict(base_meta or {})
    if patch_meta is not None:
        _validate_metadata(patch_meta, "metadata")
        meta.update(patch_meta)
    if store_id is not None:
        meta["store_id"] = store_id
    if scope is not None:
        meta["scope"] = scope
    if status is not None:
        meta["status"] = status
    return meta or None


def _context_user_pk(context: Optional[RequestContext]) -> Optional[str]:
    if context and context.auth and context.auth.user_id is not None:
        return str(context.auth.user_id)
    return None


def _require_chain(db, tenant_id: str, chain_id: str) -> MemoryChain:
    """Get chain or raise ChainNotFoundError."""
    chain = (
        db.query(MemoryChain)
        .filter(MemoryChain.tenant_id == tenant_id)
        .filter(MemoryChain.id == chain_id)
        .first()
    )
    if not chain:
        raise ChainNotFoundError(f"Chain not found: {chain_id}")
    return chain


def _normalize_seq(seq: int) -> int:
    if seq < 1:
        return 1
    return seq


def _ensure_memory_exists(
    db,
    tenant_id: str,
    memory_ref: str,
) -> tuple[str, int, str]:
    """Validate that a memory reference exists and return normalized info."""
    if ":" not in memory_ref:
        raise ValidationIssue(
            "memory_id must be in 'type:id' format",
            field="memory_id",
            error_type="invalid_format",
        )
    mem_type, mem_id_str = memory_ref.split(":", 1)
    mem_type = mem_type.strip().lower()
    mem_id_str = mem_id_str.strip()

    try:
        mem_id = int(mem_id_str)
    except ValueError as exc:
        raise ValidationIssue(
            "memory_id must have a numeric id",
            field="memory_id",
            error_type="invalid_id",
        ) from exc

    model = MEMORY_MODELS.get(mem_type)
    if not model:
        raise ValidationIssue(
            f"Unknown memory type: {mem_type}",
            field="memory_id",
            error_type="invalid_type",
        )

    record = (
        db.query(model)
        .filter(model.tenant_id == tenant_id)
        .filter(model.id == mem_id)
        .first()
    )
    if not record:
        raise ValidationIssue(
            f"Memory not found: {memory_ref}",
            field="memory_id",
            error_type="not_found",
        )

    return mem_type, mem_id, f"{mem_type}:{mem_id}"


def _normalize_memory_ref(raw, *, field: str = "memory_id") -> str:
    """Normalize a memory ref without requiring the record to exist."""
    if isinstance(raw, int):
        if raw <= 0:
            raise ValidationIssue(
                f"{field} must be a positive integer",
                field=field,
                error_type="invalid_id",
            )
        return f"observation:{raw}"
    if not isinstance(raw, str):
        raise ValidationIssue(
            f"{field} must be a string or integer",
            field=field,
            error_type="invalid_type",
        )
    value = raw.strip()
    if not value:
        raise ValidationIssue(
            f"{field} is required",
            field=field,
            error_type="required",
        )
    if ":" in value:
        mem_type, mem_id = value.split(":", 1)
        mem_type = mem_type.strip().lower()
        mem_id = mem_id.strip()
        if not mem_type or not mem_id:
            raise ValidationIssue(
                f"{field} must include both type and id",
                field=field,
                error_type="invalid_type",
            )
        if mem_type not in MEMORY_MODELS:
            raise ValidationIssue(
                f"Unknown memory type: {mem_type}",
                field=field,
                error_type="invalid_type",
            )
        return f"{mem_type}:{mem_id}"
    if value.isdigit():
        return f"observation:{value}"
    raise ValidationIssue(
        f"{field} must be in 'type:id' format",
        field=field,
        error_type="invalid_type",
    )


@service_tool
def memory_create_chain(
    title: Optional[str] = None,
    kind: Optional[str] = None,
    meta: Optional[dict] = None,
    initial_memory_ids: Optional[list[str]] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Create a new memory chain with optional initial items."""
    _validate_optional_text(title, "title", MAX_TITLE_LENGTH)
    _validate_optional_text(kind, "kind", MAX_SHORT_TEXT_LENGTH)
    if meta is not None:
        _validate_metadata(meta, "meta")

    chain_type = kind or "chain"
    tenant_id = resolve_tenant_id(context, required=True)
    bytes_to_write = estimate_chain_create_bytes(
        chain_type=chain_type,
        title=title,
        metadata=meta,
        store_id=None,
        scope=None,
    )

    db = DB.SessionLocal()
    try:
        if initial_memory_ids:
            for memory_id in initial_memory_ids:
                normalized = _normalize_memory_ref(memory_id, field="initial_memory_ids")
                mem_type, mem_id = normalized.split(":", 1)
                bytes_to_write += estimate_chain_append_bytes(
                    item_type=mem_type,
                    item_id=mem_id,
                    text=None,
                    role=None,
                )

        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)

        chain = MemoryChain(
            tenant_id=tenant_id,
            created_by_user_id=context.auth.user_id if context and context.auth else None,
            created_by_type=context.auth.actor if context and context.auth else None,
            created_by_id=str(context.auth.user_id)
            if context and context.auth and context.auth.user_id is not None
            else None,
            title=title,
            kind=chain_type,
            meta=meta,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(chain)
        db.flush()

        items = []
        if initial_memory_ids:
            for idx, memory_id in enumerate(initial_memory_ids, start=1):
                mem_type, mem_id, normalized = _ensure_memory_exists(
                    db,
                    tenant_id,
                    _normalize_memory_ref(memory_id, field="initial_memory_ids"),
                )
                item = MemoryChainItem(
                    tenant_id=tenant_id,
                    chain_id=chain.id,
                    memory_id=normalized,
                    observation_id=mem_id if mem_type == "observation" else None,
                    seq=_normalize_seq(idx),
                    role=None,
                    linked_at=datetime.utcnow(),
                )
                db.add(item)
                items.append(item)

        db.commit()
        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)
        return {
            "ack": "SUCCESS",
            "status": "created",
            "message": f"Chain created successfully (id={chain.id})",
            "chain": _serialize_chain(chain, items),
        }
    except IntegrityError as exc:
        db.rollback()
        raise ChainConflictError("Chain creation conflict") from exc
    finally:
        db.close()


@service_tool
def memory_get_chain(
    chain_id: str,
    context: Optional[RequestContext] = None,
) -> dict:
    """Get chain details with all items."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        chain = _require_chain(db, tenant_id, chain_id)
        rows = (
            db.query(MemoryChainItem)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
            .order_by(MemoryChainItem.seq.asc())
            .all()
        )
        return {"status": "ok", "chain": _serialize_chain(chain, rows)}
    finally:
        db.close()


@service_tool
def memory_add_to_chain(
    chain_id: str,
    memory_id: str,
    seq: Optional[int] = None,
    role: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Add a memory item to an existing chain."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(memory_id, "memory_id", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(role, "role", MAX_SHORT_TEXT_LENGTH)

    if seq is not None and (not isinstance(seq, int) or seq < 1):
        raise ValidationIssue(
            "seq must be a positive integer",
            field="seq",
            error_type="invalid_value",
        )

    tenant_id = resolve_tenant_id(context, required=True)
    normalized_ref = _normalize_memory_ref(memory_id, field="memory_id")
    mem_type, mem_id = normalized_ref.split(":", 1)
    bytes_to_write = estimate_chain_append_bytes(
        item_type=mem_type,
        item_id=mem_id,
        text=None,
        role=role,
    )

    db = DB.SessionLocal()
    try:
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        chain = _require_chain(db, tenant_id, chain_id)
        chain.updated_at = datetime.utcnow()

        mem_type, mem_id_value, normalized_ref = _ensure_memory_exists(
            db,
            tenant_id,
            normalized_ref,
        )
        observation_id = mem_id_value if mem_type == "observation" else None

        existing = (
            db.query(MemoryChainItem.id)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
            .filter(MemoryChainItem.memory_id == normalized_ref)
            .first()
        )
        if existing:
            raise ValidationIssue(
                "memory already in chain",
                field="memory_id",
                error_type="conflict",
            )

        if seq is None:
            max_seq = (
                db.query(func.max(MemoryChainItem.seq))
                .filter(MemoryChainItem.tenant_id == tenant_id)
                .filter(MemoryChainItem.chain_id == chain_id)
                .scalar()
            )
            seq_value = _normalize_seq((max_seq or 0) + 1)
        else:
            seq_value = _normalize_seq(seq)
            existing_seq = (
                db.query(MemoryChainItem.id)
                .filter(MemoryChainItem.tenant_id == tenant_id)
                .filter(MemoryChainItem.chain_id == chain_id)
                .filter(MemoryChainItem.seq == seq_value)
                .first()
            )
            if existing_seq:
                raise ValidationIssue(
                    "seq already in use for chain",
                    field="seq",
                    error_type="conflict",
                )

        item = MemoryChainItem(
            tenant_id=tenant_id,
            chain_id=chain_id,
            memory_id=normalized_ref,
            observation_id=observation_id,
            seq=seq_value,
            role=role,
            linked_at=datetime.utcnow(),
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)

        return {
            "ack": "SUCCESS",
            "status": "added",
            "message": f"Memory added to chain (seq={item.seq})",
            "item": _serialize_chain_item(item),
        }
    except IntegrityError:
        db.rollback()
        return {"status": "error", "message": "Chain entry conflict"}
    finally:
        db.close()


@service_tool
def memory_remove_from_chain(
    chain_id: str,
    memory_id: str,
    context: Optional[RequestContext] = None,
) -> dict:
    """Remove an item from a chain."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(memory_id, "memory_id", MAX_SHORT_TEXT_LENGTH)

    tenant_id = resolve_tenant_id(context, required=True)
    normalized_ref = _normalize_memory_ref(memory_id, field="memory_id")

    db = DB.SessionLocal()
    try:
        row = (
            db.query(MemoryChainItem)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
            .filter(MemoryChainItem.memory_id == normalized_ref)
            .first()
        )
        if not row:
            return {"status": "error", "message": "Memory not found in chain"}
        chain = (
            db.query(MemoryChain)
            .filter(MemoryChain.tenant_id == tenant_id)
            .filter(MemoryChain.id == row.chain_id)
            .first()
        )
        if chain:
            chain.updated_at = datetime.utcnow()
        db.delete(row)
        db.commit()
        return {"ack": "SUCCESS", "status": "removed", "removed": True}
    finally:
        db.close()


@service_tool
def memory_list_chains_for_memory(
    memory_id: str,
    context: Optional[RequestContext] = None,
) -> dict:
    """List all chains containing this memory."""
    _validate_required_text(memory_id, "memory_id", MAX_SHORT_TEXT_LENGTH)
    tenant_id = resolve_tenant_id(context, required=True)
    normalized_ref = _normalize_memory_ref(memory_id, field="memory_id")

    db = DB.SessionLocal()
    try:
        rows = (
            db.query(MemoryChain)
            .join(MemoryChainItem, MemoryChainItem.chain_id == MemoryChain.id)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.memory_id == normalized_ref)
            .order_by(MemoryChain.created_at.asc())
            .all()
        )
        return {"status": "ok", "chains": [_serialize_chain(row) for row in rows]}
    finally:
        db.close()


@service_tool
def memory_list_chains_for_observation(
    observation_id: int,
    context: Optional[RequestContext] = None,
) -> dict:
    """List all chains containing this observation."""
    if not isinstance(observation_id, int) or observation_id < 1:
        raise ValidationIssue(
            "observation_id must be a positive integer",
            field="observation_id",
            error_type="invalid_value",
        )
    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        rows = (
            db.query(MemoryChain)
            .join(MemoryChainItem, MemoryChainItem.chain_id == MemoryChain.id)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.observation_id == observation_id)
            .order_by(MemoryChain.created_at.asc())
            .all()
        )
        return {"status": "ok", "chains": [_serialize_chain(row) for row in rows]}
    finally:
        db.close()


@service_tool
def memory_chain_create(
    chain_type: str,
    name: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    store_id: Optional[str] = None,
    scope: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Create a new memory chain."""
    _validate_required_text(chain_type, "chain_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(name, "name", MAX_TITLE_LENGTH)
    _validate_optional_text(title, "title", MAX_TITLE_LENGTH)
    _validate_optional_text(store_id, "store_id", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(scope, "scope", MAX_SHORT_TEXT_LENGTH)
    if metadata is not None:
        _validate_metadata(metadata, "metadata")

    resolved_title = name if name is not None else title
    meta = _merge_chain_meta(None, metadata, store_id=store_id, scope=scope)

    tenant_id = resolve_tenant_id(context, required=True)
    bytes_to_write = estimate_chain_create_bytes(
        chain_type=chain_type,
        title=resolved_title,
        metadata=metadata,
        store_id=store_id,
        scope=scope,
    )

    db = DB.SessionLocal()
    try:
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        chain = MemoryChain(
            tenant_id=tenant_id,
            created_by_user_id=context.auth.user_id if context and context.auth else None,
            created_by_type=context.auth.actor if context and context.auth else None,
            created_by_id=str(context.auth.user_id)
            if context and context.auth and context.auth.user_id is not None
            else None,
            title=resolved_title,
            kind=chain_type,
            meta=meta,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(chain)
        db.commit()
        db.refresh(chain)
        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)
        return {"status": "ok", "chain": _serialize_chain(chain, [])}
    except IntegrityError as exc:
        db.rollback()
        raise ChainConflictError("Chain creation conflict") from exc
    finally:
        db.close()


@service_tool
def memory_chain_append(
    chain_id: str,
    item_type: str,
    item_id: Optional[str] = None,
    text: Optional[str] = None,
    role: Optional[str] = None,
    timestamp: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Append an item to a chain."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(item_type, "item_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(role, "role", MAX_SHORT_TEXT_LENGTH)
    if text is not None:
        _validate_required_text(text, "text", MAX_TEXT_LENGTH)

    if item_id and text:
        return {"status": "error", "message": "Provide item_id or text, not both"}
    if not item_id and not text:
        return {"status": "error", "message": "item_id or text is required"}

    resolved_timestamp = _parse_timestamp(timestamp)
    tenant_id = resolve_tenant_id(context, required=True)

    normalized_item_type = item_type.strip().lower()
    resolved_item_id = None
    if item_id is not None:
        if isinstance(item_id, int):
            resolved_item_id = str(item_id)
        elif isinstance(item_id, str):
            resolved_item_id = item_id
        else:
            return {"status": "error", "message": "item_id must be a string or int"}
        _validate_optional_text(resolved_item_id, "item_id", MAX_SHORT_TEXT_LENGTH)

        if normalized_item_type == "memory":
            raw_ref = resolved_item_id.strip()
            if ":" not in raw_ref:
                return {
                    "status": "error",
                    "message": "item_type='memory' requires item_id with type prefix (e.g., 'observation:17')",
                }
            mem_type, _ = raw_ref.split(":", 1)
            mem_type = mem_type.strip().lower()
            if not mem_type:
                return {
                    "status": "error",
                    "message": "item_type='memory' requires item_id with type prefix (e.g., 'observation:17')",
                }
            if mem_type not in MEMORY_MODELS:
                raise ValidationIssue(
                    f"Unknown memory type: {mem_type}",
                    field="item_id",
                    error_type="invalid_type",
                )
            normalized_item_type = mem_type

    bytes_to_write = estimate_chain_append_bytes(
        item_type=normalized_item_type,
        item_id=resolved_item_id,
        text=None,
        role=role,
    )
    if text is not None:
        bytes_to_write += estimate_observation_bytes(text)
        bytes_to_write += estimate_embedding_bytes()

    db = DB.SessionLocal()
    created_observation = None
    try:
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        chain = _require_chain(db, tenant_id, chain_id)
        chain.updated_at = datetime.utcnow()

        observation_id = None
        if text is not None:
            # Create new observation for freeform text
            if normalized_item_type not in {"observation", "text"}:
                raise ValidationIssue(
                    "item_type must be 'observation' for freeform text",
                    field="item_type",
                    error_type="invalid_value",
                )
            created_observation = Observation(
                observation=text,
                tenant_id=tenant_id,
                created_by_user_pk=_context_user_pk(context),
                timestamp=resolved_timestamp or datetime.utcnow(),
            )
            db.add(created_observation)
            db.flush()
            memory_ref = f"observation:{created_observation.id}"
            observation_id = created_observation.id
        else:
            # Reference existing memory
            raw_id = resolved_item_id.strip()
            if ":" in raw_id:
                mem_type, mem_id = raw_id.split(":", 1)
                mem_type = mem_type.strip().lower()
                if mem_type != normalized_item_type:
                    raise ValidationIssue(
                        "item_type does not match item_id",
                        field="item_type",
                        error_type="invalid_value",
                    )
                memory_ref = f"{mem_type}:{mem_id.strip()}"
            else:
                memory_ref = f"{normalized_item_type}:{raw_id}"
            mem_type, mem_id, memory_ref = _ensure_memory_exists(db, tenant_id, memory_ref)
            if mem_type == "observation":
                observation_id = mem_id

        # Check for duplicates
        existing = (
            db.query(MemoryChainItem.id)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
            .filter(MemoryChainItem.memory_id == memory_ref)
            .first()
        )
        if existing:
            raise ValidationIssue(
                "memory already in chain",
                field="item_id",
                error_type="conflict",
            )

        # Get next sequence number
        max_seq = (
            db.query(func.max(MemoryChainItem.seq))
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
            .scalar()
        )
        seq = _normalize_seq((max_seq or 0) + 1)

        item = MemoryChainItem(
            tenant_id=tenant_id,
            chain_id=chain_id,
            memory_id=memory_ref,
            observation_id=observation_id,
            seq=seq,
            role=role,
            linked_at=resolved_timestamp or datetime.utcnow(),
        )
        db.add(item)
        db.commit()
        db.refresh(item)

        if created_observation is not None:
            _store_embedding(
                db,
                "observation",
                created_observation.id,
                text,
                tenant_id=tenant_id,
            )

        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)

        return {
            "status": "ok",
            "entry_id": str(item.id),
            "seq": item.seq,
            "entry": _serialize_chain_item(item),
        }
    except ChainNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    except IntegrityError as exc:
        db.rollback()
        return {"status": "error", "message": "Chain entry conflict"}
    finally:
        db.close()


@service_tool
def memory_chain_get(
    chain_id: str,
    limit: int = 50,
    cursor: Optional[str] = None,
    order: str = "asc",
    context: Optional[RequestContext] = None,
) -> dict:
    """Get chain details with paginated items."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    order_value = (order or "asc").strip().lower()
    if order_value not in {"asc", "desc"}:
        return {"status": "error", "message": "order must be 'asc' or 'desc'"}

    cursor_value = None
    if cursor is not None:
        if isinstance(cursor, int):
            cursor_value = cursor
        elif isinstance(cursor, str) and cursor.strip().isdigit():
            cursor_value = int(cursor.strip())
        else:
            return {"status": "error", "message": "cursor must be an integer"}

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        chain = _require_chain(db, tenant_id, chain_id)
        query = (
            db.query(MemoryChainItem)
            .filter(MemoryChainItem.tenant_id == tenant_id)
            .filter(MemoryChainItem.chain_id == chain_id)
        )
        if cursor_value is not None:
            if order_value == "asc":
                query = query.filter(MemoryChainItem.seq > cursor_value)
            else:
                query = query.filter(MemoryChainItem.seq < cursor_value)
        if order_value == "asc":
            query = query.order_by(MemoryChainItem.seq.asc())
        else:
            query = query.order_by(MemoryChainItem.seq.desc())
        rows = query.limit(limit + 1).all()
        next_cursor = None
        if len(rows) > limit:
            next_cursor = rows[limit - 1].seq
            rows = rows[:limit]
        return {
            "status": "ok",
            "chain": _serialize_chain(chain),
            "entries": [_serialize_chain_item(item) for item in rows],
            "next_cursor": next_cursor,
            "order": order_value,
            "limit": limit,
        }
    except ChainNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@service_tool
def memory_chain_list(
    chain_type: Optional[str] = None,
    store_id: Optional[str] = None,
    name_contains: Optional[str] = None,
    limit: int = 100,
    context: Optional[RequestContext] = None,
) -> dict:
    """List chains with optional filtering."""
    _validate_optional_text(chain_type, "chain_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(store_id, "store_id", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(name_contains, "name_contains", MAX_SHORT_TEXT_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        query = db.query(MemoryChain).filter(MemoryChain.tenant_id == tenant_id)
        if chain_type:
            query = query.filter(MemoryChain.kind == chain_type)
        if name_contains:
            query = query.filter(MemoryChain.title.ilike(f"%{name_contains}%"))
        if store_id:
            rows = query.order_by(MemoryChain.created_at.asc()).all()
            rows = [
                row
                for row in rows
                if (row.meta or {}).get("store_id") == store_id
            ]
            rows = rows[:limit]
        else:
            rows = query.order_by(MemoryChain.created_at.asc()).limit(limit).all()
        return {"status": "ok", "chains": [_serialize_chain(row) for row in rows]}
    finally:
        db.close()


@service_tool
def memory_chain_update(
    chain_id: str,
    name: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    status: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Update chain metadata."""
    _validate_required_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(name, "name", MAX_TITLE_LENGTH)
    _validate_optional_text(title, "title", MAX_TITLE_LENGTH)
    _validate_optional_text(status, "status", MAX_SHORT_TEXT_LENGTH)
    if metadata is not None:
        _validate_metadata(metadata, "metadata")

    resolved_title = name if name is not None else title
    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        bytes_to_write = estimate_chain_update_bytes(
            title=resolved_title,
            metadata=metadata,
            status=status,
        )
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        chain = _require_chain(db, tenant_id, chain_id)
        if resolved_title is not None:
            chain.title = resolved_title
        if metadata is not None or status is not None:
            chain.meta = _merge_chain_meta(
                chain.meta,
                metadata,
                status=status,
            )
        chain.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(chain)
        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)
        return {"status": "ok", "chain": _serialize_chain(chain)}
    except ChainNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@service_tool
def memory_chain_entry_archive(
    entry_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    seq: Optional[int] = None,
    context: Optional[RequestContext] = None,
) -> dict:
    """Archive (soft-delete) a chain entry."""
    if not entry_id and not (chain_id and seq is not None):
        return {"status": "error", "message": "entry_id or chain_id+seq is required"}
    if entry_id is not None:
        _validate_optional_text(entry_id, "entry_id", MAX_SHORT_TEXT_LENGTH)
    if chain_id is not None:
        _validate_optional_text(chain_id, "chain_id", MAX_SHORT_TEXT_LENGTH)
    if seq is not None and (not isinstance(seq, int) or seq < 1):
        return {"status": "error", "message": "seq must be a positive integer"}

    tenant_id = resolve_tenant_id(context, required=True)

    db = DB.SessionLocal()
    try:
        if entry_id:
            row = (
                db.query(MemoryChainItem)
                .filter(MemoryChainItem.tenant_id == tenant_id)
                .filter(MemoryChainItem.id == entry_id)
                .first()
            )
        else:
            row = (
                db.query(MemoryChainItem)
                .filter(MemoryChainItem.tenant_id == tenant_id)
                .filter(MemoryChainItem.chain_id == chain_id)
                .filter(MemoryChainItem.seq == seq)
                .first()
            )
        if not row:
            return {"status": "error", "message": "Chain entry not found"}
        chain = (
            db.query(MemoryChain)
            .filter(MemoryChain.tenant_id == tenant_id)
            .filter(MemoryChain.id == row.chain_id)
            .first()
        )
        if chain:
            chain.updated_at = datetime.utcnow()
        db.delete(row)
        db.commit()
        return {"status": "ok", "archived": True}
    finally:
        db.close()

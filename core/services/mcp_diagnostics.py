"""
MCP diagnostics and inventory helpers.
"""

from __future__ import annotations

from datetime import datetime
import os
import time
from typing import Optional

from sqlalchemy import text

import core.config as config
from core.context import RequestContext
from core.db import DB
from core.services.memory_docs import SPEC_VERSION
from core.services.memory_shared import service_tool


_START_TIME = time.monotonic()


def _get_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _auth_mode() -> str:
    return os.environ.get("MEMORYGATE_AUTH_MODE", "api_key").strip().lower()


def _auth_scopes() -> list[str]:
    raw = os.environ.get("MEMORYGATE_AUTH_SCOPES", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


TOOL_REGISTRY = [
    {
        "tool_name": "memory_store",
        "description": "Store a new observation.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_search",
        "description": "Search across memories.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "search_cold_memory",
        "description": "Search cold memories.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "archive_memory",
        "description": "Archive memory items.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "rehydrate_memory",
        "description": "Rehydrate archived memories.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "list_archive_candidates",
        "description": "List archive candidates.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_recall",
        "description": "Recall recent memories.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_stats",
        "description": "Get memory statistics.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_init_session",
        "description": "Initialize a memory session.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_store_document",
        "description": "Store a document memory.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_store_concept",
        "description": "Store a concept memory.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_concept",
        "description": "Fetch a concept memory.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_add_concept_alias",
        "description": "Add a concept alias.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_add_concept_relationship",
        "description": "Add a concept relationship.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_related_concepts",
        "description": "List related concepts.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_update_pattern",
        "description": "Create or update a pattern.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_pattern",
        "description": "Fetch a pattern memory.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_patterns",
        "description": "List memory patterns.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_user_guide",
        "description": "Get memory user guide.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_bootstrap",
        "description": "Bootstrap memory context.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_add_relationship",
        "description": "Add a memory relationship.",
        "enabled": True,
        "category": "relationships",
        "version": "v1",
    },
    {
        "tool_name": "memory_list_relationships",
        "description": "List memory relationships.",
        "enabled": True,
        "category": "relationships",
        "version": "v1",
    },
    {
        "tool_name": "memory_related",
        "description": "List related memory nodes.",
        "enabled": True,
        "category": "relationships",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_supersession",
        "description": "Get supersession info.",
        "enabled": True,
        "category": "relationships",
        "version": "v1",
    },
    {
        "tool_name": "relationship_add_residue",
        "description": "Attach residue to a relationship edge.",
        "enabled": True,
        "category": "residue",
        "version": "v1",
    },
    {
        "tool_name": "relationship_list_residue",
        "description": "List residue entries for an edge.",
        "enabled": True,
        "category": "residue",
        "version": "v1",
    },
    {
        "tool_name": "memory_chain_create",
        "description": "Create a chain (v2).",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "memory_create_chain",
        "description": "Create a chain with optional initial items (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_chain",
        "description": "Get full chain details (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_add_to_chain",
        "description": "Add a memory to a chain (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_remove_from_chain",
        "description": "Remove a memory from a chain (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_list_chains_for_memory",
        "description": "List chains for a memory ref (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_list_chains_for_observation",
        "description": "List chains for an observation id (v1).",
        "enabled": True,
        "category": "chains",
        "version": "v1",
    },
    {
        "tool_name": "memory_chain_append",
        "description": "Append an entry to a chain.",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "memory_chain_get",
        "description": "Get chain entries with pagination.",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "memory_chain_list",
        "description": "List chains with filters.",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "memory_chain_update",
        "description": "Update chain metadata.",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "memory_chain_entry_archive",
        "description": "Archive or delete a chain entry.",
        "enabled": True,
        "category": "chains",
        "version": "v2",
    },
    {
        "tool_name": "tool_inventory_status",
        "description": "List available MCP tools.",
        "enabled": True,
        "category": "diagnostics",
        "version": "v1",
    },
    {
        "tool_name": "capabilities_get",
        "description": "Get server capabilities.",
        "enabled": True,
        "category": "diagnostics",
        "version": "v1",
    },
    {
        "tool_name": "health_status",
        "description": "Get health status.",
        "enabled": True,
        "category": "diagnostics",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_by_ref",
        "description": "Fetch a memory by ref.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "memory_get_many_by_refs",
        "description": "Batch fetch memories by refs.",
        "enabled": True,
        "category": "memory",
        "version": "v1",
    },
    {
        "tool_name": "stores_list_accessible",
        "description": "List accessible stores.",
        "enabled": True,
        "category": "stores",
        "version": "v1",
    },
    {
        "tool_name": "stores_get_active",
        "description": "Get active store.",
        "enabled": True,
        "category": "stores",
        "version": "v1",
    },
    {
        "tool_name": "stores_set_active",
        "description": "Set active store.",
        "enabled": True,
        "category": "stores",
        "version": "v1",
    },
    {
        "tool_name": "agent_anchor_set",
        "description": "Set agent anchor chain.",
        "enabled": True,
        "category": "identity",
        "version": "v1",
    },
    {
        "tool_name": "agent_anchor_get",
        "description": "Get agent anchor chain.",
        "enabled": True,
        "category": "identity",
        "version": "v1",
    },
]


def _tool_registry_payload() -> list[dict]:
    return [dict(entry) for entry in TOOL_REGISTRY]


@service_tool
def tool_inventory_status(context: Optional[RequestContext] = None) -> dict:
    return {
        "status": "ok",
        "tool_count": len(TOOL_REGISTRY),
        "tools": _tool_registry_payload(),
    }


@service_tool
def capabilities_get(context: Optional[RequestContext] = None) -> dict:
    max_payload_bytes = int(
        os.environ.get("MEMORYGATE_MAX_PAYLOAD_BYTES", config.MAX_METADATA_BYTES)
    )
    default_result_limit = int(
        os.environ.get("MEMORYGATE_DEFAULT_RESULT_LIMIT", 20)
    )
    return {
        "version": 1,
        "server": {
            "name": "MemoryGate",
            "version": SPEC_VERSION,
            "build": os.environ.get("MEMORYGATE_BUILD", "unknown"),
            "env": os.environ.get("MEMORYGATE_ENV", "unknown"),
        },
        "auth": {
            "mode": _auth_mode(),
            "scopes": _auth_scopes(),
        },
        "features": {
            "observations": True,
            "patterns": True,
            "concepts": True,
            "documents": True,
            "artifacts": False,
            "relationships": True,
            "memory_chains": True,
            "audit": True,
            "exports": _get_bool(os.environ.get("MEMORYGATE_EXPORTS_ENABLED"), False),
        },
        "limits": {
            "max_payload_bytes": max_payload_bytes,
            "max_batch_size": config.MAX_LIST_ITEMS,
            "default_result_limit": default_result_limit,
            "hard_result_limit": config.MAX_RESULT_LIMIT,
            "max_text_length": config.MAX_TEXT_LENGTH,
            "max_metadata_bytes": config.MAX_METADATA_BYTES,
        },
        "billing": {
            "plans_enabled": _get_bool(os.environ.get("MEMORYGATE_PLANS_ENABLED"), False),
            "enforces_quota": _get_bool(
                os.environ.get("MEMORYGATE_ENFORCES_QUOTA"),
                config.STORAGE_QUOTA_BYTES > 0,
            ),
            "storage_quota_bytes": config.STORAGE_QUOTA_BYTES,
        },
    }


def _db_ping(timeout_ms: int) -> dict:
    if DB.SessionLocal is None:
        return {"status": "down", "detail": "db_not_initialized"}
    start = time.monotonic()
    db = DB.SessionLocal()
    try:
        if config.DB_BACKEND_EFFECTIVE == "postgres":
            try:
                db.execute(text("SET LOCAL statement_timeout = :ms"), {"ms": timeout_ms})
            except Exception:
                pass
        db.execute(text("SELECT 1"))
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"status": "down", "error": str(exc), "latency_ms": latency_ms}
    finally:
        try:
            db.close()
        except Exception:
            pass


@service_tool
def health_status(context: Optional[RequestContext] = None) -> dict:
    timeout_ms = int(os.environ.get("MEMORYGATE_DB_PING_TIMEOUT_MS", "500"))
    db_status = _db_ping(timeout_ms)
    vector_status = "disabled"
    if config.VECTOR_BACKEND_EFFECTIVE != "none":
        vector_status = "ok"
        if config.EMBEDDING_PROVIDER == "none":
            vector_status = "degraded"

    components = {
        "mcp": {"status": "ok"},
        "auth": {"status": "ok", "mode": _auth_mode()},
        "db": db_status,
        "vector_index": {
            "status": vector_status,
            "backend": config.VECTOR_BACKEND_EFFECTIVE,
        },
        "storage": {"status": "ok"},
    }

    overall = "ok"
    if any(component["status"] == "down" for component in components.values()):
        overall = "down"
    elif any(component["status"] == "degraded" for component in components.values()):
        overall = "degraded"

    return {
        "status": overall,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": int(time.monotonic() - _START_TIME),
        "components": components,
    }

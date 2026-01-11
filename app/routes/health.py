"""
Health and dependency endpoints.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

import core.config as config
from core.db import DB, _get_schema_revisions
from core.errors import EmbeddingProviderError
from core.mcp import tool_inventory_status
from core.services import memory_service


router = APIRouter()


def _check_db_health() -> dict:
    if DB.engine is None:
        return {"ok": False, "error": "db_not_initialized"}

    try:
        with DB.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            ext_version = None
            pgvector_installed = True
            if config.DB_BACKEND == "postgres" and config.VECTOR_BACKEND_EFFECTIVE == "pgvector":
                ext_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
                pgvector_installed = bool(ext_version)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    current_rev, head_rev = _get_schema_revisions(DB.engine)
    schema_ok = head_rev is None or current_rev == head_rev
    return {
        "ok": schema_ok,
        "pgvector_installed": pgvector_installed,
        "pgvector_version": ext_version,
        "schema_revision": current_rev,
        "schema_expected": head_rev,
        "schema_up_to_date": schema_ok,
    }


def _check_embedding_health(check_external: bool) -> dict:
    breaker_status = memory_service.embedding_circuit_breaker.status()
    embedding_status = {
        "status": "unknown",
        "provider": config.EMBEDDING_PROVIDER,
        "circuit_breaker": breaker_status,
        "checked": False,
    }

    if config.EMBEDDING_PROVIDER == "none":
        embedding_status["status"] = "disabled"
        return embedding_status

    if breaker_status.get("open"):
        embedding_status["status"] = "cooldown"
        return embedding_status

    if check_external and config.EMBEDDING_HEALTHCHECK_ENABLED:
        embedding_status["checked"] = True
        start = time.time()
        try:
            memory_service.embed_text_sync("healthcheck")
            embedding_status["status"] = "ok"
            embedding_status["latency_ms"] = int((time.time() - start) * 1000)
        except EmbeddingProviderError as exc:
            embedding_status["status"] = "error"
            embedding_status["error"] = str(exc)
        return embedding_status

    embedding_status["status"] = "skipped" if check_external else "ready"
    return embedding_status


@router.get("/health")
async def health():
    """Health check endpoint."""
    db_health = _check_db_health()
    embedding_status = _check_embedding_health(check_external=False)
    vector_required = config.DB_BACKEND == "postgres" and config.VECTOR_BACKEND_EFFECTIVE == "pgvector"
    if not db_health.get("ok") or (vector_required and not db_health.get("pgvector_installed")):
        raise HTTPException(
            status_code=503,
            detail={"database": db_health, "embedding_provider": embedding_status},
        )

    return {
        "status": "healthy",
        "service": "MemoryGate",
        "version": "0.1.0",
        "instance_id": os.environ.get("MEMORYGATE_INSTANCE_ID", "memorygate-1"),
        "tenant_mode": config.TENANCY_MODE,
        "database": db_health,
        "embedding_provider": embedding_status,
    }


@router.get("/health/tools")
async def health_tools():
    """Tool inventory health check."""
    tool_inventory = await tool_inventory_status()
    if tool_inventory.get("tool_count", 0) == 0:
        raise HTTPException(status_code=503, detail={"tool_inventory": tool_inventory})

    return {
        "status": "healthy",
        "service": "MemoryGate",
        "tool_inventory": tool_inventory,
    }


@router.get("/health/deps")
async def health_deps():
    """Dependency health checks (optional embedding provider probe)."""
    db_health = _check_db_health()
    vector_required = config.DB_BACKEND == "postgres" and config.VECTOR_BACKEND_EFFECTIVE == "pgvector"
    if not db_health.get("ok") or (vector_required and not db_health.get("pgvector_installed")):
        raise HTTPException(status_code=503, detail={"database": db_health})

    embedding_status = _check_embedding_health(check_external=True)

    return {
        "status": "healthy",
        "service": "MemoryGate",
        "tenant_mode": config.TENANCY_MODE,
        "database": db_health,
        "embedding_provider": embedding_status,
    }

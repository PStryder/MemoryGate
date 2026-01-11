"""
Standalone FastAPI app wiring for MemoryGate.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

import core.config as config
from core.db import DB, init_db
from core.mcp import (
    mcp_sse_app,
    mcp_stream_app,
    MCPRouteNormalizerASGI,
    MemoryGateAliasASGI,
    _mcp_tool_inventory_check,
)
from core.services import memory_service
from core.services import retention_service
from app.middleware import configure_middleware
from app.routes.health import router as health_router
from app.routes.root import router as root_router
from app.routes.oauth_discovery import router as oauth_discovery_router
from app.routes.oauth_routes import router as auth_router
from mcp_auth_gate import MCPAuthGateASGI
from oauth_models import cleanup_expired_sessions, cleanup_expired_states


rate_limiter = None
cleanup_task = None
retention_task = None
embedding_backfill_task = None


async def _run_cleanup_once() -> None:
    if DB.SessionLocal is None:
        return
    db = DB.SessionLocal()
    try:
        cleanup_expired_states(db)
        cleanup_expired_sessions(db)
    finally:
        db.close()


async def _cleanup_loop() -> None:
    if config.CLEANUP_INTERVAL_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(config.CLEANUP_INTERVAL_SECONDS)
        try:
            await _run_cleanup_once()
        except Exception as exc:
            config.logger.warning(f"Cleanup task error: {exc}")


async def _retention_loop() -> None:
    if config.RETENTION_TICK_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(config.RETENTION_TICK_SECONDS)
        try:
            retention_service.run_retention_tick()
        except Exception as exc:
            config.logger.warning(f"Retention task error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    global cleanup_task, retention_task, embedding_backfill_task, rate_limiter
    init_db()
    memory_service.init_http_client()
    if config.CLEANUP_INTERVAL_SECONDS > 0:
        await _run_cleanup_once()
        cleanup_task = asyncio.create_task(_cleanup_loop())
    if config.RETENTION_TICK_SECONDS > 0:
        retention_task = asyncio.create_task(_retention_loop())
    if config.EMBEDDING_BACKFILL_ENABLED:
        await asyncio.to_thread(memory_service._run_embedding_backfill)
        if config.EMBEDDING_BACKFILL_INTERVAL_SECONDS > 0:
            embedding_backfill_task = asyncio.create_task(memory_service._embedding_backfill_loop())
    try:
        async with mcp_stream_app.lifespan(mcp_stream_app):
            yield
    finally:
        if retention_task:
            retention_task.cancel()
            try:
                await retention_task
            except asyncio.CancelledError:
                pass
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        if embedding_backfill_task:
            embedding_backfill_task.cancel()
            try:
                await embedding_backfill_task
            except asyncio.CancelledError:
                pass
        memory_service.cleanup_http_client()
        if rate_limiter:
            await rate_limiter.close()
        if DB.engine:
            DB.engine.dispose()


app = FastAPI(title="MemoryGate", redirect_slashes=False, lifespan=lifespan)
rate_limiter = configure_middleware(app)

# Mount OAuth discovery and authorization routes (for Claude Desktop MCP)
app.include_router(oauth_discovery_router)

# Mount OAuth user management routes
app.include_router(auth_router)

# Health and root endpoints
app.include_router(health_router)
app.include_router(root_router)

# Mount MCP apps with auth gate (pass DB class for dynamic lookup)
mcp_sse_entry_app = MCPAuthGateASGI(mcp_sse_app, lambda: DB.SessionLocal, _mcp_tool_inventory_check)
mcp_stream_entry_app = MCPAuthGateASGI(mcp_stream_app, lambda: DB.SessionLocal, _mcp_tool_inventory_check)
app.mount("/mcp/", mcp_sse_entry_app)
app.mount("/MemoryGate", MemoryGateAliasASGI(mcp_stream_entry_app))


# =============================================================================
# ASGI Application (module-level for production deployment)
# =============================================================================

asgi_app = MCPRouteNormalizerASGI(app)

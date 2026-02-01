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
from mcp_auth_gate import MCPAuthGateASGI
from oauth_models import cleanup_expired_sessions, cleanup_expired_states


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
    init_db()
    memory_service.init_http_client()
    if config.CLEANUP_INTERVAL_SECONDS > 0:
        await _run_cleanup_once()
        app.state.cleanup_task = asyncio.create_task(_cleanup_loop())
    if config.RETENTION_TICK_SECONDS > 0:
        app.state.retention_task = asyncio.create_task(_retention_loop())
    if config.EMBEDDING_BACKFILL_ENABLED:
        await asyncio.to_thread(memory_service._run_embedding_backfill)
        if config.EMBEDDING_BACKFILL_INTERVAL_SECONDS > 0:
            app.state.embedding_backfill_task = asyncio.create_task(
                memory_service._embedding_backfill_loop()
            )
    try:
        async with mcp_stream_app.lifespan(mcp_stream_app):
            yield
    finally:
        retention_task = getattr(app.state, "retention_task", None)
        if retention_task:
            retention_task.cancel()
            try:
                await retention_task
            except asyncio.CancelledError:
                pass
        cleanup_task = getattr(app.state, "cleanup_task", None)
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        embedding_backfill_task = getattr(app.state, "embedding_backfill_task", None)
        if embedding_backfill_task:
            embedding_backfill_task.cancel()
            try:
                await embedding_backfill_task
            except asyncio.CancelledError:
                pass
        memory_service.cleanup_http_client()
        rate_limiter = getattr(app.state, "rate_limiter", None)
        if rate_limiter:
            await rate_limiter.close()
        if DB.engine:
            DB.engine.dispose()


app = FastAPI(title="MemoryGate", redirect_slashes=False, lifespan=lifespan)
app.state.cleanup_task = None
app.state.retention_task = None
app.state.embedding_backfill_task = None
app.state.rate_limiter = configure_middleware(app)

# REST endpoints removed; MCP is the only surface.

# Mount MCP apps with auth gate (pass DB class for dynamic lookup)
mcp_sse_entry_app = MCPAuthGateASGI(mcp_sse_app, lambda: DB.SessionLocal, _mcp_tool_inventory_check)
mcp_stream_entry_app = MCPAuthGateASGI(mcp_stream_app, lambda: DB.SessionLocal, _mcp_tool_inventory_check)
app.mount("/mcp/", mcp_sse_entry_app)
app.mount("/MemoryGate", MemoryGateAliasASGI(mcp_stream_entry_app))


# =============================================================================
# ASGI Application (module-level for production deployment)
# =============================================================================

asgi_app = MCPRouteNormalizerASGI(app)

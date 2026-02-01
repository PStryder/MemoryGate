"""
MCP authentication middleware for multi-tenant isolation.

Extracts API keys from MCP requests, validates them, and sets tenant context
for the duration of the request using contextvars (async-safe).
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Optional

from core.context import AuthContext, RequestContext
from core.db import DB
from app.auth import verify_request_api_key
import core.config as config

# Context var for storing per-request tenant context
_request_context: ContextVar[Optional[RequestContext]] = ContextVar(
    "mcp_request_context", default=None
)


def get_current_context() -> RequestContext:
    """Get current request context, or default if not set."""
    ctx = _request_context.get()
    if ctx is not None:
        return ctx
    # Fallback to anonymous context if not set
    return RequestContext(auth=AuthContext(actor="anonymous"))


class MCPAuthMiddleware:
    """
    ASGI middleware that validates API keys and sets tenant context.
    
    Wraps MCP endpoints to provide per-request authentication and tenant isolation.
    """
    
    def __init__(self, app):
        self.app = app
        self.require_auth = os.environ.get("REQUIRE_MCP_AUTH", "true").lower() == "true"

    def __getattr__(self, name):
        return getattr(self.app, name)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self.require_auth:
            await self.app(scope, receive, send)
            return
        
        # Extract headers (ASGI headers are bytes tuples)
        headers = {}
        for header_name, header_value in scope.get("headers", []):
            headers[header_name.decode("latin1").lower()] = header_value.decode("latin1")
        
        # Create DB session for auth check
        if DB.SessionLocal is None:
            config.logger.error("mcp_auth_middleware_no_db")
            await self._send_error(send, 500, "Database not initialized")
            return
        
        db = DB.SessionLocal()
        try:
            # Verify API key
            user = verify_request_api_key(db, headers)
            
            if user is None:
                await self._send_error(send, 401, "Valid API key required")
                return
            
            if not user.is_active:
                await self._send_error(send, 403, "User account inactive")
                return
            
            # Create auth context with tenant_id
            # For now, tenant_id = user_id (each user is their own tenant)
            # This can be extended later for org-level tenancy
            auth_ctx = AuthContext(
                user_id=user.id,
                tenant_id=str(user.id),
                actor=user.email or user.name or f"user_{user.id}"
            )
            
            req_ctx = RequestContext(
                auth=auth_ctx,
                source="mcp"
            )
            
            # Set context for this request
            token = _request_context.set(req_ctx)
            
            try:
                # Pass through to MCP app with context set
                await self.app(scope, receive, send)
            finally:
                # Clean up context
                _request_context.reset(token)
        
        except Exception as e:
            config.logger.error(
                "mcp_auth_middleware_error",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            await self._send_error(send, 500, "Internal server error")
        
        finally:
            db.close()
    
    async def _send_error(self, send, status_code: int, detail: str):
        """Send JSON error response."""
        body = json.dumps({"error": detail}).encode("utf-8")
        
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode("latin1")],
            ],
        })
        
        await send({
            "type": "http.response.body",
            "body": body,
        })

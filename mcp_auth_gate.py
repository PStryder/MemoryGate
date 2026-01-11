"""
MCP Authentication Gate - ASGI Middleware

Protects /mcp/ endpoints with API key authentication.
SSE-safe (no buffering), validates headers before forwarding to MCP app.
"""

import os
import json
from typing import Callable, Awaitable, Optional

from app.auth import verify_request_api_key


class MCPAuthGateASGI:
    """
    Pure ASGI middleware that gates /mcp/* with API key auth.
    
    - Validates Authorization: Bearer or X-API-Key header
    - Returns 401 if missing/invalid
    - Allows OPTIONS (CORS preflight)
    - No response buffering (SSE-safe)
    - Respects REQUIRE_MCP_AUTH env flag
    """
    
    def __init__(
        self,
        wrapped_app,
        sessionmaker_getter,
        tool_inventory_checker: Optional[Callable[[], Awaitable[dict]]] = None,
    ):
        self.wrapped_app = wrapped_app
        self.get_sessionmaker = sessionmaker_getter
        self.tool_inventory_checker = tool_inventory_checker
        self.require_auth = os.environ.get("REQUIRE_MCP_AUTH", "true").lower() == "true"
    
    async def __call__(self, scope, receive, send):
        # Only gate HTTP requests
        if scope["type"] != "http":
            await self.wrapped_app(scope, receive, send)
            return
        
        # Allow OPTIONS (CORS preflight)
        if scope["method"] == "OPTIONS":
            await self.wrapped_app(scope, receive, send)
            return
        
        # Check if auth is required
        if not self.require_auth:
            await self.wrapped_app(scope, receive, send)
            return
        
        # Validate API key
        headers = dict(scope.get("headers", []))
        # Convert bytes to str for header dict
        headers_str = {
            k.decode() if isinstance(k, bytes) else k: 
            v.decode() if isinstance(v, bytes) else v
            for k, v in headers.items()
        }
        
        # Get database session (late binding - sessionmaker set after init_db)
        SessionLocal = self.get_sessionmaker()
        if SessionLocal is None:
            await self._send_503(send)
            return
        
        db = SessionLocal()
        
        try:
            user = verify_request_api_key(db, headers_str)
            
            if not user:
                # Return 401 Unauthorized
                await self._send_401(scope, send)
                return
            
            # Check tool inventory health (refresh if empty)
            if self.tool_inventory_checker:
                status = await self.tool_inventory_checker()
                if status.get("tool_count", 0) == 0:
                    await self._send_503_tool_inventory(send, status)
                    return

            # User authenticated - forward to MCP app
            await self.wrapped_app(scope, receive, send)
            
        finally:
            db.close()
    
    async def _send_401(self, scope, send):
        """Send 401 Unauthorized response."""
        response_body = json.dumps({
            "error": "Unauthorized",
            "message": "Valid API key required. Use Authorization: Bearer mg_... or X-API-Key: mg_..."
        }).encode()
        
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ],
        })
        
        await send({
            "type": "http.response.body",
            "body": response_body,
        })
    
    async def _send_503(self, send):
        """Send 503 Service Unavailable response."""
        response_body = json.dumps({
            "error": "Service Unavailable",
            "message": "Database not initialized - server is starting up"
        }).encode()
        
        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ],
        })
        
        await send({
            "type": "http.response.body",
            "body": response_body,
        })

    async def _send_503_tool_inventory(self, send, status: dict):
        """Send 503 when tool inventory is empty."""
        retry_after = status.get("retry_after_seconds")
        response_body = json.dumps({
            "error": "Service Unavailable",
            "message": "Tool inventory empty - retry after backoff",
            "tool_inventory": status,
        }).encode()

        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(response_body)).encode()),
        ]
        if retry_after:
            headers.append((b"retry-after", str(retry_after).encode()))

        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": headers,
        })
        await send({
            "type": "http.response.body",
            "body": response_body,
        })

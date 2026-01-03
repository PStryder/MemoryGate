"""
MCP Authentication Gate - ASGI Middleware

Protects /mcp/ endpoints with API key authentication.
SSE-safe (no buffering), validates headers before forwarding to MCP app.
"""

import os
import json
from typing import Callable

from auth_middleware import verify_request_api_key


class MCPAuthGateASGI:
    """
    Pure ASGI middleware that gates /mcp/* with API key auth.
    
    - Validates Authorization: Bearer or X-API-Key header
    - Returns 401 if missing/invalid
    - Allows OPTIONS (CORS preflight)
    - No response buffering (SSE-safe)
    - Respects REQUIRE_MCP_AUTH env flag
    """
    
    def __init__(self, wrapped_app, session_factory):
        self.wrapped_app = wrapped_app
        self.session_factory = session_factory
        self.require_auth = os.environ.get("REQUIRE_MCP_AUTH", "true").lower() == "true"
    
    async def __call__(self, scope, receive, send):
        # Only gate HTTP requests to /mcp/* paths
        if scope["type"] != "http" or not scope["path"].startswith("/mcp/"):
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
        
        # Get database session (session_factory is callable that returns SessionLocal class)
        SessionLocal = self.session_factory()
        db = SessionLocal()
        
        try:
            user = verify_request_api_key(db, headers_str)
            
            if not user:
                # Return 401 Unauthorized
                await self._send_401(scope, send)
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

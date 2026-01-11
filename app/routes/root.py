"""
Root endpoint with service metadata.
"""

from __future__ import annotations

from fastapi import APIRouter

import core.config as config


router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "MemoryGate",
        "version": "0.1.0",
        "description": "Persistent Memory-as-a-Service for AI Agents",
        "tenant_mode": config.TENANCY_MODE,
        "embedding_model": config.EMBEDDING_MODEL,
        "endpoints": {
            "health": "/health",
            "health_deps": "/health/deps",
            "mcp": "/mcp",
            "memorygate": "/MemoryGate",
            "auth": {
                "client_credentials": "/auth/client",
                "login_google": "/auth/login/google",
                "login_github": "/auth/login/github",
                "me": "/auth/me",
                "api_keys": "/auth/api-keys",
            },
        },
    }

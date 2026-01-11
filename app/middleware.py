"""
Middleware configuration for the standalone FastAPI app.
"""

from __future__ import annotations

import os

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rate_limiter import (
    RateLimitMiddleware,
    build_rate_limiter_from_env,
    load_rate_limit_config_from_env,
)
from security_middleware import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    load_request_size_limit_config_from_env,
    load_security_headers_config_from_env,
)


def configure_middleware(app):
    """Configure security, rate limiting, and CORS middleware for the FastAPI app."""
    rate_limit_config = load_rate_limit_config_from_env()
    rate_limiter = build_rate_limiter_from_env(rate_limit_config)

    request_size_config = load_request_size_limit_config_from_env()
    security_headers_config = load_security_headers_config_from_env()

    # Request size limits (keep CORS outermost to add headers on 413 responses)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        config=request_size_config,
    )

    # Rate limiting middleware (outer CORS will still add headers on 429 responses)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=rate_limiter,
        config=rate_limit_config,
    )

    # Security headers (applies to all responses)
    app.add_middleware(
        SecurityHeadersMiddleware,
        config=security_headers_config,
    )

    # Optional host allowlist for production deployments
    trusted_hosts_env = os.environ.get("TRUSTED_HOSTS", "")
    trusted_hosts = [host.strip() for host in trusted_hosts_env.split(",") if host.strip()]
    if trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts,
        )

    # Add CORS middleware
    cors_allowed_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if cors_allowed_env.strip():
        allow_origins = [origin.strip() for origin in cors_allowed_env.split(",") if origin.strip()]
    else:
        allow_origins = [
            os.environ.get("FRONTEND_URL", "http://localhost:3000"),
            "http://localhost:3000",
            "https://memorygate.ai",
            "https://www.memorygate.ai",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return rate_limiter

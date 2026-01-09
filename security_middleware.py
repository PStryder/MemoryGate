"""
Security middlewares for MemoryGate.

- SecurityHeadersMiddleware: adds hardening headers
- RequestSizeLimitMiddleware: rejects oversized request bodies
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


def _get_bool(env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_str(env_name: str, default: str) -> str:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value


@dataclass(frozen=True)
class SecurityHeadersConfig:
    enabled: bool
    enable_hsts: bool
    hsts_max_age: int
    hsts_include_subdomains: bool
    hsts_preload: bool
    referrer_policy: str
    frame_options: str
    permissions_policy: str
    content_security_policy: Optional[str]


@dataclass(frozen=True)
class RequestSizeLimitConfig:
    enabled: bool
    max_body_bytes: int


def load_security_headers_config_from_env() -> SecurityHeadersConfig:
    return SecurityHeadersConfig(
        enabled=_get_bool("SECURITY_HEADERS_ENABLED", True),
        enable_hsts=_get_bool("SECURITY_HEADERS_ENABLE_HSTS", False),
        hsts_max_age=_get_int("SECURITY_HEADERS_HSTS_MAX_AGE", 31536000),
        hsts_include_subdomains=_get_bool("SECURITY_HEADERS_HSTS_INCLUDE_SUBDOMAINS", True),
        hsts_preload=_get_bool("SECURITY_HEADERS_HSTS_PRELOAD", False),
        referrer_policy=_get_str("SECURITY_HEADERS_REFERRER_POLICY", "no-referrer"),
        frame_options=_get_str("SECURITY_HEADERS_X_FRAME_OPTIONS", "DENY"),
        permissions_policy=_get_str(
            "SECURITY_HEADERS_PERMISSIONS_POLICY",
            "geolocation=(), microphone=(), camera=()",
        ),
        content_security_policy=os.environ.get("SECURITY_HEADERS_CSP"),
    )


def load_request_size_limit_config_from_env() -> RequestSizeLimitConfig:
    return RequestSizeLimitConfig(
        enabled=_get_bool("REQUEST_SIZE_LIMIT_ENABLED", True),
        max_body_bytes=_get_int("MAX_REQUEST_BODY_BYTES", 262144),
    )


class SecurityHeadersMiddleware:
    def __init__(self, app, config: SecurityHeadersConfig):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.config.enabled:
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {
                    (k.decode("latin1") if isinstance(k, bytes) else k).lower()
                    for k, _ in headers
                }

                def _add(name: str, value: str) -> None:
                    if name.lower() not in existing and value:
                        headers.append((name.encode("latin1"), value.encode("latin1")))

                _add("X-Content-Type-Options", "nosniff")
                _add("X-Frame-Options", self.config.frame_options)
                _add("Referrer-Policy", self.config.referrer_policy)
                _add("Permissions-Policy", self.config.permissions_policy)

                if self.config.content_security_policy:
                    _add("Content-Security-Policy", self.config.content_security_policy)

                if self.config.enable_hsts:
                    hsts = f"max-age={self.config.hsts_max_age}"
                    if self.config.hsts_include_subdomains:
                        hsts += "; includeSubDomains"
                    if self.config.hsts_preload:
                        hsts += "; preload"
                    _add("Strict-Transport-Security", hsts)

                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_wrapper)


class _RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app, config: RequestSizeLimitConfig):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or not self.config.enabled
            or self.config.max_body_bytes <= 0
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        headers = self._normalize_headers(scope.get("headers", []))
        content_length = headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > self.config.max_body_bytes:
                await self._send_413(send, self.config.max_body_bytes)
                return

        received = 0

        async def receive_limited():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"") or b""
                received += len(body)
                if received > self.config.max_body_bytes:
                    raise _RequestTooLarge()
            return message

        try:
            await self.app(scope, receive_limited, send)
        except _RequestTooLarge:
            await self._send_413(send, self.config.max_body_bytes)

    @staticmethod
    def _normalize_headers(headers) -> Dict[str, str]:
        return {
            (k.decode() if isinstance(k, bytes) else str(k)).lower():
            (v.decode() if isinstance(v, bytes) else str(v))
            for k, v in headers
        }

    @staticmethod
    async def _send_413(send, max_body_bytes: int) -> None:
        payload = {
            "error": "request_too_large",
            "max_body_bytes": max_body_bytes,
        }
        body = json.dumps(payload).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })

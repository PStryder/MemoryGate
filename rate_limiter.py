"""
Rate limiting utilities for MemoryGate.

Supports in-memory limits by default and optional Redis backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import redis.asyncio as redis_async  # type: ignore
except Exception:
    redis_async = None

logger = logging.getLogger("memorygate.rate_limit")


@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool
    global_ip: RateLimitRule
    api_key: RateLimitRule
    auth_ip: RateLimitRule
    max_cache_entries: int
    trusted_proxy_count: int
    trusted_proxy_ips: Tuple[str, ...]
    redis_fail_open: bool


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_epoch: int
    limit: int


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


def load_rate_limit_config_from_env() -> RateLimitConfig:
    """Load rate limiting configuration from environment variables."""
    enabled = _get_bool("RATE_LIMIT_ENABLED", True)
    global_ip_limit = _get_int("RATE_LIMIT_GLOBAL_PER_IP", 120)
    global_ip_window = _get_int("RATE_LIMIT_GLOBAL_WINDOW_SECONDS", 60)
    api_key_limit = _get_int("RATE_LIMIT_API_KEY_PER_KEY", 600)
    api_key_window = _get_int("RATE_LIMIT_API_KEY_WINDOW_SECONDS", 60)
    auth_ip_limit = _get_int("RATE_LIMIT_AUTH_PER_IP", 10)
    auth_ip_window = _get_int("RATE_LIMIT_AUTH_WINDOW_SECONDS", 60)
    max_cache_entries = _get_int("RATE_LIMIT_MAX_CACHE_ENTRIES", 10000)
    trusted_proxy_count = _get_int("RATE_LIMIT_TRUSTED_PROXY_COUNT", 0)
    redis_fail_open = _get_bool("RATE_LIMIT_REDIS_FAIL_OPEN", True)
    trusted_proxy_ips = tuple(
        ip.strip()
        for ip in os.environ.get("RATE_LIMIT_TRUSTED_PROXY_IPS", "").split(",")
        if ip.strip()
    )

    return RateLimitConfig(
        enabled=enabled,
        global_ip=RateLimitRule(limit=global_ip_limit, window_seconds=global_ip_window),
        api_key=RateLimitRule(limit=api_key_limit, window_seconds=api_key_window),
        auth_ip=RateLimitRule(limit=auth_ip_limit, window_seconds=auth_ip_window),
        max_cache_entries=max_cache_entries,
        trusted_proxy_count=trusted_proxy_count,
        trusted_proxy_ips=trusted_proxy_ips,
        redis_fail_open=redis_fail_open,
    )


class RateLimiter:
    """Abstract rate limiter interface."""
    async def allow(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class InMemoryRateLimiter(RateLimiter):
    """In-process rate limiter backed by an in-memory counter map."""
    def __init__(self, max_entries: int = 10000):
        self._lock = asyncio.Lock()
        self._counters: Dict[str, Tuple[int, float]] = {}
        self._max_entries = max_entries
        self._last_sweep = 0.0

    async def allow(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        now = time.time()
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        reset = window_start + rule.window_seconds

        async with self._lock:
            count, stored_reset = self._counters.get(key, (0, reset))
            if stored_reset <= now:
                count = 0
                stored_reset = reset

            count += 1
            self._counters[key] = (count, stored_reset)

            if (
                len(self._counters) > self._max_entries
                and (now - self._last_sweep) > rule.window_seconds
            ):
                self._sweep(now)
                self._last_sweep = now

        remaining = max(0, rule.limit - count)
        return RateLimitResult(
            allowed=count <= rule.limit,
            remaining=remaining,
            reset_epoch=int(stored_reset),
            limit=rule.limit,
        )

    def _sweep(self, now: float) -> None:
        expired = [key for key, (_, reset) in self._counters.items() if reset <= now]
        for key in expired:
            self._counters.pop(key, None)


class RedisRateLimiter(RateLimiter):
    """Redis-backed rate limiter with optional in-memory fallback."""
    def __init__(self, redis_client, key_prefix: str = "rl", fallback: RateLimiter | None = None):
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._fallback = fallback

    async def allow(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        now = time.time()
        window_seconds = rule.window_seconds
        window_id = int(now // window_seconds)
        reset = int((window_id + 1) * window_seconds)
        redis_key = f"{self._key_prefix}:{key}:{window_id}"
        try:
            pipeline = self._redis.pipeline()
            pipeline.incr(redis_key, 1)
            pipeline.expireat(redis_key, reset)
            count, _ = await pipeline.execute()
        except Exception:
            if self._fallback:
                logger.warning("Redis rate limiter unavailable; falling back to in-memory limiter")
                return await self._fallback.allow(key, rule)
            logger.warning("Redis rate limiter unavailable; failing closed")
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_epoch=int(now + window_seconds),
                limit=rule.limit,
            )

        count = int(count)
        remaining = max(0, rule.limit - count)
        return RateLimitResult(
            allowed=count <= rule.limit,
            remaining=remaining,
            reset_epoch=reset,
            limit=rule.limit,
        )

    async def close(self) -> None:
        if self._redis is None:
            return None
        try:
            result = self._redis.close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            return None
        if self._fallback:
            await self._fallback.close()


def build_rate_limiter_from_env(config: RateLimitConfig) -> RateLimiter:
    """Create a rate limiter using Redis when configured, else in-memory."""
    redis_url = os.environ.get("RATE_LIMIT_REDIS_URL") or os.environ.get("REDIS_URL")
    if redis_url and redis_async is not None:
        client = redis_async.from_url(redis_url)
        logger.info("Rate limiting using Redis backend")
        fallback = None
        if config.redis_fail_open:
            fallback = InMemoryRateLimiter(max_entries=config.max_cache_entries)
        return RedisRateLimiter(client, fallback=fallback)
    if redis_url and redis_async is None:
        logger.warning(
            "RATE_LIMIT_REDIS_URL set but redis package not installed; using in-memory limiter"
        )
    return InMemoryRateLimiter(max_entries=config.max_cache_entries)


class RateLimitMiddleware:
    """ASGI middleware that enforces rate limit rules per request."""
    AUTH_PATH_PREFIXES = ("/auth", "/oauth", "/.well-known", "/mcp/.well-known")

    def __init__(self, app, limiter: RateLimiter, config: RateLimitConfig):
        self.app = app
        self.limiter = limiter
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self.config.enabled or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        headers = self._normalize_headers(scope.get("headers", []))
        path = scope.get("path", "")
        client_ip = self._get_client_ip(scope, headers)
        api_key_prefix = self._extract_api_key_prefix(headers)

        result = await self.limiter.allow(f"ip:{client_ip}", self.config.global_ip)
        if not result.allowed:
            await self._send_429(send, result, "global_ip")
            return

        if self._is_auth_path(path):
            result = await self.limiter.allow(f"auth-ip:{client_ip}", self.config.auth_ip)
            if not result.allowed:
                await self._send_429(send, result, "auth_ip")
                return

        if api_key_prefix:
            result = await self.limiter.allow(f"key:{api_key_prefix}", self.config.api_key)
            if not result.allowed:
                await self._send_429(send, result, "api_key")
                return

        await self.app(scope, receive, send)

    @staticmethod
    def _normalize_headers(headers) -> Dict[str, str]:
        return {
            (k.decode() if isinstance(k, bytes) else str(k)).lower():
            (v.decode() if isinstance(v, bytes) else str(v))
            for k, v in headers
        }

    def _get_client_ip(self, scope, headers: Dict[str, str]) -> str:
        client = scope.get("client")
        client_ip = client[0] if client else ""

        forwarded_for = headers.get("x-forwarded-for")
        if forwarded_for and self._is_trusted_proxy(client_ip):
            forwarded_ip = self._select_forwarded_ip(forwarded_for)
            if forwarded_ip:
                return forwarded_ip

        real_ip = headers.get("x-real-ip")
        if real_ip and self._is_trusted_proxy(client_ip):
            return real_ip.strip()

        if client_ip:
            return client_ip
        return "unknown"

    def _is_trusted_proxy(self, client_ip: str) -> bool:
        if self.config.trusted_proxy_count > 0:
            return True
        if self.config.trusted_proxy_ips and client_ip:
            return client_ip in self.config.trusted_proxy_ips
        return False

    def _select_forwarded_ip(self, forwarded_for: str) -> Optional[str]:
        ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if not ips:
            return None
        if self.config.trusted_proxy_count > 0:
            index = max(0, len(ips) - self.config.trusted_proxy_count - 1)
            return ips[index]
        return ips[0]

    @staticmethod
    def _extract_api_key_prefix(headers: Dict[str, str]) -> Optional[str]:
        auth_header = headers.get("authorization", "")
        api_key = ""
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header.split(" ", 1)[1].strip()
        elif "x-api-key" in headers:
            api_key = headers.get("x-api-key", "").strip()

        if api_key.startswith("mg_") and len(api_key) >= 11:
            return api_key[:11]
        return None

    def _is_auth_path(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.AUTH_PATH_PREFIXES)

    async def _send_429(self, send, result: RateLimitResult, limit_type: str) -> None:
        retry_after = max(0, result.reset_epoch - int(time.time()))
        payload = {
            "error": "rate_limit_exceeded",
            "limit_type": limit_type,
            "limit": result.limit,
            "remaining": result.remaining,
            "reset_epoch": result.reset_epoch,
            "retry_after_seconds": retry_after,
        }
        body = json.dumps(payload).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(retry_after).encode()),
                (b"x-ratelimit-limit", str(result.limit).encode()),
                (b"x-ratelimit-remaining", str(result.remaining).encode()),
                (b"x-ratelimit-reset", str(result.reset_epoch).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })

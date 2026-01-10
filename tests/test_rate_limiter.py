from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rate_limiter import (
    InMemoryRateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    RateLimitRule,
)


def build_app(config: RateLimitConfig) -> Starlette:
    limiter = InMemoryRateLimiter(max_entries=config.max_cache_entries)

    def ping(request):
        return JSONResponse({"ok": True})

    def auth_client(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[
        Route("/ping", ping, methods=["GET"]),
        Route("/auth/client", auth_client, methods=["POST"]),
    ])
    app.add_middleware(RateLimitMiddleware, limiter=limiter, config=config)

    return app


def test_global_ip_limit_blocks_after_limit():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=2, window_seconds=60),
        api_key=RateLimitRule(limit=100, window_seconds=60),
        auth_ip=RateLimitRule(limit=100, window_seconds=60),
        max_cache_entries=100,
        trusted_proxy_count=0,
        trusted_proxy_ips=(),
        redis_fail_open=True,
    )
    app = build_app(config)
    client = TestClient(app)

    headers = {"X-Forwarded-For": "203.0.113.10"}
    assert client.get("/ping", headers=headers).status_code == 200
    assert client.get("/ping", headers=headers).status_code == 200

    resp = client.get("/ping", headers=headers)
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limit_exceeded"


def test_api_key_limit_isolated_by_key():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=100, window_seconds=60),
        api_key=RateLimitRule(limit=1, window_seconds=60),
        auth_ip=RateLimitRule(limit=100, window_seconds=60),
        max_cache_entries=100,
        trusted_proxy_count=0,
        trusted_proxy_ips=(),
        redis_fail_open=True,
    )
    app = build_app(config)
    client = TestClient(app)

    headers_key_a = {"Authorization": "Bearer mg_aaaaaaaaaaaa"}
    headers_key_b = {"Authorization": "Bearer mg_bbbbbbbbbbbb"}

    assert client.get("/ping", headers=headers_key_a).status_code == 200
    assert client.get("/ping", headers=headers_key_a).status_code == 429
    assert client.get("/ping", headers=headers_key_b).status_code == 200


def test_auth_path_has_stricter_limit():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=100, window_seconds=60),
        api_key=RateLimitRule(limit=100, window_seconds=60),
        auth_ip=RateLimitRule(limit=1, window_seconds=60),
        max_cache_entries=100,
        trusted_proxy_count=0,
        trusted_proxy_ips=(),
        redis_fail_open=True,
    )
    app = build_app(config)
    client = TestClient(app)

    headers = {"X-Forwarded-For": "203.0.113.11"}
    assert client.post("/auth/client", headers=headers).status_code == 200
    assert client.post("/auth/client", headers=headers).status_code == 429
    assert client.get("/ping", headers=headers).status_code == 200


def test_untrusted_proxy_ignores_forwarded_for():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=1, window_seconds=60),
        api_key=RateLimitRule(limit=100, window_seconds=60),
        auth_ip=RateLimitRule(limit=100, window_seconds=60),
        max_cache_entries=100,
        trusted_proxy_count=0,
        trusted_proxy_ips=(),
        redis_fail_open=True,
    )
    app = build_app(config)
    client = TestClient(app)

    headers_a = {"X-Forwarded-For": "203.0.113.10"}
    headers_b = {"X-Forwarded-For": "203.0.113.11"}

    assert client.get("/ping", headers=headers_a).status_code == 200
    assert client.get("/ping", headers=headers_b).status_code == 429


def test_trusted_proxy_uses_forwarded_for():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=1, window_seconds=60),
        api_key=RateLimitRule(limit=100, window_seconds=60),
        auth_ip=RateLimitRule(limit=100, window_seconds=60),
        max_cache_entries=100,
        trusted_proxy_count=1,
        trusted_proxy_ips=(),
        redis_fail_open=True,
    )
    app = build_app(config)
    client = TestClient(app)

    headers_a = {"X-Forwarded-For": "203.0.113.10, 10.0.0.1"}
    headers_b = {"X-Forwarded-For": "203.0.113.11, 10.0.0.1"}

    assert client.get("/ping", headers=headers_a).status_code == 200
    assert client.get("/ping", headers=headers_b).status_code == 200
    assert client.get("/ping", headers=headers_a).status_code == 429

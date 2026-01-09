from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_limiter import (
    InMemoryRateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    RateLimitRule,
)


def build_app(config: RateLimitConfig) -> FastAPI:
    limiter = InMemoryRateLimiter(max_entries=config.max_cache_entries)
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limiter=limiter, config=config)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.post("/auth/client")
    def auth_client():
        return {"ok": True}

    return app


def test_global_ip_limit_blocks_after_limit():
    config = RateLimitConfig(
        enabled=True,
        global_ip=RateLimitRule(limit=2, window_seconds=60),
        api_key=RateLimitRule(limit=100, window_seconds=60),
        auth_ip=RateLimitRule(limit=100, window_seconds=60),
        max_cache_entries=100,
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
    )
    app = build_app(config)
    client = TestClient(app)

    headers = {"X-Forwarded-For": "203.0.113.11"}
    assert client.post("/auth/client", headers=headers).status_code == 200
    assert client.post("/auth/client", headers=headers).status_code == 429
    assert client.get("/ping", headers=headers).status_code == 200

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from security_middleware import (
    RequestSizeLimitConfig,
    RequestSizeLimitMiddleware,
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)


def test_security_headers_added():
    config = SecurityHeadersConfig(
        enabled=True,
        enable_hsts=True,
        hsts_max_age=60,
        hsts_include_subdomains=True,
        hsts_preload=True,
        referrer_policy="no-referrer",
        frame_options="DENY",
        permissions_policy="geolocation=()",
        content_security_policy="default-src 'self'",
    )

    def ping(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/ping", ping, methods=["GET"])])
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get("/ping")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "geolocation=()"
    assert response.headers["content-security-policy"] == "default-src 'self'"
    assert "strict-transport-security" in response.headers


def test_request_size_limit_blocks_large_body():
    config = RequestSizeLimitConfig(enabled=True, max_body_bytes=8)
    async def echo(request: Request):
        body = await request.body()
        return JSONResponse({"length": len(body)})

    app = Starlette(routes=[Route("/echo", echo, methods=["POST"])])
    app.add_middleware(RequestSizeLimitMiddleware, config=config)

    client = TestClient(app)
    response = client.post("/echo", data="x" * 16)

    assert response.status_code == 413
    assert response.json()["error"] == "request_too_large"


def test_request_size_limit_allows_small_body():
    config = RequestSizeLimitConfig(enabled=True, max_body_bytes=16)
    async def echo(request: Request):
        body = await request.body()
        return JSONResponse({"length": len(body)})

    app = Starlette(routes=[Route("/echo", echo, methods=["POST"])])
    app.add_middleware(RequestSizeLimitMiddleware, config=config)

    client = TestClient(app)
    response = client.post("/echo", data="ok")

    assert response.status_code == 200
    assert response.json()["length"] == 2

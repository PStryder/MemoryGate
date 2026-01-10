import os
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("GOOGLE_CLIENT_ID", "test-google")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-google-secret")
os.environ.setdefault("OAUTH_REDIRECT_BASE", "http://localhost:8080")
os.environ.setdefault("FRONTEND_URL", "http://localhost:3000")

from oauth import OAuthUserInfo  # noqa: E402
from oauth_models import OAuthState, User, UserSession  # noqa: E402
import server  # noqa: E402
import oauth_routes  # noqa: E402


class StubProvider:
    def generate_pkce_pair(self):
        return "verifier", "challenge"

    def get_authorization_url(self, state: str, redirect_uri: str, code_challenge: str | None = None) -> str:
        return f"https://example.com/auth?state={state}"

    def exchange_code(self, code: str, redirect_uri: str, code_verifier: str | None = None):
        return {"access_token": "access-token"}

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        return OAuthUserInfo(
            subject="user-123",
            email="user@example.com",
            name="Test User",
            avatar_url=None,
            email_verified=True,
            raw_data={"id": "user-123"},
        )


def test_oauth_login_creates_state_and_redirect(monkeypatch, db_engine):
    """Test OAuth login flow creates state and redirects correctly."""
    from sqlalchemy.orm import sessionmaker
    
    SessionLocal = sessionmaker(bind=db_engine)
    server.DB.SessionLocal = SessionLocal

    monkeypatch.setattr(
        oauth_routes.OAuthProviderFactory,
        "create_provider",
        lambda *_args, **_kwargs: StubProvider(),
    )

    app = FastAPI()
    app.include_router(oauth_routes.router)
    client = TestClient(app)

    response = client.get("/auth/login/google")
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://example.com/auth")

    db = SessionLocal()
    try:
        state = db.query(OAuthState).first()
        assert state is not None
        assert state.provider == "google"
        assert state.redirect_uri.endswith("/auth/callback/google")
    finally:
        db.close()


def test_oauth_callback_happy_path(monkeypatch, db_engine):
    """Test OAuth callback successfully creates user and session."""
    from sqlalchemy.orm import sessionmaker
    
    SessionLocal = sessionmaker(bind=db_engine)
    server.DB.SessionLocal = SessionLocal

    # Setup test data
    db = SessionLocal()
    try:
        oauth_state = OAuthState(
            provider="google",
            redirect_uri="http://localhost:8080/auth/callback/google",
            code_verifier="verifier",
        )
        oauth_state.state = "state-123"
        oauth_state.expires_at = datetime.utcnow() + timedelta(minutes=5)
        db.add(oauth_state)
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(
        oauth_routes.OAuthProviderFactory,
        "create_provider",
        lambda *_args, **_kwargs: StubProvider(),
    )

    app = FastAPI()
    app.include_router(oauth_routes.router)
    client = TestClient(app)

    response = client.get("/auth/callback/google", params={"code": "abc", "state": "state-123"})
    assert response.status_code == 302
    assert response.headers["location"].endswith("/dashboard")
    assert "mg_session=" in response.headers.get("set-cookie", "")

    db = SessionLocal()
    try:
        assert db.query(User).count() == 1
        assert db.query(UserSession).count() == 1
        assert db.query(OAuthState).count() == 0
    finally:
        db.close()


def test_oauth_callback_rejects_expired_state(monkeypatch, db_engine):
    """Test OAuth callback rejects expired state tokens."""
    from sqlalchemy.orm import sessionmaker
    
    SessionLocal = sessionmaker(bind=db_engine)
    server.DB.SessionLocal = SessionLocal

    # Setup expired state
    db = SessionLocal()
    try:
        oauth_state = OAuthState(
            provider="google",
            redirect_uri="http://localhost:8080/auth/callback/google",
            code_verifier="verifier",
        )
        oauth_state.state = "expired-state"
        oauth_state.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db.add(oauth_state)
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(
        oauth_routes.OAuthProviderFactory,
        "create_provider",
        lambda *_args, **_kwargs: StubProvider(),
    )

    app = FastAPI()
    app.include_router(oauth_routes.router)
    client = TestClient(app)

    response = client.get(
        "/auth/callback/google",
        params={"code": "abc", "state": "expired-state"},
    )
    assert response.status_code == 400

import asyncio
import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")

from oauth_models import User, APIKey  # noqa: E402
from auth_middleware import hash_api_key, verify_request_api_key  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def _create_user(db) -> User:
    user = User(
        email="user@example.com",
        name="Test User",
        oauth_provider="client_credentials",
        oauth_subject="test-client",
        is_verified=True,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_verify_request_api_key_handles_prefix_collision(db_session):
    """Test that API key verification correctly handles multiple keys with same prefix."""
    db = db_session
    user = _create_user(db)
    prefix = "mg_testpref"
    key_a = prefix + "A" * 20
    key_b = prefix + "B" * 20

    db.add(APIKey(
        user_id=user.id,
        key_prefix=prefix,
        key_hash=hash_api_key(key_a),
        name="Key A",
        expires_at=None,
    ))
    db.add(APIKey(
        user_id=user.id,
        key_prefix=prefix,
        key_hash=hash_api_key(key_b),
        name="Key B",
        expires_at=None,
    ))
    db.commit()

    headers = {"Authorization": f"Bearer {key_b}"}
    authed_user = verify_request_api_key(db, headers)
    assert authed_user is not None
    assert authed_user.id == user.id

    headers_bad = {"Authorization": "Bearer mg_testprefBADKEY"}
    assert verify_request_api_key(db, headers_bad) is None


def test_revoke_api_key_invalid_id_returns_400(db_session):
    """Test that revoking API key with invalid ID returns 400."""
    # Lazy import to avoid circular dependency
    from oauth_routes import revoke_api_key
    
    db = db_session
    user = _create_user(db)

    try:
        asyncio.run(revoke_api_key("not-a-uuid", user=user, db=db))
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        assert False, "Expected HTTPException for invalid key id"

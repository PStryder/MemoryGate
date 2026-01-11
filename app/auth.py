"""
Authentication Middleware and Utilities for MemoryGate

Provides authentication via:
1. Session tokens (cookies)
2. API keys (headers)
"""

from typing import Optional, Annotated
from datetime import datetime
import bcrypt

from fastapi import HTTPException, Header, Cookie, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from oauth_models import User, UserSession, APIKey


# FastAPI security scheme
bearer_scheme = HTTPBearer(auto_error=False)


def hash_api_key(api_key: str) -> str:
    """Hash API key with bcrypt"""
    return bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """Verify API key against hash"""
    return bcrypt.checkpw(api_key.encode(), key_hash.encode())


def generate_api_key() -> tuple[str, str, str]:
    """Generate API key with prefix and hash
    
    Returns:
        tuple: (full_key, prefix, hash)
    """
    import secrets
    full_key = f"mg_{secrets.token_urlsafe(32)}"
    prefix = full_key[:11]  # "mg_" + first 8 chars
    key_hash = hash_api_key(full_key)
    return full_key, prefix, key_hash


def verify_request_api_key(db: Session, headers: dict) -> Optional[User]:
    """
    Verify API key from request headers (sync, no FastAPI dependencies).
    
    Checks Authorization: Bearer or X-API-Key header.
    Returns User if valid, None if missing/invalid.
    """
    # Extract API key from headers
    api_key = None
    
    # Check Authorization header (case-insensitive)
    auth_header = None
    for key, value in headers.items():
        if key.lower() == 'authorization':
            auth_header = value
            break
    
    if auth_header:
        parts = auth_header.split(' ', 1)
        if len(parts) == 2 and parts[0].lower() == 'bearer':
            api_key = parts[1]
    
    # Fallback to X-API-Key header (case-insensitive)
    if not api_key:
        for key, value in headers.items():
            if key.lower() == 'x-api-key':
                api_key = value
                break
    
    if not api_key:
        return None
    
    # Validate prefix - reject non-MemoryGate tokens early (e.g., JWT tokens from other systems)
    if not api_key.startswith("mg_"):
        return None
    
    key_prefix = api_key[:11]
    
    # Find keys by prefix (prefix collisions are possible)
    api_keys = db.query(APIKey).filter(
        APIKey.key_prefix == key_prefix
    ).all()

    if not api_keys:
        return None

    # Verify full key against hash
    for api_key_obj in api_keys:
        if not verify_api_key(api_key, api_key_obj.key_hash):
            continue
        if not api_key_obj.is_valid:
            return None
        user = api_key_obj.user
        if not user or not user.is_active:
            return None

        # Update usage tracking
        api_key_obj.increment_usage()
        db.commit()

        return user

    return None



async def get_current_user_from_session(
    db,
    session_token: Annotated[Optional[str], Cookie(alias="mg_session")] = None
) -> Optional[User]:
    """Get user from session token (cookie)"""
    if not session_token:
        return None
    
    user_session = db.query(UserSession).filter(
        UserSession.token == session_token
    ).first()
    
    if not user_session or not user_session.is_valid:
        return None
    
    # Refresh activity
    user_session.refresh_activity()
    db.commit()
    
    return user_session.user


async def get_current_user_from_api_key(
    db,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Annotated[Optional[str], Header()] = None
) -> Optional[User]:
    """Get user from API key (Authorization header or X-API-Key header)"""
    
    # Try Bearer token first
    api_key = None
    if credentials and credentials.scheme.lower() == "bearer":
        api_key = credentials.credentials
    # Fallback to X-API-Key header
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        return None
    
    # Extract prefix for faster lookup
    if not api_key.startswith("mg_"):
        return None
    
    key_prefix = api_key[:11]
    
    # Find keys by prefix (prefix collisions are possible)
    api_keys = db.query(APIKey).filter(
        APIKey.key_prefix == key_prefix
    ).all()

    if not api_keys:
        return None

    # Verify full key against hash
    for api_key_obj in api_keys:
        if not verify_api_key(api_key, api_key_obj.key_hash):
            continue
        if not api_key_obj.is_valid:
            return None
        user = api_key_obj.user
        if not user or not user.is_active:
            return None

        # Update usage tracking
        api_key_obj.increment_usage()
        db.commit()

        return user

    return None


async def get_current_user(
    db,
    user_from_session: Annotated[Optional[User], Depends(get_current_user_from_session)],
    user_from_api_key: Annotated[Optional[User], Depends(get_current_user_from_api_key)]
) -> Optional[User]:
    """Get current user from either session or API key"""
    return user_from_session or user_from_api_key


async def require_auth(
    user: Annotated[Optional[User], Depends(get_current_user)]
) -> User:
    """Require authentication - raises 401 if not authenticated"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user

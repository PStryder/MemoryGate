"""
OAuth Routes for MemoryGate

Handles OAuth 2.0 authentication flow
"""

import os
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from oauth import OAuthProviderFactory
from oauth_models import User, UserSession, OAuthState, APIKey
from auth_middleware import get_current_user, require_auth


# Router
router = APIRouter(prefix="/auth", tags=["authentication"])


# Configuration from environment
OAUTH_REDIRECT_BASE = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:8000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")


# OAuth provider configs (from environment)
OAUTH_PROVIDERS = {
    "google": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
    },
    "github": {
        "client_id": os.environ.get("GITHUB_CLIENT_ID"),
        "client_secret": os.environ.get("GITHUB_CLIENT_SECRET"),
    }
}


def get_db_session():
    """Database dependency"""
    import server  # Import module, not the name
    if server.DB.SessionLocal is None:
        raise RuntimeError("Database not initialized - SessionLocal is None")
    db = server.DB.SessionLocal()  # Access via DB class
    try:
        yield db
    finally:
        db.close()


@router.get("/login/{provider}")
async def oauth_login(
    provider: str,
    request: Request,
    db = Depends(get_db_session)
):
    """Initiate OAuth flow for a provider"""
    
    # Validate provider
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}"
        )
    
    config = OAUTH_PROVIDERS[provider]
    if not config["client_id"] or not config["client_secret"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OAuth provider {provider} not configured"
        )
    
    # Create OAuth provider
    oauth_provider = OAuthProviderFactory.create_provider(
        provider,
        config["client_id"],
        config["client_secret"]
    )
    
    # Generate PKCE pair
    code_verifier, code_challenge = oauth_provider.generate_pkce_pair()
    
    # Create state with PKCE verifier
    redirect_uri = f"{OAUTH_REDIRECT_BASE}/auth/callback/{provider}"
    oauth_state = OAuthState(
        provider=provider,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier
    )
    
    db.add(oauth_state)
    db.commit()
    
    # Generate authorization URL
    auth_url = oauth_provider.get_authorization_url(
        state=oauth_state.state,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge
    )
    
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    request: Request,
    response: Response,
    db = Depends(get_db_session)
):
    """Handle OAuth callback from provider"""
    
    # Verify state exists and is valid
    oauth_state = db.query(OAuthState).filter(
        OAuthState.state == state
    ).first()
    
    if not oauth_state or oauth_state.is_expired:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state"
        )
    
    if oauth_state.provider != provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="State provider mismatch"
        )
    
    # Exchange code for token (keep state alive for retry if this fails)
    config = OAUTH_PROVIDERS[provider]
    oauth_provider = OAuthProviderFactory.create_provider(
        provider,
        config["client_id"],
        config["client_secret"]
    )
    
    try:
        token_response = oauth_provider.exchange_code(
            code=code,
            redirect_uri=oauth_state.redirect_uri,
            code_verifier=oauth_state.code_verifier
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange OAuth code: {str(e)}"
        )
    
    # Get user info from provider
    access_token = token_response.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No access token in OAuth response"
        )
    
    try:
        user_info = oauth_provider.get_user_info(access_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch user info: {str(e)}"
        )
    
    # Find or create user
    user = db.query(User).filter(
        User.oauth_provider == provider,
        User.oauth_subject == user_info.subject
    ).first()
    
    if user:
        # Update existing user
        user.name = user_info.name
        user.avatar_url = user_info.avatar_url
        user.is_verified = user_info.email_verified
        user.last_login = datetime.utcnow()
    else:
        # Create new user
        user = User(
            email=user_info.email,
            name=user_info.name,
            avatar_url=user_info.avatar_url,
            oauth_provider=provider,
            oauth_subject=user_info.subject,
            is_verified=user_info.email_verified,
            metadata_=user_info.raw_data
        )
        db.add(user)
    
    db.commit()
    
    # Create session
    user_session = UserSession(
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    db.add(user_session)
    db.commit()
    
    # Delete OAuth state after successful authentication (one-time use)
    db.delete(oauth_state)
    db.commit()
    
    # Set session cookie
    response = RedirectResponse(
        url=f"{FRONTEND_URL}/dashboard",
        status_code=status.HTTP_302_FOUND
    )
    response.set_cookie(
        key="mg_session",
        value=user_session.token,
        httponly=True,
        secure=True,  # HTTPS only in production
        samesite="lax",  # Use "none" + secure=True for cross-origin auth flows
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    return response


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    user: User = Depends(require_auth),
    db = Depends(get_db_session)
):
    """Logout user by revoking session"""
    
    # Get session token from cookie
    session_token = request.cookies.get("mg_session")
    if session_token:
        user_session = db.query(UserSession).filter(
            UserSession.token == session_token
        ).first()
        
        if user_session:
            user_session.is_revoked = True
            db.commit()
    
    # Clear cookie
    response.delete_cookie("mg_session")
    
    return {"message": "Logged out successfully"}



class ClientCredentialsRequest(BaseModel):
    client_id: str
    client_secret: str


@router.post("/client")
async def client_credentials_auth(
    request: ClientCredentialsRequest,
    db = Depends(get_db_session)
):
    """
    Client credentials authentication for MCP tools.
    
    Validates client_id/client_secret against environment variables,
    gets or creates a user for the client, and returns an API key.
    """
    # Validate credentials against environment
    expected_client_id = os.environ.get("PSTRYDER_DESKTOP")
    expected_client_secret = os.environ.get("PSTRYDER_DESKTOP_SECRET")
    
    if not expected_client_id or not expected_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client credentials authentication not configured"
        )
    
    if request.client_id != expected_client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )
    
    if request.client_secret != expected_client_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )
    
    # Get or create user for this client
    client_email = f"{request.client_id}@client.memorygate.internal"
    user = db.query(User).filter(
        User.email == client_email
    ).first()
    
    if not user:
        user = User(
            email=client_email,
            name=request.client_id,
            oauth_provider="client_credentials",
            oauth_subject=request.client_id,
            is_verified=True,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Generate API key
    from auth_middleware import generate_api_key
    
    full_key, prefix, key_hash = generate_api_key()
    
    api_key = APIKey(
        user_id=user.id,
        key_prefix=prefix,
        key_hash=key_hash,
        name=f"{request.client_id} - Auto-generated",
        expires_at=None  # Never expires
    )
    
    db.add(api_key)
    db.commit()
    
    return {
        "api_key": full_key,
        "key_prefix": prefix,
        "user_id": str(user.id),
        "expires_at": None
    }


@router.get("/me")
async def get_current_user_info(user: User = Depends(require_auth)):
    """Get current authenticated user information"""
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "oauth_provider": user.oauth_provider,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat()
    }


class CreateAPIKeyRequest(BaseModel):
    name: str
    expires_in_days: Optional[int] = None  # None = never expires


@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(require_auth),
    db = Depends(get_db_session)
):
    """Create a new API key for the authenticated user"""
    from auth_middleware import generate_api_key
    from datetime import timedelta
    
    full_key, prefix, key_hash = generate_api_key()
    
    api_key = APIKey(
        user_id=user.id,
        key_prefix=prefix,
        key_hash=key_hash,
        name=request.name,
        expires_at=(
            datetime.utcnow() + timedelta(days=request.expires_in_days)
            if request.expires_in_days else None
        )
    )
    
    db.add(api_key)
    db.commit()
    
    return {
        "api_key": full_key,  # Only time it's shown!
        "key_id": str(api_key.id),
        "key_prefix": prefix,
        "name": api_key.name,
        "created_at": api_key.created_at.isoformat(),
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
    }



@router.get("/api-keys")
async def list_api_keys(
    user: User = Depends(require_auth),
    db = Depends(get_db_session)
):
    """List all API keys for the authenticated user"""
    api_keys = db.query(APIKey).filter(APIKey.user_id == user.id).all()
    
    return {
        "api_keys": [
            {
                "key_id": str(key.id),
                "key_prefix": key.key_prefix,
                "name": key.name,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "is_revoked": key.is_revoked
            }
            for key in api_keys
        ]
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_auth),
    db = Depends(get_db_session)
):
    """Revoke an API key"""
    from uuid import UUID
    
    api_key = db.query(APIKey).filter(
        APIKey.id == UUID(key_id),
        APIKey.user_id == user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    api_key.is_revoked = True
    db.commit()
    
    return {"message": "API key revoked successfully"}

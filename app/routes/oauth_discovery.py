"""
OAuth 2.0 Discovery and Authorization Flow for MCP Clients

Implements standard OAuth discovery endpoints that Claude Desktop expects,
using PKCE (Proof Key for Code Exchange) for security.

Flow:
1. Client discovers endpoints via /.well-known/oauth-authorization-server
2. Client redirects user to /oauth/authorize with PKCE challenge
3. User enters client_secret (admin passphrase)
4. Server creates auth code, redirects back to client
5. Client exchanges code + PKCE verifier for API key
6. API key used for all subsequent MCP requests
"""

import os
import secrets
import hashlib
import base64
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from string import Template

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from oauth_models import APIKey, User, OAuthAuthorizationCode
from app.auth import hash_api_key, generate_api_key


# =============================================================================
# Configuration
# =============================================================================

OAUTH_ISSUER = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:8000")
OAUTH_CLIENT_ID = os.environ.get("PSTRYDER_DESKTOP", "")
OAUTH_CLIENT_SECRET = os.environ.get("PSTRYDER_DESKTOP_SECRET", "")
OAUTH_ALLOWED_REDIRECT_URIS = [
    uri.strip()
    for uri in os.environ.get("PSTRYDER_DESKTOP_REDIRECT_URIS", "").split(",")
    if uri.strip()
]

AUTH_CODE_TTL_SECONDS = 60  # Auth codes expire in 60 seconds

# =============================================================================
# Database-backed auth codes (multi-instance safe)
# =============================================================================


# =============================================================================
# Helpers
# =============================================================================

def _b64url(data: bytes) -> str:
    """URL-safe base64 encoding without padding"""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_s256_challenge(verifier: str) -> str:
    """Generate PKCE S256 challenge from verifier"""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _b64url(digest)


def _get_db_session() -> Session:
    from core.db import DB
    if DB.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return DB.SessionLocal()


def clean_expired_auth_codes(db: Session, commit: bool = True) -> None:
    """Remove expired authorization codes"""
    cutoff = datetime.utcnow()
    db.query(OAuthAuthorizationCode).filter(
        OAuthAuthorizationCode.expires_at <= cutoff
    ).delete()
    if commit:
        db.commit()


def require_oauth_env() -> None:
    """Verify OAuth environment variables are set"""
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="OAuth not configured - missing PSTRYDER_DESKTOP or PSTRYDER_DESKTOP_SECRET"
        )


def validate_authorize_request(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    scope: Optional[str],
    state: Optional[str],
    code_challenge: str,
    code_challenge_method: str,
) -> Dict[str, Any]:
    """Validate OAuth authorization request parameters"""
    require_oauth_env()
    
    if response_type != "code":
        raise HTTPException(status_code=400, detail="response_type must be 'code'")
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Unknown client_id")
    if not redirect_uri:
        raise HTTPException(status_code=400, detail="Missing redirect_uri")
    if not OAUTH_ALLOWED_REDIRECT_URIS:
        raise HTTPException(
            status_code=500,
            detail="OAuth redirect URIs not configured"
        )
    if redirect_uri not in OAUTH_ALLOWED_REDIRECT_URIS:
        raise HTTPException(status_code=400, detail="redirect_uri not allowed")
    if not code_challenge:
        raise HTTPException(status_code=400, detail="Missing code_challenge (PKCE required)")
    if code_challenge_method != "S256":
        raise HTTPException(status_code=400, detail="code_challenge_method must be S256")
    
    scope_norm = scope or "memorygate:full"
    
    return {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope_norm,
        "state": state or "",
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }


# =============================================================================
# Authorization Form Template
# =============================================================================

AUTHORIZE_FORM = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MemoryGate Authorization</title>
  <style>
    body { 
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; 
      margin: 3rem; 
      background: #f5f5f5;
    }
    .card { 
      max-width: 560px; 
      margin: 0 auto;
      padding: 2rem; 
      background: white;
      border-radius: 16px; 
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    h2 { margin-top: 0; color: #333; }
    input { 
      width: 100%; 
      padding: 0.85rem; 
      margin-top: 0.5rem; 
      border-radius: 8px; 
      border: 1px solid #ccc; 
      font-size: 1rem; 
      box-sizing: border-box;
    }
    button { 
      width: 100%;
      margin-top: 1rem; 
      padding: 0.85rem; 
      border-radius: 8px; 
      border: none; 
      background: #5469d4;
      color: white;
      cursor: pointer; 
      font-size: 1rem;
      font-weight: 500;
    }
    button:hover { background: #4355c4; }
    .muted { 
      color: #666; 
      font-size: 0.9rem; 
      line-height: 1.5; 
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid #eee;
    }
    code { 
      background: #f5f5f5; 
      padding: 0.2rem 0.4rem; 
      border-radius: 4px;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>🔐 Authorize MemoryGate</h2>
    <p>Enter your client secret to authorize access.</p>
    <form method="post" action="/oauth/authorize">
      <input type="hidden" name="response_type" value="$response_type">
      <input type="hidden" name="client_id" value="$client_id">
      <input type="hidden" name="redirect_uri" value="$redirect_uri">
      <input type="hidden" name="scope" value="$scope">
      <input type="hidden" name="state" value="$state">
      <input type="hidden" name="code_challenge" value="$code_challenge">
      <input type="hidden" name="code_challenge_method" value="$code_challenge_method">
      <label for="secret">Client Secret</label>
      <input type="password" id="secret" name="client_secret" placeholder="Enter your client secret" autofocus>
      <button type="submit">Authorize Access</button>
    </form>
    <div class="muted">
      <strong>Client ID:</strong> <code>$client_id</code><br/>
      <strong>Requested Scope:</strong> <code>$scope</code>
    </div>
  </div>
</body>
</html>
""")


# =============================================================================
# Router
# =============================================================================

router = APIRouter()


# =============================================================================
# Discovery Endpoints
# =============================================================================

@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_metadata():
    """OAuth Authorization Server Metadata (RFC 8414)"""
    require_oauth_env()
    base = OAUTH_ISSUER.rstrip("/")
    return {
        "issuer": OAUTH_ISSUER,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        "scopes_supported": ["memorygate:full"],
    }


@router.get("/.well-known/openid-configuration")
async def openid_configuration():
    """OpenID Connect Discovery (for compatibility)"""
    require_oauth_env()
    base = OAUTH_ISSUER.rstrip("/")
    return {
        "issuer": OAUTH_ISSUER,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "subject_types_supported": ["public"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["memorygate:full"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }


@router.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource_metadata():
    """OAuth Protected Resource Metadata (RFC 8707)"""
    require_oauth_env()
    base = OAUTH_ISSUER.rstrip("/")
    return {
        "resource": f"{base}/mcp",
        "authorization_servers": [OAUTH_ISSUER],
        "scopes_supported": ["memorygate:full"],
    }


# Claude Desktop also probes relative to MCP base path
@router.get("/mcp/.well-known/oauth-authorization-server")
async def oauth_authorization_server_metadata_mcp():
    return await oauth_authorization_server_metadata()


@router.get("/mcp/.well-known/openid-configuration")
async def openid_configuration_mcp():
    return await openid_configuration()


@router.get("/mcp/.well-known/oauth-protected-resource")
async def oauth_protected_resource_metadata_mcp():
    return await oauth_protected_resource_metadata()


# =============================================================================
# OAuth Authorization Flow
# =============================================================================

@router.get("/oauth/authorize", response_class=HTMLResponse)
async def oauth_authorize_get(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    scope: Optional[str] = None,
    state: Optional[str] = None,
    code_challenge: str = "",
    code_challenge_method: str = "S256",
):
    """Show authorization form"""
    meta = validate_authorize_request(
        response_type=response_type,
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )
    
    html = AUTHORIZE_FORM.safe_substitute(
        response_type=response_type,
        client_id=meta["client_id"],
        redirect_uri=meta["redirect_uri"],
        scope=meta["scope"],
        state=meta["state"],
        code_challenge=meta["code_challenge"],
        code_challenge_method=meta["code_challenge_method"],
    )
    return HTMLResponse(content=html, status_code=200)


@router.post("/oauth/authorize")
async def oauth_authorize_post(
    client_secret: str = Form(...),
    response_type: str = Form(...),
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    scope: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    code_challenge: str = Form(""),
    code_challenge_method: str = Form("S256"),
):
    """Process authorization and return code"""
    meta = validate_authorize_request(
        response_type=response_type,
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )
    
    # Verify client secret
    if not secrets.compare_digest(client_secret, OAUTH_CLIENT_SECRET):
        raise HTTPException(status_code=401, detail="Invalid client secret")
    
    db = _get_db_session()
    try:
        # Clean up expired codes
        clean_expired_auth_codes(db)

        # Generate authorization code
        code = secrets.token_urlsafe(32)
        auth_code = OAuthAuthorizationCode(
            code=code,
            client_id=meta["client_id"],
            redirect_uri=meta["redirect_uri"],
            scope=meta["scope"],
            code_challenge=meta["code_challenge"],
            state=meta["state"],
            expires_at=datetime.utcnow() + timedelta(seconds=AUTH_CODE_TTL_SECONDS),
        )
        db.add(auth_code)
        db.commit()
    finally:
        db.close()
    
    # Redirect back to client with code
    sep = "&" if "?" in meta["redirect_uri"] else "?"
    redir = f"{meta['redirect_uri']}{sep}code={code}"
    if meta["state"]:
        redir += f"&state={meta['state']}"
    
    return RedirectResponse(url=redir, status_code=302)


@router.post("/oauth/token")
async def oauth_token(request: Request):
    """
    Exchange authorization code for API key (access token).
    
    Supports:
    - PKCE verification (required)
    - Public clients (client_secret optional)
    - Form and JSON bodies
    
    Returns an actual API key from the database as the access_token.
    """
    require_oauth_env()
    
    # Parse body (form or JSON)
    data: Dict[str, Any] = {}
    try:
        form = await request.form()
        data = dict(form)
    except Exception:
        pass
    
    if not data:
        try:
            j = await request.json()
            if isinstance(j, dict):
                data = j
        except Exception:
            pass
    
    grant_type = (data.get("grant_type") or "").strip()
    code = (data.get("code") or "").strip()
    redirect_uri = (data.get("redirect_uri") or "").strip()
    code_verifier = (data.get("code_verifier") or "").strip()
    client_id = (data.get("client_id") or "").strip()
    client_secret = (data.get("client_secret") or "").strip()
    
    # Validate grant type
    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="grant_type must be authorization_code")
    
    # Validate client_id (secret is optional for PKCE)
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Invalid client_id")
    
    if client_secret and not secrets.compare_digest(client_secret, OAUTH_CLIENT_SECRET):
        raise HTTPException(status_code=401, detail="Invalid client_secret")
    
    # Validate required fields
    if not (code and redirect_uri and code_verifier):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: code, redirect_uri, code_verifier"
        )
    
    db = _get_db_session()
    try:
        # Clean up expired codes
        clean_expired_auth_codes(db, commit=False)

        # Retrieve and validate authorization code (one-time use)
        auth_code = (
            db.query(OAuthAuthorizationCode)
            .filter(OAuthAuthorizationCode.code == code)
            .with_for_update()
            .first()
        )
        if not auth_code:
            raise HTTPException(status_code=400, detail="Invalid or expired authorization code")

        meta = {
            "client_id": auth_code.client_id,
            "redirect_uri": auth_code.redirect_uri,
            "scope": auth_code.scope,
            "code_challenge": auth_code.code_challenge,
            "state": auth_code.state,
        }
        is_expired = auth_code.is_expired
        db.delete(auth_code)
        db.commit()

        if is_expired:
            raise HTTPException(status_code=400, detail="Invalid or expired authorization code")

        if redirect_uri != meta["redirect_uri"]:
            raise HTTPException(status_code=400, detail="redirect_uri mismatch")

        # Verify PKCE challenge
        if pkce_s256_challenge(code_verifier) != meta["code_challenge"]:
            raise HTTPException(status_code=400, detail="PKCE verification failed")

        # Find or create user for this client
        user = db.query(User).filter(
            User.email == f"{client_id}@client.memorygate.internal"
        ).first()
        
        if not user:
            # Shouldn't happen (user created during /auth/client), but handle it
            user = User(
                email=f"{client_id}@client.memorygate.internal",
                name=client_id,
                oauth_provider="client_credentials",
                oauth_subject=client_id,
                is_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Generate new API key
        full_key, prefix, key_hash = generate_api_key()
        
        api_key = APIKey(
            user_id=user.id,
            key_prefix=prefix,
            key_hash=key_hash,
            name=f"OAuth {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            scopes=[meta["scope"]],
        )
        db.add(api_key)
        db.commit()
        
        # Return API key as access_token
        return {
            "access_token": full_key,
            "token_type": "bearer",
            "scope": meta["scope"],
        }
        
    finally:
        db.close()

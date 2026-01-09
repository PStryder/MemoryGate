"""
OAuth and User Management Models for MemoryGate

Extends existing models.py with authentication tables.
"""

from datetime import datetime, timedelta
from typing import Optional
import os
import uuid
import secrets

from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from models import Base  # Import existing Base

DB_BACKEND = os.environ.get("DB_BACKEND", "postgres").strip().lower()
JSON_TYPE = JSONB if DB_BACKEND == "postgres" else JSON
UUID_TYPE = UUID(as_uuid=True) if DB_BACKEND == "postgres" else String(36)


def _uuid_default():
    value = uuid.uuid4()
    return value if DB_BACKEND == "postgres" else str(value)


class User(Base):
    """User account - created via OAuth flow"""
    __tablename__ = "users"
    
    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    
    # OAuth provider info
    oauth_provider = Column(String, nullable=False)  # 'google', 'github', etc.
    oauth_subject = Column(String, nullable=False)   # Provider's user ID
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata storage for provider-specific data
    metadata_ = Column("metadata", JSON_TYPE, default=dict, nullable=False)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    # Unique constraint on provider + subject
    __table_args__ = (
        Index('idx_oauth_provider_subject', 'oauth_provider', 'oauth_subject', unique=True),
    )
    
    def __repr__(self):
        return f"<User {self.email} ({self.oauth_provider})>"


class OAuthState(Base):
    """Temporary storage for OAuth state parameter (CSRF protection)"""
    __tablename__ = "oauth_states"
    
    state = Column(String, primary_key=True)
    provider = Column(String, nullable=False)
    redirect_uri = Column(String, nullable=True)
    
    # PKCE support
    code_verifier = Column(String, nullable=True)
    
    # Metadata for state restoration
    metadata_ = Column("metadata", JSON_TYPE, default=dict, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    def __init__(self, provider: str, redirect_uri: Optional[str] = None, 
                 code_verifier: Optional[str] = None, **kwargs):
        self.state = secrets.token_urlsafe(32)
        self.provider = provider
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self.metadata_ = kwargs.get('metadata', {})
        self.created_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(minutes=10)
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class OAuthAuthorizationCode(Base):
    """Short-lived authorization codes for PKCE flow."""
    __tablename__ = "oauth_authorization_codes"

    code = Column(String, primary_key=True)
    client_id = Column(String, nullable=False)
    redirect_uri = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    code_challenge = Column(String, nullable=False)
    state = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class UserSession(Base):
    """Active user sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    user_id = Column(UUID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Session token (secure random, sent as cookie/header)
    token = Column(String, unique=True, nullable=False, index=True)
    
    # Session metadata
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON_TYPE, default=dict, nullable=False)
    
    # Expiry
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Revocation
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    def __init__(self, user_id: UUID, expires_in_hours: int = 24 * 7, **kwargs):
        self.user_id = user_id
        self.token = secrets.token_urlsafe(48)
        self.ip_address = kwargs.get('ip_address')
        self.user_agent = kwargs.get('user_agent')
        self.metadata_ = kwargs.get('metadata', {})
        self.created_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        self.last_activity = datetime.utcnow()
    
    @property
    def is_valid(self) -> bool:
        return (
            not self.is_revoked 
            and datetime.utcnow() < self.expires_at
        )
    
    def refresh_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()


class APIKey(Base):
    """API keys for programmatic access (MCP tools, etc.)"""
    __tablename__ = "api_keys"
    
    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    user_id = Column(UUID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Key components
    key_prefix = Column(String, nullable=False)  # First 8 chars for identification
    key_hash = Column(String, nullable=False, unique=True)  # bcrypt hash of full key
    
    # Key metadata
    name = Column(String, nullable=False)  # User-provided name
    scopes = Column(JSON_TYPE, default=list, nullable=False)  # Permission scopes
    
    # Usage tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Expiry and revocation
    expires_at = Column(DateTime, nullable=True)  # None = no expiry
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
    
    @property
    def is_valid(self) -> bool:
        if self.is_revoked:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def increment_usage(self):
        """Track key usage"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()


# Cleanup utilities
def cleanup_expired_states(session):
    """Remove expired OAuth states"""
    cutoff = datetime.utcnow()
    session.query(OAuthState).filter(OAuthState.expires_at < cutoff).delete()
    session.commit()


def cleanup_expired_sessions(session):
    """Remove expired user sessions"""
    cutoff = datetime.utcnow()
    session.query(UserSession).filter(
        UserSession.expires_at < cutoff,
        UserSession.is_revoked == False
    ).delete()
    session.commit()
